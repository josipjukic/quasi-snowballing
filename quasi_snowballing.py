from numpy.lib.shape_base import split
import pandas as pd
import numpy as np

import itertools
import logging
import random
import time
import sys

from managers import NumpyManager, TorchManager


def run_quasi_snowballing(
    df_load_path,
    df_save_path,
    emb_load_path=None,
    primitives="numpy",
    sim_lb=0.9,
    sim_ub=1.0,
    scaling_factor=1.0,
    log_file=None,
    num_phases=None,
    split_size=0.1,
):
    """
    Quasi-Snowballing pipeline. The process is executed iteratively until it converges,
    i.e. when there aren't any new candidate hits in the current iteration.

    Parameters
    ----------
    df_load_path: str
        Data frame load path.
    df_save_path: str
        Path for storing the data frame with results.
    emb_load_path: str
        Embedding matrix load path.
    primitives: str, default 'numpy'
        'numpy' | 'torch'
    sim_lb: float, default 0.9
        Similarity lower bound. Only sentences with greater
        or equal similarity score will be considered as hits.
    sim_ub: float, default 1.0
        Similarity upper bound. Only sentences with lower
        or equal similarity score will be considered as hits.
    scaling_factor: float, default 1.0
        Scaling factor for similarity bounds. Applied in each phase.
    num_phases: int, default None
        Number of phases for snowballing. If None, the process
        will run until it converges.
    split_size: float, default 0.1
        Approximate size (in GB) of a single candidate embedding split.
    """
    phase_col = "phase"
    seed_parent_col = "seed_parent"
    sim_score_col = "sim_score"
    hit_col = "hit"

    col_list = ["sentence", "seed"]
    df = pd.read_csv(df_load_path, usecols=col_list)

    if not log_file:
        log_file = f'quasi_snowballing_{time.strftime("%Y%m%d-%H%M%S")}.log'

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s | %(levelname)s: %(message)s",
    )

    seed_mask = df.seed
    df[phase_col] = -1
    df.loc[seed_mask, phase_col] = 0

    df[seed_parent_col] = ""
    df[sim_score_col] = ""

    primitives = primitives.lower()
    if primitives == "numpy":
        manager = NumpyManager()
    elif primitives == "torch":
        manager = TorchManager()
    else:
        raise ValueError(
            f"{primitives} primitives are not supported."
            f'Available options: "numpy" | "torch"'
        )

    emb = manager.load(emb_load_path)

    emb_size = round(emb.__sizeof__() / 1024 ** 3, 2)
    # Canidate embedding part should take up around 'split_size' GB.
    n_parts = int(emb_size / split_size)
    n_parts = max(1, n_parts)

    logging.info(f"{len(df.index)} sentences loaded.")
    logging.info(f"Embedding matrix size: {emb_size} GB.")
    logging.info(f"Splitting embedding matrix into: {n_parts} parts.")
    logging.info(f"Each part takes up ~{split_size} GB.")

    sim_lb = np.clip(sim_lb, a_min=0.0, a_max=1.0)
    sim_ub = np.clip(sim_ub, a_min=0.0, a_max=1.0)

    if sim_lb > sim_ub:
        raise ValueError(
            f"Similarity lower bound ({sim_lb}) cannot be"
            f"greater than similarity upper bound ({sim_ub})."
        )

    logging.info(f"Similarity lower bound: {sim_lb}.")
    logging.info(f"Similarity upper bound: {sim_ub}.")
    logging.info(f"Scaling factor: {scaling_factor}.")

    phase = 1
    mask = seed_mask
    new_mask = seed_mask
    start_seed_count = seed_mask.sum()
    prev_count = start_seed_count

    logging.info("Quasi-Snowballing initiated...")

    while True:
        logging.info("-" * 50)
        logging.info(f"Snowballing phase {phase} started...")
        snow = paraphrase_snowballing(
            df=df,
            emb=emb,
            manager=manager,
            seed_mask=mask,
            n_parts=n_parts,
            sim_lb=sim_lb,
            sim_ub=sim_ub,
        )

        logging.info(f"Snowballing phase {phase} finished.")

        # Finish if there aren't any candidates left.
        if not snow:
            logging.info("All candidates retrieved.")
            break

        origs, sims, scores = snow

        # Set seed parent => sentence that was responsible for the
        # similarity hit.
        df.loc[sims, seed_parent_col] = origs

        # Set similarity score of the corresponding hit.
        df.loc[sims, sim_score_col] = scores

        new_mask = expand_seeds(df, mask, sims, phase, phase_col)
        new_count = new_mask.sum()

        logging.info(f"Phase {phase}: added={new_count-prev_count}, total={new_count}")

        if prev_count == new_count:
            break

        phase += 1
        if num_phases and (phase > num_phases):
            break

        sim_lb = scale_bound(sim_lb, scaling_factor)
        sim_ub = scale_bound(sim_ub, scaling_factor)
        prev_count = new_count
        mask = new_mask

    end_seed_count = new_mask.sum()
    df[hit_col] = new_mask
    df.to_csv(df_save_path)

    logging.info("-" * 50)
    logging.info(f'New data frame saved to "{df_save_path}"')
    logging.info(
        f"Started with {start_seed_count} sentences, "
        f"finished with {end_seed_count} sentences."
    )
    logging.info("Quasi-Snowballing finished.")
    logging.info(f'Logging file: "{log_file}".')


def paraphrase_snowballing(
    df, emb, seed_mask, manager, n_parts=1, sim_lb=0.9, sim_ub=1.0, log_iter=10
):

    """
    Set phase information and update seed mask.
    The newly acquired sentences, i.e. candidate hits
    are being upgraded to seeds.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame which contains the sentences for snowballing.
    emb: ndarray
        Sentence embedding matrix. The order of the embeddings
        coincides with the sentence order in the data frame.
    seed_mask: [bool]
        List of Boolean values for easy seed retrieval.
    manager: managers.AbstractManager
        Strategy for handling primitives (e.g. numpy, torch).
    n_parts: int, default 1.0
        Number of parts to split the candidate embeddings into.
        The splitting reduces the amount of memory required.
    sim_lb: float, default 0.9
        Similarity lower bound. Only sentences with greater
        or equal similarity score will be considered as hits.
    sim_ub: float, default 1.0
        Similarity upper bound. Only sentences with lower
        or equal similarity score will be considered as hits.
    log_iter: int, default 10
        Write progress to log every 'log_iter' iterations.

    Returns
    -------
    [int]:
        Original seed indices.
    [int]:
        Newly acquired candidate hits.
    [score]:
        Similarity scores between the original seeds and
        the candidate hits.

    None:
        Returns None if there aren't any available candidates.
    """
    cand_mask = ~seed_mask
    seeds = df[seed_mask]
    cands = df[cand_mask]

    if cands.empty:
        return None

    seed_emb = emb[seed_mask]
    cand_emb = emb[cand_mask]

    splits = manager.splits(cand_emb, n_parts)

    origs = []
    sims = []
    scores = []

    start_index = 0
    for split_ind, cand_split in enumerate(splits, 1):
        cosine_scores = manager.cos_similarity(seed_emb, cand_split)

        hits = manager.where((cosine_scores >= sim_lb) & (cosine_scores <= sim_ub))
        seed_hit_inds, cand_hit_inds = hits
        unique_cand_inds, inds_of_inds = manager.unique(cand_hit_inds)
        unique_seed_inds = seed_hit_inds[inds_of_inds]

        new_scores = cosine_scores[unique_seed_inds, unique_cand_inds]

        unique_cand_inds += start_index
        cand_hits = cands.index[unique_cand_inds]
        seed_hits = seeds.index[unique_seed_inds]

        seed_hits, cand_hits, new_scores = manager.prepare_updates(
            seed_hits, cand_hits, new_scores
        )
        origs.extend(seed_hits)
        sims.extend(cand_hits)
        scores.extend(new_scores)

        start_index += cand_split.shape[0]

        if (split_ind % log_iter == 0) or split_ind == n_parts:
            logging.info(f"{split_ind}/{n_parts} splits processed.")

    return origs, sims, scores


def expand_seeds(df, seed_mask, sims, phase, phase_col):
    """
    Set phase information and update seed mask.
    The newly acquired sentences, i.e. candidate hits
    are being upgraded to seeds.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame which contains the sentences for snowballing.
    seed_mask: [bool]
        List of Boolean values for easy seed retrieval.
    sims: [int]
        List of data frame indices of the candidate hits.
    phase: int
        Current phase of the quasi-snowballing process.
    phase_col: str
        Name of the phase column.

    Returns
    -------
    [bool]:
        Updated seed mask.
    """
    new_mask = df.index.isin(sims)
    df.loc[new_mask, phase_col] = phase
    return seed_mask | new_mask


def scale_bound(bound, scaling_factor):
    """
    Helper method for scaling similarity bounds.
    The updated bound is always in the interval [0, 1].

    Parameters
    ----------
    bound: float
        Similarity bound.
    scaling_factor: float
        Scaling factor for the similarity bound.

    Returns
    -------
    float:
        Scaled similarity bound.
    """
    scaled_bound = bound * scaling_factor
    scaled_bound = np.clip(scaled_bound, a_min=0.0, a_max=1.0)
    return scaled_bound
