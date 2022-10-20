import argparse

from quasi_snowballing import run_quasi_snowballing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quasi-Snowballing: paraphrase mining."
    )

    parser.add_argument("--df-load-path", type=str, help="Data frame path.")
    parser.add_argument("--emb-load-path", type=str, help="Embedding matrix path.")
    parser.add_argument(
        "--df-save-path",
        type=str,
        default="snowballed.csv",
        help="Path for storing the data frame with results.",
    )
    parser.add_argument(
        "--primitives",
        type=str,
        default="numpy",
        help='Embedding type. "numpy" | "torch"',
    )
    parser.add_argument(
        "--sim-lb", type=float, default=0.9, help="Similarity lower bound."
    )
    parser.add_argument(
        "--sim-ub", type=float, default=1.0, help="Similarity upper bound."
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=1.0,
        help="Similarity bound scaling factor.",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Logging file.")
    parser.add_argument(
        "--num-phases", type=int, default=None, help="Number of snowballing phases."
    )
    parser.add_argument(
        "--split-size",
        type=float,
        default=0.1,
        help="Approximate size (in GB) of a single embedding split.",
    )

    args = parser.parse_args()

    run_quasi_snowballing(
        df_load_path=args.df_load_path,
        df_save_path=args.df_save_path,
        emb_load_path=args.emb_load_path,
        primitives=args.primitives,
        sim_lb=args.sim_lb,
        sim_ub=args.sim_ub,
        scaling_factor=args.scaling_factor,
        log_file=args.log_file,
        num_phases=args.num_phases,
        split_size=args.split_size,
    )
