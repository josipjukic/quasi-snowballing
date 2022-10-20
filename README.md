# Quasi-Snowballing

Quasi-Snowballing pipeline. A tool for paraphrase mining on large datasets.


## General Information & Prerequisites

The pipeline can be used for searching sentence paraphrases.
A data frame with sentences should be prepared in the `csv` format.
The sentences are presumed to be listed in the column `sentence`, along with the
Boolean values which indicate whether sentences are in the seed or not,
under the column `seed`.
Additionally, sentence embeddings should be precomputed and stored in a file as
a `NumPy` array or a `PyTorch` tensor.
To avoid memory issues when calculating similarities, array and tensor splitting
is supported for candidate computation. The number of the parts is dynamically adjusted, depending on
the embedding size.



## Installation

Relevant packages are listed in the `requirements.txt` file.


## Instructions

To run Quasi-Snowballing, use:
```
python run_qsb.py --df-load-path df.csv --df-emb-path emb.npy --primitives numpy --sim-lb 0.95
```

Sentence embeddings should be stored as type defined in the `primitives` argument.
Adjust arguments as needed. This example will run snowballing iterations until
the process converges, i.e., there aren't any hits in the last phase.
The explicit number of phases can be defined with `--num-phases` argument.
Furthermore, the similarity lower bound is set to 0.95.
Generally, the snowballing process will consider candidates as hits if
their similarity with a seed sentence is in the interval `[--sim-lb, --sim-ub]`.
The upper similarity bound is defaulted to 1.
In each iteration, the bounds will be scaled with the `--scaling-factor`.
The scaling factor's default value is 1, thus not influencing the similarity bounds' values.


### Arguments

- **`--df-load-path`**: str, **required**
    - Data frame load path.
- **`--emb-load-path`**: str, **required**
    - Embedding matrix load path.
- `--df-save-path`: str, default 'snowballed.csv'
    - Path for storing the data frame with results.
- `--primitives`: str, default 'numpy'
    - 'numpy' | 'torch'
- `--sim-lb`: float, default 0.9
    - Similarity lower bound. Only sentences with greater
      or equal similarity score will be considered as hits.
- `--sim-ub`: float, default 1.0
    - Similarity upper bound. Only sentences with lower or equal
     similarity score will be considered as hits.
- `--scaling-factor`: float, default 1.0
    - Scaling factor for similarity bounds. Applied in each phase.
- `--num-phases`: int, default None
    - Number of snowballing phases. If None, the process
      will run until it converges.
- `--split-size`: float, default 0.1
    - Approximate size (in GB) of a single embedding split.


### Results

The experiment will be logged to the defined logging file
(it is going to be automatically generated if not provided).
The end result will be a data frame with sentences (the original `sentence` and `seed` columns) and the newly acquired
paraphrases annotated with:
- `phase` - snowballing iteration in which they were matched.
- `seed_parent` - sentence parent from the seed set in the corresponding iteration.
- `sim_score` - similarity score of the corresponding candidate hit.
- `hit` - indicates if sentence is a hit or not (it can be a seed sentence or a paraphrase).


### Limitations

Currently, it is required for the embeddings to fit in the memory with some overhead for similarity computations.
For example, if you use embeddings that require 40 GB of RAM, you should be able to provide 20 GB of RAM in addition.
Furthermore, if GPU is used, the size of the embeddings is limited by the GPU memory.