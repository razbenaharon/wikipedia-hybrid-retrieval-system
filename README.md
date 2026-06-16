# Section B - Wikipedia Retrieval

Video presentation: [Google Drive](https://drive.google.com/file/d/1OTeE8505G-K7uFsKdHm5gg1vetxAvULA/view?usp=sharing)

This is my solution for Section B. The program gets a list of search queries and
returns Wikipedia `page_id` results for each query. The grader only checks the
top 10 results, so the main goal was to get good `NDCG@10` while keeping the
query time reasonable.

## How the Project Works

The main entry point is:

```python
from main import run

results = run(queries)
```

`queries` is a list of strings, and the output is a list of lists. Each inner
list contains page ids ordered from most relevant to least relevant.

The index is built offline and saved in the `artifacts/` folder. During the
actual run, the code loads these files and only does the query embedding and
ranking part.

## Setup

From the project folder, install the requirements:

```bash
pip install -r requirements.txt
```

The main packages I used are:

- `sentence-transformers` for MiniLM embeddings
- `numpy` for the scoring calculations
- `nltk` for stemming words
- `faiss-cpu`, which is included in the requirements although the current main
  retriever does not depend on it

The Wikipedia files should be in:

```text
data/Wikipedia Entries/
```

Each entry is expected to have `page_id`, `title`, and `content`.

## Running

To test the public queries:

```bash
python scripts/eval_public.py
```

To rebuild the artifacts first:

```bash
python scripts/build_index.py
python scripts/eval_public.py
```

Building the index is not part of the timed query stage. It just creates the
files that are later loaded by `run()`.

## Retrieval Method

I used a hybrid retrieval approach:

- semantic similarity with `sentence-transformers/all-MiniLM-L6-v2`
- BM25-style lexical scoring
- a separate BM25 score for the beginning of the article, since the first part
  of a Wikipedia page usually contains the most important information
- a small coverage score that checks how many query terms appear in a page

The final score is a weighted combination of these parts. The current weights in
`retrieve.py` are:

```text
semantic:   0.60
BM25 lead:  0.30
coverage:   0.10
```

I also kept some extra scoring code in the project, like full-article BM25,
title overlap, popularity, pseudo-relevance feedback, and proximity scoring.
Those are currently turned off because this combination gave better results on
the public queries.

## Main Files

```text
main.py                 run() entry point for the grader
retrieve.py             query-time ranking code
index.py                builds and loads index artifacts
embed.py                document and query embeddings
utils.py                tokenization, paths, and corpus helpers
eval.py                 NDCG@10 evaluation code
tune_hyperparameters.py offline tuning script
scripts/build_index.py  rebuilds artifacts
scripts/eval_public.py  evaluates public queries
artifacts/              saved index files
data/                   queries and Wikipedia entries
```

## Notes

- CUDA is used automatically if PyTorch detects a GPU.
- If there is no GPU, the code still works on CPU, but embedding can be slower.
- The public tuning was done with `tune_hyperparameters.py`, then I copied the
  best constants into `retrieve.py`.
