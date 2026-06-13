# Section B Hybrid Wikipedia Retrieval

**Video Presentation Link: TODO**

This repository contains the finalized Section B retrieval system. The public
entry point remains:

```python
run(queries: list[str]) -> list[list[int]]
```

For each query, the system returns ranked Wikipedia `page_id` values. The first
10 results are scored by mean NDCG@10.

## Setup

```bash
cd "Section B"
pip install -r requirements.txt
```

The Wikipedia corpus is expected at `data/Wikipedia Entries/`. The repository
also includes prebuilt retrieval artifacts under `artifacts/`, so evaluation can
run without rebuilding the index.

## How To Run

Run the public evaluation:

```bash
python scripts/eval_public.py
```

To rebuild artifacts from scratch before evaluating:

```bash
python scripts/build_index.py
python scripts/eval_public.py
```

`scripts/build_index.py` is offline and not part of query-time grading. Query
time is handled by `main.run()`, which loads the prebuilt artifacts and ranks the
provided query batch.

## Architecture Overview

The final retriever is a four-channel hybrid system:

- **Semantic channel**: `sentence-transformers/all-MiniLM-L6-v2` embeds raw
  `entry_text(record)` at document level. The model's native truncation acts as
  a Wikipedia lead-section extractor.
- **Lexical channel**: BM25 over page-level posting lists, using alphanumeric
  tokens, lightweight suffix variants, and selected high-value phrase tokens.
- **Title channel**: IDF-weighted normalized overlap between query terms and
  title terms.
- **Popularity channel**: offline title-mention counts across the corpus,
  log-normalized into `popularity_scores.npy` and used as a small prior.

Offline indexing includes two Wikipedia-specific optimizations:

- **Lead-section BM25 boost**: terms in the first 100 words receive a 5x term
  frequency boost.
- **Boilerplate filtering**: common Wikipedia structural terms such as
  `references`, `external`, `links`, `category`, and related metadata tokens are
  removed from BM25 artifacts.

The final tuned runtime constants are locked in `retrieve.py`:

```text
BM25_K1 = 1.50
BM25_B = 0.90
semantic = 0.50
lexical = 0.35
coverage = 0.10
title = 0.00
popularity = 0.05
```

## Artifacts

The required runtime artifacts are:

```text
artifacts/index_vectors.npy
artifacts/index_meta.json
artifacts/lexicon.json
artifacts/post_doc_ids.npy
artifacts/post_tfs.npy
artifacts/doc_lengths.npy
artifacts/popularity_scores.npy
```

`index_vectors.npy` stores one normalized MiniLM vector per page. The lexical
files store compact BM25 posting lists and document lengths. The popularity file
is row-aligned with the page IDs in `index_meta.json`.

## Experiments

Additional ablation utilities live in `experiments/`. They are not required for
normal evaluation or submission.
