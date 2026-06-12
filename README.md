# Section B — Retrieval pipeline

This project implements a hybrid Wikipedia page retriever for the Section B
autograder. The public entry point is unchanged:

```python
run(queries: list[str]) -> list[list[int]]
```

For each query, the system returns ranked `page_id` values. Only the first 10
IDs are scored.

## Setup

```bash
cd "Section B"
pip install -r requirements.txt
```

Corpus lives at **`data/Wikipedia Entries/`** (included in the handout).

## Retrieval approach

The retriever combines two signals:

- semantic similarity from `sentence-transformers/all-MiniLM-L6-v2` over full
  page title/content text
- BM25-style lexical matching over alphanumeric tokens, including numbers and
  simple suffix variants
- selected phrase tokens for high-value clue pairs such as roles, locations,
  research terms, and artifact/transport phrases

The query-time code loads prebuilt artifacts, embeds the query batch, retrieves
semantic and lexical candidates, then reranks them with a weighted blend of
lexical score, semantic score, and a small title-match bonus.

## Build artifacts (offline, not timed)

Run once locally to create `artifacts/`. **Submit these files** in your repo; staff do not rebuild the index at grading time.

```bash
python scripts/build_index.py
```

Required artifact files:

```text
artifacts/index_vectors.npy
artifacts/index_meta.json
artifacts/lexicon.json
artifacts/post_doc_ids.npy
artifacts/post_tfs.npy
artifacts/doc_lengths.npy
```

`index_vectors.npy` stores one normalized MiniLM vector per page. The lexical
files store page-level BM25 posting lists and document lengths.

## Public self-test

After building, verify a fresh run loads your submitted artifacts (no rebuild):

```bash
python scripts/eval_public.py
```

## Submit

Public GitHub repo with this code and the required `artifacts/` directory. A
fresh clone should be able to install requirements and run
`python scripts/eval_public.py` without running `scripts/build_index.py`.
