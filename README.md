# Section B — Hybrid Wikipedia Retrieval

**Video Presentation Link:** _TODO_

A hybrid information-retrieval system over a Wikipedia corpus. Given a batch of
natural-language queries, it returns a ranked list of Wikipedia `page_id` values
per query. Only the first 10 results per query are scored, using **mean
NDCG@10**.

## Contents

- [Public Entry Point](#public-entry-point)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Project Layout](#project-layout)
- [Architecture](#architecture)
- [Tuned Runtime Configuration](#tuned-runtime-configuration)
- [Artifacts](#artifacts)
- [Tuning](#tuning)

## Public Entry Point

The grader calls a single function, once, with the full batch of queries:

```python
from main import run

run(queries: list[str]) -> list[list[int]]
```

Each inner list is a ranked set of `page_id` values, most relevant first. Query
embedding and ranking happen inside `run()` and are the timed portion of
grading; index construction is offline and untimed.

## Setup

```bash
cd "Section B"
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

| Package | Used for |
| --- | --- |
| `numpy` | vector math, posting-list scoring |
| `sentence-transformers` | MiniLM document/query embeddings |
| `torch` | embedding backend (CUDA used automatically if available) |
| `nltk` | Snowball stemming of lexical tokens |
| `faiss-cpu` | optional ANN backend (not required by the default retriever) |

The corpus is expected at `data/Wikipedia Entries/` (one JSON file per page,
each with `page_id`, `title`, and `content`). Prebuilt retrieval artifacts ship
under `artifacts/`, so evaluation runs without rebuilding the index.

## How to Run

Evaluate on the 50 public queries (`data/public_queries.json`):

```bash
python scripts/eval_public.py
```

This prints the mean NDCG@10 and the query-phase wall-clock time. To rebuild all
artifacts from the corpus first:

```bash
python scripts/build_index.py
python scripts/eval_public.py
```

`scripts/build_index.py` is offline and not part of query-time grading. At query
time, `main.run()` loads the prebuilt artifacts (cached once per process) and
ranks the provided batch.

> **GPU note:** embedding uses CUDA when PyTorch detects it, otherwise falls back
> to CPU with a printed warning.

## Project Layout

```text
Section B/
├── main.py                  # run() entry point + offline build trigger
├── retrieve.py              # query-time hybrid scoring and fusion (timed)
├── index.py                 # offline artifact build + artifact loaders
├── embed.py                 # MiniLM embedding (documents and queries)
├── utils.py                 # paths, tokenizer, stemming, corpus iteration
├── chunk.py                 # optional passage chunking helpers
├── eval.py                  # NDCG@10 scoring (read-only, do not modify)
├── tune_hyperparameters.py  # offline grid search over fusion weights / BM25
├── requirements.txt
├── scripts/
│   ├── build_index.py       # offline: build artifacts/ from the corpus
│   └── eval_public.py       # self-eval on public queries (mean NDCG@10)
├── artifacts/               # prebuilt retrieval artifacts (see below)
└── data/
    ├── public_queries.json
    └── Wikipedia Entries/   # one JSON file per page
```

## Architecture

Retrieval is a multi-channel hybrid. Each channel produces a per-query score
vector over all pages; scores are min-max normalized per query and combined with
fixed fusion weights in `retrieve.py`.

- **Semantic** — `sentence-transformers/all-MiniLM-L6-v2` embeds the full page
  (`title + content`) at document level. The model's native truncation keeps the
  lead of each article. Query/document similarity is a normalized dot product.
- **BM25F (lead / body)** — two field-specific BM25 indexes split each article at
  `LEAD_WORDS` (the first 200 words are the *lead* field, the rest are *body*),
  scored with independent `k1`/`b` parameters.
- **Lexical BM25** — a page-level BM25 index over the whole article, with a
  term-frequency boost applied to lead-section terms (see below).
- **Coverage** — IDF-weighted fraction of query terms present in a page's lexical
  postings, a soft "how many of my words appear here" prior.
- **Title** — IDF-weighted normalized overlap between query terms and title
  terms.
- **Popularity** — offline title-mention counts across the corpus, log-normalized
  into `popularity_scores.npy`.

Candidates are gathered from the top semantic and top lexical (incl. BM25F)
results, then fully re-scored by the fused combination.

Two experimental query-time stages exist but are disabled in the locked
configuration: **pseudo-relevance feedback** query expansion (`USE_PRF`) and a
**minimum-window proximity** re-rank (`USE_PROXIMITY`).

### Offline indexing optimizations

- **Lead-section TF boost** — terms in the first `LEAD_WORDS` (200) words have
  their term frequency multiplied by `LEAD_TF_BOOST` (3×) in the page-level BM25
  index.
- **Boilerplate filtering** — common Wikipedia structural tokens (`references`,
  `external`, `links`, `category`, `archive`, `isbn`, `doi`, `see`, `main`, …)
  are stripped from all BM25 artifacts.
- **Token expansion** — lexical tokenization applies Snowball stemming, light
  suffix/decade variants, and a curated set of high-value bigram phrases.

## Tuned Runtime Configuration

The locked query-time constants live at the top of `retrieve.py`:

```text
BM25_K1            = 1.20      BM25F_LEAD_K1 = 1.20    BM25F_BODY_K1 = 1.20
BM25_B             = 0.25      BM25F_LEAD_B  = 0.50    BM25F_BODY_B  = 0.25

FUSION_SEMANTIC    = 0.60      # semantic channel
FUSION_BM25F_LEAD  = 0.30      # BM25F lead field
FUSION_COVERAGE    = 0.10      # query-term coverage prior
FUSION_LEXICAL     = 0.00      # page-level BM25 (available, currently off)
FUSION_BM25F_BODY  = 0.00      # BM25F body field (available, currently off)
FUSION_TITLE       = 0.00      # title overlap (available, currently off)
FUSION_POPULARITY  = 0.00      # popularity prior (available, currently off)
```

The active mix is therefore **semantic (0.60) + BM25F-lead (0.30) +
coverage (0.10)**. The remaining channels are fully implemented and left in place
so the fusion can be re-tuned without code changes — see [Tuning](#tuning).

## Artifacts

`build_index.py` writes the following into `artifacts/`:

```text
index_vectors.npy            # one L2-normalized MiniLM vector per page
index_meta.json              # page_id order, titles, model, per-channel metadata
lexicon.json                 # page-level BM25 lexicon (offset, df, idf per token)
post_doc_ids.npy             # page-level BM25 posting doc ids
post_tfs.npy                 # page-level BM25 posting term frequencies
doc_lengths.npy              # page-level BM25 document lengths
popularity_scores.npy        # log-normalized title-mention prior

bm25f_lead_lexicon.json      # BM25F lead-field lexicon
bm25f_lead_post_doc_ids.npy  # BM25F lead-field postings
bm25f_lead_post_tfs.npy
bm25f_lead_doc_lengths.npy
bm25f_body_lexicon.json      # BM25F body-field lexicon
bm25f_body_post_doc_ids.npy  # BM25F body-field postings
bm25f_body_post_tfs.npy
bm25f_body_doc_lengths.npy
```

All per-page arrays are row-aligned with the `page_ids` order stored in
`index_meta.json` (`artifact_version: 6`).

## Tuning

`tune_hyperparameters.py` runs an offline grid search over BM25 `k1`/`b` and
0.05-step fusion weights on the public queries. It reuses the prebuilt artifacts,
precomputes the semantic and title scores once, and does **not** modify
`retrieve.py` — apply any winning configuration by editing the constants above by
hand.

```bash
python tune_hyperparameters.py
```
