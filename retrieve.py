"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import POPULARITY_SCORES_NAME, load_index, load_lexical_artifacts, load_meta
from utils import ARTIFACTS_DIR, K_EVAL, STOPWORDS, entry_text, iter_entries, tokenize_text

SEMANTIC_CANDIDATES = 800
LEXICAL_CANDIDATES = 800
PRF_SEMANTIC_DOCS = 3
PRF_EXPANSION_TERMS = 5
PROXIMITY_CANDIDATES = 100
BM25_K1 = 1.20
BM25_B = 0.25
USE_BM25F = True
BM25F_LEAD_K1 = 1.20
BM25F_LEAD_B = 0.50
BM25F_BODY_K1 = 1.20
BM25F_BODY_B = 0.25
FUSION_SEMANTIC = 0.60
FUSION_LEXICAL = 0.00
FUSION_COVERAGE = 0.10
FUSION_TITLE = 0.00
FUSION_POPULARITY = 0.00
FUSION_BM25F_LEAD = 0.30
FUSION_BM25F_BODY = 0.00
USE_PROXIMITY = False
FUSION_PROXIMITY = 0.00
USE_PRF = False


@dataclass
class RetrievalArtifacts:
    root: Path
    vectors: np.ndarray
    page_ids: List[int]
    title_token_sets: List[set[str]]
    lexicon: dict[str, tuple[int, int, float]]
    post_doc_ids: np.ndarray
    post_tfs: np.ndarray
    doc_lengths: np.ndarray
    avg_doc_length: float
    popularity_scores: np.ndarray
    bm25f_lead_lexicon: Optional[dict[str, tuple[int, int, float]]] = None
    bm25f_lead_post_doc_ids: Optional[np.ndarray] = None
    bm25f_lead_post_tfs: Optional[np.ndarray] = None
    bm25f_lead_doc_lengths: Optional[np.ndarray] = None
    bm25f_lead_avg_doc_length: float = 0.0
    bm25f_body_lexicon: Optional[dict[str, tuple[int, int, float]]] = None
    bm25f_body_post_doc_ids: Optional[np.ndarray] = None
    bm25f_body_post_tfs: Optional[np.ndarray] = None
    bm25f_body_doc_lengths: Optional[np.ndarray] = None
    bm25f_body_avg_doc_length: float = 0.0
    document_texts: Optional[List[str]] = None


_ARTIFACT_CACHE: RetrievalArtifacts | None = None


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Hybrid retrieval: MiniLM semantic scores plus BM25 lexical scores and a
    small title-match bonus. Artifacts are loaded once per Python process.
    """
    if not queries:
        return []

    artifacts = _get_artifacts(artifacts_dir)
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    semantic_scores = _semantic_page_scores(query_vectors, artifacts)
    retrieval_queries = (
        [
            _expand_query_with_prf(query, semantic_scores[i], artifacts)
            for i, query in enumerate(queries)
        ]
        if USE_PRF
        else queries
    )
    lexical_rows: List[np.ndarray] = []
    coverage_rows: List[np.ndarray] = []
    bm25f_lead_rows: List[np.ndarray] = []
    bm25f_body_rows: List[np.ndarray] = []
    for query in retrieval_queries:
        lexical_scores, coverage_scores = _lexical_scores(
            query,
            artifacts,
            k1=BM25_K1,
            b=BM25_B,
        )
        lexical_rows.append(lexical_scores)
        coverage_rows.append(coverage_scores)
        if USE_BM25F:
            lead_scores, body_scores = _bm25f_scores(query, artifacts)
            bm25f_lead_rows.append(lead_scores)
            bm25f_body_rows.append(body_scores)

    lexical_scores_matrix = np.vstack(lexical_rows).astype(np.float32, copy=False)
    coverage_scores_matrix = np.vstack(coverage_rows).astype(np.float32, copy=False)
    semantic_norm = _normalize_rows(semantic_scores.astype(np.float32, copy=False))
    lexical_norm = _normalize_rows(lexical_scores_matrix)
    if USE_BM25F:
        bm25f_lead_norm = _normalize_rows(np.vstack(bm25f_lead_rows).astype(np.float32))
        bm25f_body_norm = _normalize_rows(np.vstack(bm25f_body_rows).astype(np.float32))
    else:
        bm25f_lead_norm = np.zeros_like(lexical_norm)
        bm25f_body_norm = np.zeros_like(lexical_norm)

    ranked: List[List[int]] = []
    for i, query in enumerate(queries):
        semantic_row = semantic_norm[i]
        lexical_scores = lexical_norm[i]
        coverage_scores = coverage_scores_matrix[i]
        lexical_candidates = lexical_scores
        if USE_BM25F:
            lexical_candidates = (
                lexical_candidates + bm25f_lead_norm[i] + bm25f_body_norm[i]
            )
        candidate_ids = _candidate_doc_ids(semantic_row, lexical_candidates, top_k)
        ranked.append(
            _rank_candidates(
                query,
                semantic_row,
                lexical_scores,
                coverage_scores,
                bm25f_lead_norm[i],
                bm25f_body_norm[i],
                candidate_ids,
                artifacts,
                top_k,
            )
        )
    return ranked


def _get_artifacts(artifacts_dir: Optional[Path]) -> RetrievalArtifacts:
    global _ARTIFACT_CACHE

    root = artifacts_dir or ARTIFACTS_DIR
    root = root.resolve()
    if _ARTIFACT_CACHE is not None and _ARTIFACT_CACHE.root == root:
        if (USE_PRF or USE_PROXIMITY) and _ARTIFACT_CACHE.document_texts is None:
            _ARTIFACT_CACHE.document_texts = _load_document_texts(
                _ARTIFACT_CACHE.page_ids
            )
        return _ARTIFACT_CACHE

    vectors, page_ids = load_index(root)
    meta = load_meta(root)
    titles = [str(title) for title in meta.get("titles", [""] * len(page_ids))]
    lexical = load_lexical_artifacts(root)
    popularity_scores = _load_popularity_scores(root, meta, len(page_ids))
    _ARTIFACT_CACHE = RetrievalArtifacts(
        root=root,
        vectors=vectors,
        page_ids=page_ids,
        title_token_sets=[set(tokenize_text(title, expand=True)) for title in titles],
        lexicon=lexical.lexicon,
        post_doc_ids=lexical.post_doc_ids,
        post_tfs=lexical.post_tfs,
        doc_lengths=lexical.doc_lengths.astype(np.float32, copy=False),
        avg_doc_length=lexical.avg_doc_length,
        popularity_scores=popularity_scores,
        bm25f_lead_lexicon=lexical.bm25f_lead_lexicon,
        bm25f_lead_post_doc_ids=lexical.bm25f_lead_post_doc_ids,
        bm25f_lead_post_tfs=lexical.bm25f_lead_post_tfs,
        bm25f_lead_doc_lengths=(
            lexical.bm25f_lead_doc_lengths.astype(np.float32, copy=False)
            if lexical.bm25f_lead_doc_lengths is not None
            else None
        ),
        bm25f_lead_avg_doc_length=lexical.bm25f_lead_avg_doc_length,
        bm25f_body_lexicon=lexical.bm25f_body_lexicon,
        bm25f_body_post_doc_ids=lexical.bm25f_body_post_doc_ids,
        bm25f_body_post_tfs=lexical.bm25f_body_post_tfs,
        bm25f_body_doc_lengths=(
            lexical.bm25f_body_doc_lengths.astype(np.float32, copy=False)
            if lexical.bm25f_body_doc_lengths is not None
            else None
        ),
        bm25f_body_avg_doc_length=lexical.bm25f_body_avg_doc_length,
        document_texts=_load_document_texts(page_ids) if USE_PRF or USE_PROXIMITY else None,
    )
    return _ARTIFACT_CACHE


def _load_popularity_scores(root: Path, meta: dict, expected_len: int) -> np.ndarray:
    popularity_meta = meta.get("popularity", {})
    scores_path = root / popularity_meta.get("scores", POPULARITY_SCORES_NAME)
    if not scores_path.exists():
        return np.zeros(expected_len, dtype=np.float32)

    scores = np.load(scores_path).astype(np.float32, copy=False)
    if scores.shape != (expected_len,):
        return np.zeros(expected_len, dtype=np.float32)
    return scores


def _lexical_scores(
    query: str,
    artifacts: RetrievalArtifacts,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> tuple[np.ndarray, np.ndarray]:
    return _bm25_scores_from_postings(
        query,
        artifacts.lexicon,
        artifacts.post_doc_ids,
        artifacts.post_tfs,
        artifacts.doc_lengths,
        artifacts.avg_doc_length,
        len(artifacts.page_ids),
        k1=k1,
        b=b,
        include_coverage=True,
    )


def _bm25f_scores(
    query: str,
    artifacts: RetrievalArtifacts,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        artifacts.bm25f_lead_lexicon is None
        or artifacts.bm25f_lead_post_doc_ids is None
        or artifacts.bm25f_lead_post_tfs is None
        or artifacts.bm25f_lead_doc_lengths is None
        or artifacts.bm25f_body_lexicon is None
        or artifacts.bm25f_body_post_doc_ids is None
        or artifacts.bm25f_body_post_tfs is None
        or artifacts.bm25f_body_doc_lengths is None
    ):
        zeros = np.zeros(len(artifacts.page_ids), dtype=np.float32)
        return zeros, zeros

    lead_scores, _ = _bm25_scores_from_postings(
        query,
        artifacts.bm25f_lead_lexicon,
        artifacts.bm25f_lead_post_doc_ids,
        artifacts.bm25f_lead_post_tfs,
        artifacts.bm25f_lead_doc_lengths,
        artifacts.bm25f_lead_avg_doc_length,
        len(artifacts.page_ids),
        k1=BM25F_LEAD_K1,
        b=BM25F_LEAD_B,
        include_coverage=False,
    )
    body_scores, _ = _bm25_scores_from_postings(
        query,
        artifacts.bm25f_body_lexicon,
        artifacts.bm25f_body_post_doc_ids,
        artifacts.bm25f_body_post_tfs,
        artifacts.bm25f_body_doc_lengths,
        artifacts.bm25f_body_avg_doc_length,
        len(artifacts.page_ids),
        k1=BM25F_BODY_K1,
        b=BM25F_BODY_B,
        include_coverage=False,
    )
    return lead_scores, body_scores


def _bm25_scores_from_postings(
    query: str,
    lexicon: dict[str, tuple[int, int, float]],
    post_doc_ids: np.ndarray,
    post_tfs: np.ndarray,
    doc_lengths: np.ndarray,
    avg_doc_length: float,
    num_docs: int,
    *,
    k1: float,
    b: float,
    include_coverage: bool,
) -> tuple[np.ndarray, np.ndarray]:
    bm25_scores = np.zeros(num_docs, dtype=np.float32)
    coverage_scores = np.zeros(num_docs, dtype=np.float32)
    query_counts = Counter(tokenize_text(query, expand=True))
    if not query_counts or avg_doc_length <= 0:
        return bm25_scores, coverage_scores

    total_idf = 0.0
    for token, query_tf in query_counts.items():
        posting = lexicon.get(token)
        if posting is None:
            continue
        offset, length, idf = posting
        total_idf += float(idf)
        doc_ids = post_doc_ids[offset : offset + length]
        term_freqs = post_tfs[offset : offset + length].astype(np.float32)
        denom = term_freqs + k1 * (1.0 - b + b * doc_lengths[doc_ids] / avg_doc_length)
        increments = float(query_tf) * float(idf) * (term_freqs * (k1 + 1.0) / denom)
        np.add.at(bm25_scores, doc_ids, increments.astype(np.float32, copy=False))
        if include_coverage:
            np.add.at(coverage_scores, doc_ids, np.float32(idf))

    if include_coverage and total_idf > 0.0:
        coverage_scores /= np.float32(total_idf)
    return bm25_scores, coverage_scores


def _semantic_page_scores(
    query_vectors: np.ndarray,
    artifacts: RetrievalArtifacts,
) -> np.ndarray:
    return query_vectors @ artifacts.vectors.T


def _candidate_doc_ids(
    semantic_scores: np.ndarray,
    lexical_scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    semantic_limit = min(len(semantic_scores), max(SEMANTIC_CANDIDATES, top_k))
    semantic_ids = _top_indices(semantic_scores, semantic_limit)

    nonzero_lexical = np.flatnonzero(lexical_scores > 0)
    if len(nonzero_lexical) > LEXICAL_CANDIDATES:
        local = _top_indices(lexical_scores[nonzero_lexical], LEXICAL_CANDIDATES)
        lexical_ids = nonzero_lexical[local]
    else:
        lexical_ids = nonzero_lexical

    if len(lexical_ids) == 0:
        return semantic_ids
    return np.unique(np.concatenate([semantic_ids, lexical_ids]))


def _rank_candidates(
    query: str,
    semantic_scores: np.ndarray,
    lexical_scores: np.ndarray,
    coverage_scores: np.ndarray,
    bm25f_lead_scores: np.ndarray,
    bm25f_body_scores: np.ndarray,
    candidate_ids: np.ndarray,
    artifacts: RetrievalArtifacts,
    top_k: int,
) -> List[int]:
    if len(candidate_ids) == 0:
        return []

    semantic_part = semantic_scores[candidate_ids]
    lexical_part = lexical_scores[candidate_ids]
    coverage_part = coverage_scores[candidate_ids]
    title_part = _title_bonus(query, candidate_ids, artifacts)
    popularity_part = artifacts.popularity_scores[candidate_ids]
    bm25f_lead_part = bm25f_lead_scores[candidate_ids]
    bm25f_body_part = bm25f_body_scores[candidate_ids]
    combined = (
        FUSION_LEXICAL * lexical_part
        + FUSION_SEMANTIC * semantic_part
        + FUSION_COVERAGE * coverage_part
        + FUSION_TITLE * title_part
        + FUSION_POPULARITY * popularity_part
        + FUSION_BM25F_LEAD * bm25f_lead_part
        + FUSION_BM25F_BODY * bm25f_body_part
    )
    if USE_PROXIMITY:
        base_order = np.argsort(-combined, kind="mergesort")[
            : min(PROXIMITY_CANDIDATES, len(combined))
        ]
        prox_candidate_ids = candidate_ids[base_order]
        proximity_part = _proximity_scores(query, prox_candidate_ids, artifacts)
        combined = combined[base_order] + FUSION_PROXIMITY * proximity_part
        candidate_ids = prox_candidate_ids

    order = np.argsort(-combined, kind="mergesort")

    ids: List[int] = []
    seen: set[int] = set()
    for idx in candidate_ids[order]:
        page_id = artifacts.page_ids[int(idx)]
        if page_id in seen:
            continue
        seen.add(page_id)
        ids.append(page_id)
        if len(ids) >= top_k:
            break
    return ids


def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    if limit >= len(scores):
        return np.argsort(-scores)
    partition = np.argpartition(-scores, limit - 1)[:limit]
    return partition[np.argsort(-scores[partition])]


def _normalize_rows(scores: np.ndarray) -> np.ndarray:
    """Vectorized per-query min-max normalization for score matrices."""
    row_min = scores.min(axis=1, keepdims=True)
    row_max = scores.max(axis=1, keepdims=True)
    denom = row_max - row_min
    return np.divide(
        scores - row_min,
        denom,
        out=np.zeros_like(scores, dtype=np.float32),
        where=denom > 0,
    )


def _load_document_texts(page_ids: List[int]) -> List[str]:
    records_by_page_id = {int(record["page_id"]): record for record in iter_entries()}
    return [entry_text(records_by_page_id[page_id]) for page_id in page_ids]


def _expand_query_with_prf(
    query: str,
    semantic_scores: np.ndarray,
    artifacts: RetrievalArtifacts,
) -> str:
    if artifacts.document_texts is None:
        return query

    top_doc_idxs = _top_indices(
        semantic_scores,
        min(PRF_SEMANTIC_DOCS, len(semantic_scores)),
    )
    original_terms = set(tokenize_text(query, expand=False))
    expansion_scores: dict[str, float] = {}
    for doc_idx in top_doc_idxs:
        text = artifacts.document_texts[int(doc_idx)]
        for token in tokenize_text(
            text,
            expand=False,
            include_bigrams=False,
            include_char_ngrams=False,
        ):
            if token in STOPWORDS or token in original_terms:
                continue
            posting = artifacts.lexicon.get(token)
            if posting is None:
                continue
            expansion_scores[token] = max(expansion_scores.get(token, 0.0), posting[2])

    if not expansion_scores:
        return query

    expansion_terms = [
        token
        for token, _ in sorted(
            expansion_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:PRF_EXPANSION_TERMS]
    ]
    return " ".join([query, *expansion_terms])


def _proximity_scores(
    query: str,
    candidate_ids: np.ndarray,
    artifacts: RetrievalArtifacts,
) -> np.ndarray:
    if artifacts.document_texts is None:
        return np.zeros(len(candidate_ids), dtype=np.float32)

    query_terms = set(
        tokenize_text(
            query,
            expand=False,
            include_bigrams=False,
            include_char_ngrams=False,
        )
    )
    if not query_terms:
        return np.zeros(len(candidate_ids), dtype=np.float32)

    scores = np.zeros(len(candidate_ids), dtype=np.float32)
    for row_idx, doc_idx in enumerate(candidate_ids):
        text = artifacts.document_texts[int(doc_idx)]
        doc_tokens = tokenize_text(
            text,
            expand=False,
            include_bigrams=False,
            include_char_ngrams=False,
        )
        scores[row_idx] = _minimum_window_score(doc_tokens, query_terms)
    return scores


def _minimum_window_score(tokens: List[str], query_terms: set[str]) -> float:
    positions_by_term: dict[str, int] = {}
    best_window = None
    left = 0
    relevant_terms = [token for token in tokens if token in query_terms]
    if not relevant_terms:
        return 0.0

    for right, token in enumerate(tokens):
        if token not in query_terms:
            continue
        positions_by_term[token] = positions_by_term.get(token, 0) + 1
        while left <= right:
            left_token = tokens[left]
            if left_token not in query_terms:
                left += 1
                continue
            if positions_by_term.get(left_token, 0) <= 1:
                break
            positions_by_term[left_token] -= 1
            left += 1
        coverage = len(positions_by_term) / len(query_terms)
        if coverage >= 0.6:
            window = right - left + 1
            if best_window is None or window < best_window:
                best_window = window

    if best_window is None:
        return len(set(relevant_terms)) / len(query_terms) * 0.25
    return float(min(1.0, len(query_terms) / max(best_window, len(query_terms))))


def _title_bonus(
    query: str,
    candidate_ids: np.ndarray,
    artifacts: RetrievalArtifacts,
) -> np.ndarray:
    query_terms = set(tokenize_text(query, expand=True))
    if not query_terms:
        return np.zeros(len(candidate_ids), dtype=np.float32)

    query_idfs = {
        token: artifacts.lexicon[token][2]
        for token in query_terms
        if token in artifacts.lexicon
    }
    total_idf = float(sum(query_idfs.values()))
    if total_idf <= 0.0:
        return np.zeros(len(candidate_ids), dtype=np.float32)

    bonuses = np.zeros(len(candidate_ids), dtype=np.float32)
    for i, doc_idx in enumerate(candidate_ids):
        title_terms = artifacts.title_token_sets[int(doc_idx)]
        if title_terms:
            overlap_idf = sum(
                idf for token, idf in query_idfs.items() if token in title_terms
            )
            bonuses[i] = min(1.0, overlap_idf / total_idf)
    return bonuses
