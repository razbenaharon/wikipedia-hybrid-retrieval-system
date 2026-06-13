"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import POPULARITY_SCORES_NAME, load_index, load_lexical_artifacts, load_meta
from utils import ARTIFACTS_DIR, K_EVAL, tokenize_text

SEMANTIC_CANDIDATES = 800
LEXICAL_CANDIDATES = 800
BM25_K1 = 1.50
BM25_B = 0.90
FUSION_SEMANTIC = 0.50
FUSION_LEXICAL = 0.35
FUSION_COVERAGE = 0.10
FUSION_TITLE = 0.00
FUSION_POPULARITY = 0.05


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
    lexical_rows: List[np.ndarray] = []
    coverage_rows: List[np.ndarray] = []
    for query in queries:
        lexical_scores, coverage_scores = _lexical_scores(
            query,
            artifacts,
            k1=BM25_K1,
            b=BM25_B,
        )
        lexical_rows.append(lexical_scores)
        coverage_rows.append(coverage_scores)

    lexical_scores_matrix = np.vstack(lexical_rows).astype(np.float32, copy=False)
    coverage_scores_matrix = np.vstack(coverage_rows).astype(np.float32, copy=False)
    semantic_norm = _normalize_rows(semantic_scores.astype(np.float32, copy=False))
    lexical_norm = _normalize_rows(lexical_scores_matrix)

    ranked: List[List[int]] = []
    for i, query in enumerate(queries):
        semantic_row = semantic_norm[i]
        lexical_scores = lexical_norm[i]
        coverage_scores = coverage_scores_matrix[i]
        candidate_ids = _candidate_doc_ids(semantic_row, lexical_scores, top_k)
        ranked.append(
            _rank_candidates(
                query,
                semantic_row,
                lexical_scores,
                coverage_scores,
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
    bm25_scores = np.zeros(len(artifacts.page_ids), dtype=np.float32)
    coverage_scores = np.zeros(len(artifacts.page_ids), dtype=np.float32)
    query_counts = Counter(tokenize_text(query, expand=True))
    if not query_counts or artifacts.avg_doc_length <= 0:
        return bm25_scores, coverage_scores

    doc_lengths = artifacts.doc_lengths
    avg_len = artifacts.avg_doc_length
    total_idf = 0.0
    for token, query_tf in query_counts.items():
        posting = artifacts.lexicon.get(token)
        if posting is None:
            continue
        offset, length, idf = posting
        total_idf += float(idf)
        doc_ids = artifacts.post_doc_ids[offset : offset + length]
        term_freqs = artifacts.post_tfs[offset : offset + length].astype(np.float32)
        denom = term_freqs + k1 * (1.0 - b + b * doc_lengths[doc_ids] / avg_len)
        increments = float(query_tf) * float(idf) * (term_freqs * (k1 + 1.0) / denom)
        np.add.at(bm25_scores, doc_ids, increments.astype(np.float32, copy=False))
        np.add.at(coverage_scores, doc_ids, np.float32(idf))

    if total_idf > 0.0:
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
    combined = (
        FUSION_LEXICAL * lexical_part
        + FUSION_SEMANTIC * semantic_part
        + FUSION_COVERAGE * coverage_part
        + FUSION_TITLE * title_part
        + FUSION_POPULARITY * popularity_part
    )
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
