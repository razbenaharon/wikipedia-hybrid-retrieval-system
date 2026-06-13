"""Tune hybrid retrieval fusion weights and BM25 parameters.

This script does not rebuild artifacts and does not modify retrieve.py. It
precomputes semantic and title scores once, then sweeps BM25 k1/b settings and
0.05-step fusion weights on the public queries.
"""
from __future__ import annotations

import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from embed import embed_queries
from eval import K_EVAL, load_query_file, mean_ndcg_at_k
from retrieve import (
    _get_artifacts,
    _lexical_scores,
    _normalize_rows,
    _semantic_page_scores,
)
from utils import PUBLIC_QUERIES_PATH, tokenize_text

GRID_UNITS = 20
BM25_K1_GRID = (1.2, 1.5, 1.8)
BM25_B_GRID = (0.25, 0.5, 0.75, 0.9)
USE_POPULARITY_IN_TUNING = False


class CoreScores:
    def __init__(
        self,
        semantic_norm: np.ndarray,
        title_scores: np.ndarray,
        popularity_scores: np.ndarray | None,
    ) -> None:
        self.semantic_norm = semantic_norm
        self.title_scores = title_scores
        self.popularity_scores = popularity_scores


class LexicalScores:
    def __init__(self, lexical_norm: np.ndarray, coverage_scores: np.ndarray) -> None:
        self.lexical_norm = lexical_norm
        self.coverage_scores = coverage_scores


class Weights:
    def __init__(
        self,
        semantic: float,
        lexical: float,
        coverage: float,
        title: float,
        popularity: float = 0.0,
    ) -> None:
        self.semantic = semantic
        self.lexical = lexical
        self.coverage = coverage
        self.title = title
        self.popularity = popularity


class TuneResult:
    def __init__(
        self,
        score: float,
        weights: Weights,
        k1: float,
        b: float,
    ) -> None:
        self.score = score
        self.weights = weights
        self.k1 = k1
        self.b = b


def main() -> None:
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [row["query"] for row in rows]
    ground_truth = [row["relevant_page_ids"] for row in rows]

    print(f"Loaded public queries: {len(queries)}")
    artifacts = _get_artifacts(None)

    t0 = time.perf_counter()
    core_scores = precompute_core_scores(queries, artifacts)
    print(f"semantic_scores shape: {core_scores.semantic_norm.shape}")
    print(f"title_scores shape:    {core_scores.title_scores.shape}")
    print(f"Core precompute time: {time.perf_counter() - t0:.2f}s")

    best = TuneResult(
        score=-1.0,
        weights=Weights(semantic=0.0, lexical=0.0, coverage=0.0, title=0.0),
        k1=BM25_K1_GRID[0],
        b=BM25_B_GRID[0],
    )

    page_ids = np.asarray(artifacts.page_ids, dtype=np.int64)
    for k1 in BM25_K1_GRID:
        for b in BM25_B_GRID:
            t_lex = time.perf_counter()
            lexical_scores = precompute_lexical_scores(queries, artifacts, k1=k1, b=b)
            print(
                f"BM25 k1={k1:.2f} b={b:.2f}: "
                f"lexical_scores shape={lexical_scores.lexical_norm.shape}, "
                f"precompute={time.perf_counter() - t_lex:.2f}s"
            )
            result = run_grid_search(
                core_scores,
                lexical_scores,
                ground_truth,
                page_ids,
                k1=k1,
                b=b,
                current_best=best.score,
            )
            if result.score > best.score:
                best = result

    print()
    print("Best result")
    print(f"mean_ndcg@10={best.score:.6f}")
    weights_text = (
        "weights="
        f"semantic={best.weights.semantic:.2f}, "
        f"lexical={best.weights.lexical:.2f}, "
        f"coverage={best.weights.coverage:.2f}, "
        f"title={best.weights.title:.2f}, "
    )
    if USE_POPULARITY_IN_TUNING:
        weights_text += f"popularity={best.weights.popularity:.2f}, "
    weights_text += f"k1={best.k1:.2f}, b={best.b:.2f}"
    print(weights_text)
    print()
    print("Update retrieve.py constants to:")
    print(f"BM25_K1 = {best.k1:.2f}")
    print(f"BM25_B = {best.b:.2f}")
    if USE_POPULARITY_IN_TUNING:
        print(f"FUSION_POPULARITY = {best.weights.popularity:.2f}")
    print("combined = (")
    print(f"    {best.weights.lexical:.2f} * lexical_part")
    print(f"    + {best.weights.semantic:.2f} * semantic_part")
    print(f"    + {best.weights.coverage:.2f} * coverage_part")
    print(f"    + {best.weights.title:.2f} * title_part")
    if USE_POPULARITY_IN_TUNING:
        print(f"    + {best.weights.popularity:.2f} * popularity_part")
    print(")")


def precompute_core_scores(queries: List[str], artifacts) -> CoreScores:
    query_vectors = embed_queries(queries)
    semantic_scores = _semantic_page_scores(query_vectors, artifacts)
    title_scores = _title_score_matrix(queries, artifacts)
    popularity_scores = None
    if USE_POPULARITY_IN_TUNING:
        popularity_scores = np.broadcast_to(
            artifacts.popularity_scores,
            (len(queries), len(artifacts.page_ids)),
        ).astype(np.float32, copy=False)
    return CoreScores(
        semantic_norm=_normalize_rows(semantic_scores.astype(np.float32, copy=False)),
        title_scores=title_scores,
        popularity_scores=popularity_scores,
    )


def precompute_lexical_scores(
    queries: List[str],
    artifacts,
    *,
    k1: float,
    b: float,
) -> LexicalScores:
    lexical_rows: List[np.ndarray] = []
    coverage_rows: List[np.ndarray] = []
    for query in queries:
        lexical_scores, coverage_scores = _lexical_scores(
            query,
            artifacts,
            k1=k1,
            b=b,
        )
        lexical_rows.append(lexical_scores)
        coverage_rows.append(coverage_scores)

    lexical_scores_matrix = np.vstack(lexical_rows).astype(np.float32, copy=False)
    coverage_scores_matrix = np.vstack(coverage_rows).astype(np.float32, copy=False)
    return LexicalScores(
        lexical_norm=_normalize_rows(lexical_scores_matrix),
        coverage_scores=coverage_scores_matrix,
    )


def _title_score_matrix(queries: List[str], artifacts) -> np.ndarray:
    title_scores = np.zeros((len(queries), len(artifacts.page_ids)), dtype=np.float32)
    for row_idx, query in enumerate(queries):
        query_terms = set(tokenize_text(query, expand=True))
        if not query_terms:
            continue
        query_idfs = {
            token: artifacts.lexicon[token][2]
            for token in query_terms
            if token in artifacts.lexicon
        }
        total_idf = float(sum(query_idfs.values()))
        if total_idf <= 0.0:
            continue
        for doc_idx, title_terms in enumerate(artifacts.title_token_sets):
            if title_terms:
                overlap_idf = sum(
                    idf for token, idf in query_idfs.items() if token in title_terms
                )
                title_scores[row_idx, doc_idx] = min(1.0, overlap_idf / total_idf)
    return title_scores


def run_grid_search(
    core_scores: CoreScores,
    lexical_scores: LexicalScores,
    ground_truth: Sequence[set[int]],
    page_ids: np.ndarray,
    *,
    k1: float,
    b: float,
    current_best: float,
) -> TuneResult:
    local_best = TuneResult(
        score=-1.0,
        weights=Weights(semantic=0.0, lexical=0.0, coverage=0.0, title=0.0),
        k1=k1,
        b=b,
    )
    total = 0

    num_weights = 5 if USE_POPULARITY_IN_TUNING else 4
    for units in iter_weight_units(GRID_UNITS, num_weights):
        if USE_POPULARITY_IN_TUNING:
            sem_units, lex_units, cov_units, title_units, pop_units = units
        else:
            sem_units, lex_units, cov_units, title_units = units
            pop_units = 0
        weights = Weights(
            semantic=sem_units / GRID_UNITS,
            lexical=lex_units / GRID_UNITS,
            coverage=cov_units / GRID_UNITS,
            title=title_units / GRID_UNITS,
            popularity=pop_units / GRID_UNITS,
        )
        combined = (
            weights.lexical * lexical_scores.lexical_norm
            + weights.semantic * core_scores.semantic_norm
            + weights.coverage * lexical_scores.coverage_scores
            + weights.title * core_scores.title_scores
        )
        if USE_POPULARITY_IN_TUNING:
            if core_scores.popularity_scores is None:
                raise ValueError("Popularity scores were not precomputed")
            combined = combined + weights.popularity * core_scores.popularity_scores
        ranked = top_k_page_ids(combined, page_ids, K_EVAL)
        score = mean_ndcg_at_k(ranked, ground_truth, k=K_EVAL)
        total += 1
        if score > local_best.score:
            local_best = TuneResult(score=score, weights=weights, k1=k1, b=b)
        if score > current_best:
            current_best = score
            best_text = (
                f"new_global_best: mean_ndcg@10={score:.6f} "
                f"semantic={weights.semantic:.2f} "
                f"lexical={weights.lexical:.2f} "
                f"coverage={weights.coverage:.2f} "
                f"title={weights.title:.2f} "
            )
            if USE_POPULARITY_IN_TUNING:
                best_text += f"popularity={weights.popularity:.2f} "
            best_text += f"k1={k1:.2f} b={b:.2f}"
            print(best_text)

    print(
        f"Finished k1={k1:.2f} b={b:.2f}: "
        f"local_best={local_best.score:.6f}, searched={total}"
    )
    return local_best


def iter_weight_units(total_units: int, num_weights: int) -> Iterable[Tuple[int, ...]]:
    if num_weights <= 1:
        yield (total_units,)
        return

    for value in range(total_units + 1):
        for rest in iter_weight_units(total_units - value, num_weights - 1):
            yield (value, *rest)


def top_k_page_ids(
    combined_scores: np.ndarray,
    page_ids: np.ndarray,
    k: int,
) -> List[List[int]]:
    k = min(k, combined_scores.shape[1])
    candidate_idx = np.argpartition(-combined_scores, k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(combined_scores, candidate_idx, axis=1)
    order = np.argsort(-candidate_scores, axis=1, kind="mergesort")
    top_idx = np.take_along_axis(candidate_idx, order, axis=1)
    top_page_ids = page_ids[top_idx]
    return [[int(pid) for pid in row] for row in top_page_ids]


if __name__ == "__main__":
    main()
