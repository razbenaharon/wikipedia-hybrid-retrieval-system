"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from embed import embed_texts
from utils import (
    ARTIFACTS_DIR,
    EMBEDDING_MODEL_NAME,
    USE_STEMMING,
    ensure_artifacts_dir,
    entry_text,
    iter_entries,
    tokenize_text,
)

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
LEXICON_NAME = "lexicon.json"
POST_DOC_IDS_NAME = "post_doc_ids.npy"
POST_TFS_NAME = "post_tfs.npy"
DOC_LENGTHS_NAME = "doc_lengths.npy"
POPULARITY_SCORES_NAME = "popularity_scores.npy"
BM25F_LEAD_LEXICON_NAME = "bm25f_lead_lexicon.json"
BM25F_LEAD_POST_DOC_IDS_NAME = "bm25f_lead_post_doc_ids.npy"
BM25F_LEAD_POST_TFS_NAME = "bm25f_lead_post_tfs.npy"
BM25F_LEAD_DOC_LENGTHS_NAME = "bm25f_lead_doc_lengths.npy"
BM25F_BODY_LEXICON_NAME = "bm25f_body_lexicon.json"
BM25F_BODY_POST_DOC_IDS_NAME = "bm25f_body_post_doc_ids.npy"
BM25F_BODY_POST_TFS_NAME = "bm25f_body_post_tfs.npy"
BM25F_BODY_DOC_LENGTHS_NAME = "bm25f_body_doc_lengths.npy"
LEAD_WORDS = 200
LEAD_TF_BOOST = 3
USE_BM25F = True
WIKIPEDIA_META_STOPWORDS = {
    "also",
    "archive",
    "archived",
    "article",
    "articles",
    "categories",
    "category",
    "doi",
    "external",
    "isbn",
    "link",
    "links",
    "main",
    "reference",
    "references",
    "retrieved",
    "see",
}


@dataclass(frozen=True)
class LexicalArtifacts:
    lexicon: Dict[str, Tuple[int, int, float]]
    post_doc_ids: np.ndarray
    post_tfs: np.ndarray
    doc_lengths: np.ndarray
    avg_doc_length: float
    bm25f_lead_lexicon: Optional[Dict[str, Tuple[int, int, float]]] = None
    bm25f_lead_post_doc_ids: Optional[np.ndarray] = None
    bm25f_lead_post_tfs: Optional[np.ndarray] = None
    bm25f_lead_doc_lengths: Optional[np.ndarray] = None
    bm25f_lead_avg_doc_length: float = 0.0
    bm25f_body_lexicon: Optional[Dict[str, Tuple[int, int, float]]] = None
    bm25f_body_post_doc_ids: Optional[np.ndarray] = None
    bm25f_body_post_tfs: Optional[np.ndarray] = None
    bm25f_body_doc_lengths: Optional[np.ndarray] = None
    bm25f_body_avg_doc_length: float = 0.0


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    lead_words: int = LEAD_WORDS,
    lead_tf_boost: int = LEAD_TF_BOOST,
) -> Tuple[np.ndarray, List[int]]:
    """
    Embed full pages and persist page-level lexical artifacts.

    Returns (vectors, page_ids) where row i corresponds to page_ids[i].
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_entries(entries_dir))
    texts = [entry_text(record) for record in records]
    vectors = embed_texts(texts)
    page_ids = [int(record["page_id"]) for record in records]
    titles = [str(record.get("title", "")) for record in records]

    lexicon, post_doc_ids, post_tfs, doc_lengths, avg_doc_length = _build_lexical_index(
        records,
        lead_words=lead_words,
        lead_tf_boost=lead_tf_boost,
    )
    bm25f_artifacts = None
    if USE_BM25F:
        bm25f_artifacts = _build_bm25f_indexes(records, lead_words=lead_words)

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    np.save(out_dir / POST_DOC_IDS_NAME, post_doc_ids)
    np.save(out_dir / POST_TFS_NAME, post_tfs)
    np.save(out_dir / DOC_LENGTHS_NAME, doc_lengths)
    if bm25f_artifacts is not None:
        lead, body = bm25f_artifacts
        _save_lexical_artifact_bundle(
            out_dir,
            lead,
            lexicon_name=BM25F_LEAD_LEXICON_NAME,
            post_doc_ids_name=BM25F_LEAD_POST_DOC_IDS_NAME,
            post_tfs_name=BM25F_LEAD_POST_TFS_NAME,
            doc_lengths_name=BM25F_LEAD_DOC_LENGTHS_NAME,
        )
        _save_lexical_artifact_bundle(
            out_dir,
            body,
            lexicon_name=BM25F_BODY_LEXICON_NAME,
            post_doc_ids_name=BM25F_BODY_POST_DOC_IDS_NAME,
            post_tfs_name=BM25F_BODY_POST_TFS_NAME,
            doc_lengths_name=BM25F_BODY_DOC_LENGTHS_NAME,
        )
    popularity_scores = _build_popularity_scores(records)
    np.save(out_dir / POPULARITY_SCORES_NAME, popularity_scores)

    meta = {
        "artifact_version": 6,
        "page_ids": page_ids,
        "titles": titles,
        "model": EMBEDDING_MODEL_NAME,
        "num_vectors": len(page_ids),
        "semantic": {
            "algorithm": "minilm_document_native_truncated",
            "vectors": INDEX_VECTORS_NAME,
        },
        "lexical": {
            "algorithm": "bm25_lead_boost",
            "use_stemming": USE_STEMMING,
            "use_bm25f": USE_BM25F,
            "lexicon": LEXICON_NAME,
            "post_doc_ids": POST_DOC_IDS_NAME,
            "post_tfs": POST_TFS_NAME,
            "doc_lengths": DOC_LENGTHS_NAME,
            "avg_doc_length": avg_doc_length,
            "lead_words": lead_words,
            "lead_tf_boost": lead_tf_boost,
        },
        "popularity": {
            "algorithm": "normalized_title_phrase_mentions",
            "scores": POPULARITY_SCORES_NAME,
        },
    }
    if bm25f_artifacts is not None:
        lead, body = bm25f_artifacts
        meta["bm25f"] = {
            "algorithm": "bm25f_lead_body",
            "lead_words": lead_words,
            "lead": {
                "lexicon": BM25F_LEAD_LEXICON_NAME,
                "post_doc_ids": BM25F_LEAD_POST_DOC_IDS_NAME,
                "post_tfs": BM25F_LEAD_POST_TFS_NAME,
                "doc_lengths": BM25F_LEAD_DOC_LENGTHS_NAME,
                "avg_doc_length": lead[4],
            },
            "body": {
                "lexicon": BM25F_BODY_LEXICON_NAME,
                "post_doc_ids": BM25F_BODY_POST_DOC_IDS_NAME,
                "post_tfs": BM25F_BODY_POST_TFS_NAME,
                "doc_lengths": BM25F_BODY_DOC_LENGTHS_NAME,
                "avg_doc_length": body[4],
            },
        }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (out_dir / LEXICON_NAME).write_text(
        json.dumps(lexicon, separators=(",", ":")), encoding="utf-8"
    )
    return vectors, page_ids


def _build_lexical_index(
    records: List[dict],
    *,
    lead_words: int = LEAD_WORDS,
    lead_tf_boost: int = LEAD_TF_BOOST,
) -> Tuple[Dict[str, Tuple[int, int, float]], np.ndarray, np.ndarray, np.ndarray, float]:
    """Build compact BM25 posting-list artifacts."""
    num_docs = len(records)
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_lengths = np.zeros(num_docs, dtype=np.float32)

    for doc_idx, record in enumerate(records):
        text = entry_text(record)
        tokens = _filter_wikipedia_meta_tokens(tokenize_text(text, expand=True))
        counts = Counter(tokens)
        lead_text = " ".join(text.split()[:lead_words])
        lead_tokens = _filter_wikipedia_meta_tokens(
            tokenize_text(lead_text, expand=True)
        )
        lead_counts = Counter(lead_tokens)
        for token, lead_tf in lead_counts.items():
            counts[token] += (lead_tf_boost - 1) * lead_tf

        doc_lengths[doc_idx] = float(sum(counts.values()))
        for token, tf in counts.items():
            postings.setdefault(token, []).append((doc_idx, min(tf, 65535)))

    avg_doc_length = float(doc_lengths.mean()) if num_docs else 0.0
    flat_doc_ids: list[int] = []
    flat_tfs: list[int] = []
    lexicon: Dict[str, Tuple[int, int, float]] = {}

    for token in sorted(postings):
        rows = postings[token]
        offset = len(flat_doc_ids)
        df = len(rows)
        idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        flat_doc_ids.extend(doc_idx for doc_idx, _ in rows)
        flat_tfs.extend(tf for _, tf in rows)
        lexicon[token] = (offset, df, round(idf, 6))

    return (
        lexicon,
        np.asarray(flat_doc_ids, dtype=np.int32),
        np.asarray(flat_tfs, dtype=np.uint16),
        doc_lengths,
        avg_doc_length,
    )


def _build_bm25f_indexes(
    records: List[dict],
    *,
    lead_words: int = LEAD_WORDS,
) -> Tuple[
    Tuple[Dict[str, Tuple[int, int, float]], np.ndarray, np.ndarray, np.ndarray, float],
    Tuple[Dict[str, Tuple[int, int, float]], np.ndarray, np.ndarray, np.ndarray, float],
]:
    lead_counts: List[Counter[str]] = []
    body_counts: List[Counter[str]] = []
    for record in records:
        text = entry_text(record)
        words = text.split()
        lead_text = " ".join(words[:lead_words])
        body_text = " ".join(words[lead_words:])
        lead_tokens = _filter_wikipedia_meta_tokens(
            tokenize_text(lead_text, expand=True)
        )
        body_tokens = _filter_wikipedia_meta_tokens(
            tokenize_text(body_text, expand=True)
        )
        lead_counts.append(Counter(lead_tokens))
        body_counts.append(Counter(body_tokens))
    return (
        _build_lexical_artifacts_from_counts(lead_counts),
        _build_lexical_artifacts_from_counts(body_counts),
    )


def _build_lexical_artifacts_from_counts(
    counts_by_doc: List[Counter[str]],
) -> Tuple[Dict[str, Tuple[int, int, float]], np.ndarray, np.ndarray, np.ndarray, float]:
    num_docs = len(counts_by_doc)
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_lengths = np.zeros(num_docs, dtype=np.float32)

    for doc_idx, counts in enumerate(counts_by_doc):
        doc_lengths[doc_idx] = float(sum(counts.values()))
        for token, tf in counts.items():
            postings.setdefault(token, []).append((doc_idx, min(tf, 65535)))

    avg_doc_length = float(doc_lengths.mean()) if num_docs else 0.0
    flat_doc_ids: list[int] = []
    flat_tfs: list[int] = []
    lexicon: Dict[str, Tuple[int, int, float]] = {}

    for token in sorted(postings):
        rows = postings[token]
        offset = len(flat_doc_ids)
        df = len(rows)
        idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        flat_doc_ids.extend(doc_idx for doc_idx, _ in rows)
        flat_tfs.extend(tf for _, tf in rows)
        lexicon[token] = (offset, df, round(idf, 6))

    return (
        lexicon,
        np.asarray(flat_doc_ids, dtype=np.int32),
        np.asarray(flat_tfs, dtype=np.uint16),
        doc_lengths,
        avg_doc_length,
    )


def _save_lexical_artifact_bundle(
    out_dir: Path,
    bundle: Tuple[Dict[str, Tuple[int, int, float]], np.ndarray, np.ndarray, np.ndarray, float],
    *,
    lexicon_name: str,
    post_doc_ids_name: str,
    post_tfs_name: str,
    doc_lengths_name: str,
) -> None:
    lexicon, post_doc_ids, post_tfs, doc_lengths, _ = bundle
    (out_dir / lexicon_name).write_text(
        json.dumps(lexicon, separators=(",", ":")),
        encoding="utf-8",
    )
    np.save(out_dir / post_doc_ids_name, post_doc_ids)
    np.save(out_dir / post_tfs_name, post_tfs)
    np.save(out_dir / doc_lengths_name, doc_lengths)


def _filter_wikipedia_meta_tokens(tokens: List[str]) -> List[str]:
    """Drop common Wikipedia structure terms from BM25 artifacts."""
    return [token for token in tokens if token not in WIKIPEDIA_META_STOPWORDS]


def _build_popularity_scores(records: List[dict]) -> np.ndarray:
    """Compute row-aligned title mention popularity scores from page content."""
    title_candidates: dict[str, list[tuple[tuple[str, ...], tuple[int, ...]]]] = {}
    title_to_doc_idxs: dict[tuple[str, ...], list[int]] = {}
    for doc_idx, record in enumerate(records):
        title_tokens = tuple(
            tokenize_text(
                str(record.get("title", "")),
                expand=False,
                include_bigrams=False,
                include_char_ngrams=False,
            )
        )
        if not title_tokens:
            continue
        title_to_doc_idxs.setdefault(title_tokens, []).append(doc_idx)

    for title_tokens, doc_idxs in title_to_doc_idxs.items():
        title_candidates.setdefault(title_tokens[0], []).append(
            (title_tokens, tuple(doc_idxs))
        )

    counts = np.zeros(len(records), dtype=np.float32)
    for source_doc_idx, record in enumerate(records):
        content_tokens = tokenize_text(
            str(record.get("content", "")),
            expand=False,
            include_bigrams=False,
            include_char_ngrams=False,
        )
        for pos, token in enumerate(content_tokens):
            for title_tokens, target_doc_idxs in title_candidates.get(token, ()):
                end = pos + len(title_tokens)
                if end > len(content_tokens):
                    continue
                if tuple(content_tokens[pos:end]) != title_tokens:
                    continue
                for target_doc_idx in target_doc_idxs:
                    if target_doc_idx != source_doc_idx:
                        counts[target_doc_idx] += 1.0

    if counts.max(initial=0.0) <= 0.0:
        return counts
    log_counts = np.log1p(counts)
    return (log_counts / log_counts.max()).astype(np.float32, copy=False)


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed document vectors and page_id map from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    semantic_meta = meta.get("semantic", {})
    vectors = np.load(root / semantic_meta.get("vectors", INDEX_VECTORS_NAME))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids


def load_meta(artifacts_dir: Optional[Path] = None) -> dict:
    root = artifacts_dir or ARTIFACTS_DIR
    return json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))


def load_lexical_artifacts(artifacts_dir: Optional[Path] = None) -> LexicalArtifacts:
    """Load BM25 posting-list artifacts."""
    root = artifacts_dir or ARTIFACTS_DIR
    meta = load_meta(root)
    lexical_meta = meta.get("lexical", {})
    raw_lexicon = json.loads(
        (root / lexical_meta.get("lexicon", LEXICON_NAME)).read_text(encoding="utf-8")
    )
    lexicon = {
        token: (int(values[0]), int(values[1]), float(values[2]))
        for token, values in raw_lexicon.items()
    }
    kwargs = {}
    bm25f_meta = meta.get("bm25f")
    if bm25f_meta:
        lead_meta = bm25f_meta.get("lead", {})
        body_meta = bm25f_meta.get("body", {})
        lead_lexicon = _load_lexicon(root / lead_meta.get("lexicon", BM25F_LEAD_LEXICON_NAME))
        body_lexicon = _load_lexicon(root / body_meta.get("lexicon", BM25F_BODY_LEXICON_NAME))
        kwargs.update(
            {
                "bm25f_lead_lexicon": lead_lexicon,
                "bm25f_lead_post_doc_ids": np.load(
                    root / lead_meta.get("post_doc_ids", BM25F_LEAD_POST_DOC_IDS_NAME)
                ),
                "bm25f_lead_post_tfs": np.load(
                    root / lead_meta.get("post_tfs", BM25F_LEAD_POST_TFS_NAME)
                ),
                "bm25f_lead_doc_lengths": np.load(
                    root / lead_meta.get("doc_lengths", BM25F_LEAD_DOC_LENGTHS_NAME)
                ),
                "bm25f_lead_avg_doc_length": float(
                    lead_meta.get("avg_doc_length", 0.0)
                ),
                "bm25f_body_lexicon": body_lexicon,
                "bm25f_body_post_doc_ids": np.load(
                    root / body_meta.get("post_doc_ids", BM25F_BODY_POST_DOC_IDS_NAME)
                ),
                "bm25f_body_post_tfs": np.load(
                    root / body_meta.get("post_tfs", BM25F_BODY_POST_TFS_NAME)
                ),
                "bm25f_body_doc_lengths": np.load(
                    root / body_meta.get("doc_lengths", BM25F_BODY_DOC_LENGTHS_NAME)
                ),
                "bm25f_body_avg_doc_length": float(
                    body_meta.get("avg_doc_length", 0.0)
                ),
            }
        )

    return LexicalArtifacts(
        lexicon=lexicon,
        post_doc_ids=np.load(root / lexical_meta.get("post_doc_ids", POST_DOC_IDS_NAME)),
        post_tfs=np.load(root / lexical_meta.get("post_tfs", POST_TFS_NAME)),
        doc_lengths=np.load(root / lexical_meta.get("doc_lengths", DOC_LENGTHS_NAME)),
        avg_doc_length=float(lexical_meta.get("avg_doc_length", 0.0)),
        **kwargs,
    )


def _load_lexicon(path: Path) -> Dict[str, Tuple[int, int, float]]:
    raw_lexicon = json.loads(path.read_text(encoding="utf-8"))
    return {
        token: (int(values[0]), int(values[1]), float(values[2]))
        for token, values in raw_lexicon.items()
    }
