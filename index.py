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


@dataclass(frozen=True)
class LexicalArtifacts:
    lexicon: Dict[str, Tuple[int, int, float]]
    post_doc_ids: np.ndarray
    post_tfs: np.ndarray
    doc_lengths: np.ndarray
    avg_doc_length: float


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
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
        records
    )

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    np.save(out_dir / POST_DOC_IDS_NAME, post_doc_ids)
    np.save(out_dir / POST_TFS_NAME, post_tfs)
    np.save(out_dir / DOC_LENGTHS_NAME, doc_lengths)

    meta = {
        "artifact_version": 4,
        "page_ids": page_ids,
        "titles": titles,
        "model": EMBEDDING_MODEL_NAME,
        "num_vectors": len(page_ids),
        "lexical": {
            "algorithm": "bm25",
            "lexicon": LEXICON_NAME,
            "post_doc_ids": POST_DOC_IDS_NAME,
            "post_tfs": POST_TFS_NAME,
            "doc_lengths": DOC_LENGTHS_NAME,
            "avg_doc_length": avg_doc_length,
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
) -> Tuple[Dict[str, Tuple[int, int, float]], np.ndarray, np.ndarray, np.ndarray, float]:
    """Build compact BM25 posting-list artifacts."""
    num_docs = len(records)
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_lengths = np.zeros(num_docs, dtype=np.float32)

    for doc_idx, record in enumerate(records):
        tokens = tokenize_text(entry_text(record), expand=True)
        counts = Counter(tokens)
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


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed vectors and page_id map from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
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
    return LexicalArtifacts(
        lexicon=lexicon,
        post_doc_ids=np.load(root / lexical_meta.get("post_doc_ids", POST_DOC_IDS_NAME)),
        post_tfs=np.load(root / lexical_meta.get("post_tfs", POST_TFS_NAME)),
        doc_lengths=np.load(root / lexical_meta.get("doc_lengths", DOC_LENGTHS_NAME)),
        avg_doc_length=float(lexical_meta.get("avg_doc_length", 0.0)),
    )
