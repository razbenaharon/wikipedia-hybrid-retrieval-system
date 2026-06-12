"""Embedding utilities (sentence-transformers/all-MiniLM-L6-v2 only)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from utils import EMBEDDING_MODEL_NAME

_model: SentenceTransformer | None = None
_device: str | None = None


def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        if _device == "cpu":
            print("WARNING: CUDA is not available to PyTorch; using CPU embeddings.")
    return _device


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=get_device())
    return _model


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    """Return L2-normalized embeddings, shape (n, dim)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=get_device(),
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(queries: List[str], *, batch_size: int = 64) -> np.ndarray:
    return embed_texts(queries, batch_size=batch_size)
