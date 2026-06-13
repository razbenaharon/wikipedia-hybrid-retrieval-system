"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 50


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one article into overlapping semantic retrieval passages.

    Each passage includes the title as context, but windows are formed from the
    article content so long pages can match queries about buried details.
    """
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    words = str(record.get("content", "")).split()

    if not words:
        return [Chunk(page_id=page_id, chunk_id=0, text=title)]

    chunks: List[Chunk] = []
    for chunk_id, passage_words in enumerate(_sliding_windows(words)):
        passage = " ".join(passage_words)
        text = f"{title}\n\n{passage}".strip() if title else passage
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=text))
    return chunks


def _sliding_windows(words: List[str]) -> List[List[str]]:
    """Return overlapping word windows for passage embedding."""
    step = max(1, CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS)
    windows: List[List[str]] = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + CHUNK_SIZE_WORDS])
        if start + CHUNK_SIZE_WORDS >= len(words):
            break
    return windows


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
