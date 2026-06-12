"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Return one retrieval unit per page.

    The page-level architecture embeds the combined title/content text as a
    single vector so long pages are represented consistently with BM25 pages.
    """
    page_id = int(record["page_id"])
    return [Chunk(page_id=page_id, chunk_id=0, text=entry_text(record))]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
