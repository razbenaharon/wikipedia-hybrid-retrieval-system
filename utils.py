"""Shared paths and helpers for Section B."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List

STUDENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDENT_ROOT / "data"
ENTRIES_DIR = DATA_DIR / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR = STUDENT_ROOT / "artifacts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10

TOKEN_RE = re.compile(r"[a-z0-9]+")
CHAR_NGRAM_PREFIXES = ("ng3:", "ng4:")
INCLUDE_CHAR_NGRAMS = False
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "me",
    "more",
    "most",
    "my",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
}

IMPORTANT_BIGRAM_WORDS = {
    "acoustic",
    "airport",
    "alloy",
    "arena",
    "assembly",
    "automated",
    "banner",
    "basketball",
    "bench",
    "bridge",
    "captain",
    "championship",
    "city",
    "club",
    "commission",
    "commemorative",
    "commuter",
    "company",
    "contracts",
    "cooperative",
    "corridors",
    "crane",
    "deals",
    "delta",
    "diplomatic",
    "distribution",
    "executive",
    "exports",
    "field",
    "final",
    "finals",
    "firm",
    "fjord",
    "foundation",
    "franchise",
    "game",
    "graduate",
    "guard",
    "harbor",
    "home",
    "hometown",
    "humidity",
    "imaging",
    "institute",
    "international",
    "joint",
    "laboratory",
    "labor",
    "league",
    "light",
    "lines",
    "maritime",
    "memorial",
    "monitoring",
    "neutral",
    "observers",
    "overland",
    "overseas",
    "patent",
    "peace",
    "performance",
    "physicist",
    "player",
    "point",
    "policy",
    "pool",
    "profit",
    "rail",
    "regional",
    "research",
    "researcher",
    "results",
    "revenue",
    "river",
    "riverfront",
    "routes",
    "series",
    "service",
    "shipbuilding",
    "sister",
    "software",
    "spin",
    "stability",
    "team",
    "teaching",
    "thermal",
    "title",
    "talks",
    "tomography",
    "trade",
    "training",
    "transport",
    "treaty",
    "trials",
    "urban",
    "vibration",
}


def normalize_page_id(value: Any) -> int:
    """Coerce page_id from JSON (int or numeric string) to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid page_id: {value!r}")


def load_public_queries(path: Path | None = None) -> List[Dict[str, Any]]:
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]
    return rows


def iter_entries(entries_dir: Path | None = None) -> Iterator[Dict[str, Any]]:
    """Yield one record per JSON file in the corpus directory."""
    root = entries_dir or ENTRIES_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {root}. "
            "Expected student/data/Wikipedia Entries/ with one JSON file per page."
        )
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["page_id"] = normalize_page_id(data.get("page_id", path.stem))
        yield data


def entry_text(record: Dict[str, Any]) -> str:
    title = record.get("title", "")
    content = record.get("content", "")
    if title:
        return f"{title}\n\n{content}".strip()
    return str(content).strip()


def token_variants(token: str) -> List[str]:
    """Return lightweight suffix variants used by lexical retrieval."""
    variants = [token]
    if token.isdigit() and len(token) == 4:
        variants.append(f"{token[:3]}0s")
    if len(token) == 5 and token[:4].isdigit() and token.endswith("s"):
        decade_start = int(token[:4])
        if decade_start % 10 == 0:
            variants.extend(str(year) for year in range(decade_start, decade_start + 10))
    if len(token) > 5 and token.endswith("ies"):
        variants.append(token[:-3] + "y")
    if len(token) > 5 and token.endswith("ing"):
        variants.append(token[:-3])
    if len(token) > 4 and token.endswith("ed"):
        variants.append(token[:-2])
    if len(token) > 4 and token.endswith("es"):
        variants.append(token[:-2])
    if len(token) > 3 and token.endswith("s"):
        variants.append(token[:-1])

    deduped: List[str] = []
    seen: set[str] = set()
    for value in variants:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def char_ngrams(token: str) -> List[str]:
    """Return prefixed character 3-grams and 4-grams for fuzzy matching."""
    if not (len(token) > 5 or any(ch.isdigit() for ch in token)):
        return []

    grams: List[str] = []
    for n in (3, 4):
        if len(token) < n:
            continue
        prefix = f"ng{n}:"
        grams.extend(
            f"{prefix}{token[i : i + n]}"
            for i in range(len(token) - n + 1)
        )
    return grams


def is_char_ngram_token(token: str) -> bool:
    return token.startswith(CHAR_NGRAM_PREFIXES)


def tokenize_text(
    text: str,
    *,
    expand: bool = True,
    include_bigrams: bool = True,
    include_char_ngrams: bool = True,
) -> List[str]:
    """
    Tokenize text for BM25/title matching.

    Keeps alphanumeric tokens and numbers, drops common stopwords, and can add
    simple suffix variants so queries like "captained" can match "captain".
    Optional prefixed character n-grams improve matching for synthetic names,
    numeric facts, and small typo variations.
    """
    base_tokens = [
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]

    tokens: List[str] = []
    for token in base_tokens:
        if expand:
            tokens.extend(token_variants(token))
        else:
            tokens.append(token)
        if include_char_ngrams and INCLUDE_CHAR_NGRAMS:
            tokens.extend(char_ngrams(token))
    if include_bigrams:
        tokens.extend(
            f"{left}_{right}"
            for left, right in zip(base_tokens, base_tokens[1:])
            if left in IMPORTANT_BIGRAM_WORDS and right in IMPORTANT_BIGRAM_WORDS
        )
    return tokens


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
