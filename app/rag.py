"""
app/rag.py
Lightweight BM25-based retrieval for long RHP sections.
No vector DB or model downloads needed — fast and effective for keyword-rich financial text.
"""

import re
from typing import List

from rank_bm25 import BM25Okapi


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    """
    Split text into overlapping word-level chunks.
    chunk_size / overlap are in *words*, not characters.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple alpha-numeric tokenizer, lowercased, ignores 1-char tokens."""
    return [t for t in re.findall(r"\b[a-z][a-z0-9]{1,}\b", text.lower())]


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(
    text: str,
    query: str,
    max_chars: int = 14_000,
    top_k: int = 10,
    chunk_words: int = 400,
) -> str:
    """
    If the section text is already within max_chars, return it directly.
    Otherwise use BM25 to rank chunks and return the top-k most relevant ones
    (in document order, separated by a divider for the LLM's context).

    Args:
        text:       Full section text.
        query:      Natural language query used for BM25 ranking.
        max_chars:  Return full text unchanged below this threshold.
        top_k:      Number of top chunks to return for long sections.
        chunk_words: Word count per chunk.

    Returns:
        A string containing the most relevant content from the section.
    """
    if len(text) <= max_chars:
        return text

    chunks = chunk_text(text, chunk_size=chunk_words)
    if len(chunks) <= top_k:
        # Not many chunks anyway — return full text trimmed to max_chars
        return text[:max_chars]

    tokenized = [_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))

    # Pick top-k indices, then sort them to preserve reading order
    top_indices = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    )

    retrieved = "\n\n---\n\n".join(chunks[i] for i in top_indices)
    return retrieved
