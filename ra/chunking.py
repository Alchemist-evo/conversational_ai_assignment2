"""Chunking strategies (Task 3, Requirement 3) — implemented from scratch.

* fixed     : sliding window over GPT-4o tokens, size S, overlap = ratio * S.
* recursive : split on the coarsest separator ("\n\n" -> "\n" -> ". " -> " ") that keeps
              pieces under S tokens, then greedily merge pieces, carrying an overlap tail.
* semantic  : split into sentences, embed them, and start a new chunk where the cosine
              distance between consecutive sentences exceeds the p-th percentile
              (topic shift), subject to a hard max size; overlap carries trailing sentences.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .tokens import count_tokens, decode, encode

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return count_tokens(self.text)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]


# --------------------------------------------------------------------- fixed
def fixed_chunks(text: str, size: int = 256, overlap: float = 0.2) -> list[str]:
    ids = encode(text)
    if len(ids) <= size:
        return [text]
    step = max(1, int(size * (1 - overlap)))
    out = []
    for start in range(0, len(ids), step):
        piece = ids[start : start + size]
        out.append(decode(piece))
        if start + size >= len(ids):
            break
    return out


# ----------------------------------------------------------------- recursive
SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    if count_tokens(text) <= size:
        return [text]
    if not seps:
        return fixed_chunks(text, size, 0.0)
    sep, rest = seps[0], seps[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return _split_recursive(text, size, rest)
    out = []
    for i, p in enumerate(parts):
        p = p + (sep if sep.strip() and i < len(parts) - 1 else "")
        out.extend(_split_recursive(p, size, rest) if count_tokens(p) > size else [p])
    return out


def recursive_chunks(text: str, size: int = 256, overlap: float = 0.2) -> list[str]:
    pieces = _split_recursive(text, size, SEPARATORS)
    overlap_tokens = int(size * overlap)
    chunks, cur, cur_tok = [], [], 0
    for p in pieces:
        t = count_tokens(p)
        if cur and cur_tok + t > size:
            chunks.append(" ".join(cur).strip())
            # carry an overlap tail of whole pieces (up to overlap_tokens)
            tail, tail_tok = [], 0
            for q in reversed(cur):
                qt = count_tokens(q)
                if tail_tok + qt > overlap_tokens:
                    break
                tail.insert(0, q)
                tail_tok += qt
            cur, cur_tok = tail, tail_tok
        cur.append(p)
        cur_tok += t
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]


# ------------------------------------------------------------------ semantic
def semantic_chunks(text: str, embedder, max_size: int = 512, percentile: float = 80.0, overlap: float = 0.0,
                    min_size: int = 40) -> list[str]:
    sents = split_sentences(text)
    if len(sents) <= 1 or count_tokens(text) <= min_size:
        return [text]
    emb = embedder.encode(sents)
    dist = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)  # cosine distance (embeddings are L2-normalised)
    thresh = np.percentile(dist, percentile)
    overlap_tokens = int(max_size * overlap)
    chunks, cur, cur_tok = [], [sents[0]], count_tokens(sents[0])
    for i in range(1, len(sents)):
        s, t = sents[i], count_tokens(sents[i])
        boundary = dist[i - 1] > thresh and cur_tok >= min_size
        if boundary or cur_tok + t > max_size:
            chunks.append(" ".join(cur))
            tail, tail_tok = [], 0
            for q in reversed(cur):
                qt = count_tokens(q)
                if tail_tok + qt > overlap_tokens:
                    break
                tail.insert(0, q)
                tail_tok += qt
            cur, cur_tok = tail, tail_tok
        cur.append(s)
        cur_tok += t
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ------------------------------------------------------------------ dispatch
def chunk_document(doc: dict, strategy: str = "recursive", size: int = 256, overlap: float = 0.2,
                   embedder=None) -> list[Chunk]:
    text = doc["text"]
    if strategy == "fixed":
        pieces = fixed_chunks(text, size, overlap)
    elif strategy == "recursive":
        pieces = recursive_chunks(text, size, overlap)
    elif strategy == "semantic":
        pieces = semantic_chunks(text, embedder, max_size=size, overlap=overlap)
    elif strategy == "none":
        pieces = [text]
    else:
        raise ValueError(strategy)
    meta = {k: v for k, v in doc.items() if k != "text"}
    return [Chunk(f"{doc['doc_id']}#{i}", doc["doc_id"], doc["source"], p, meta) for i, p in enumerate(pieces)]
