"""Task 3 — agentic memory implemented from scratch.

* WorkingMemory  : bounded deque of the last N=10 research steps. Steps that fall off the left end are not
                   lost: they are folded into an LLM-written *rolling summary* (batched every `summarize_every`
                   evictions to save calls).
* EpisodicMemory : past research sessions embedded with all-MiniLM-L6-v2 and stored in a FAISS inner-product
                   index over L2-normalised vectors (= cosine similarity); top-k retrieval at query time; new
                   sessions are appended after each research query ("Memory Update").
* Semantic memory: the chunked knowledge collections in retrieval.MultiSourceRetriever.
"""
from __future__ import annotations

import json
from collections import deque

import faiss
import numpy as np

from . import config
from .llm import LLM
from .retrieval import Embedder
from .tokens import count_tokens


class WorkingMemory:
    def __init__(self, n: int = config.WORKING_MEMORY_N, llm: LLM | None = None, summarize_every: int = 3):
        self.buffer: deque = deque(maxlen=n)
        self.n = n
        self.llm = llm
        self.summary = ""
        self.pending: list[dict] = []
        self.summarize_every = summarize_every
        self.evicted_total = 0
        self.summary_calls = 0

    def add(self, step: dict) -> dict | None:
        evicted = self.buffer[0] if len(self.buffer) == self.n else None
        self.buffer.append(step)  # deque(maxlen) drops the oldest automatically
        if evicted is not None:
            self.evicted_total += 1
            self.pending.append(evicted)
            if len(self.pending) >= self.summarize_every:
                self.flush()
        return evicted

    def flush(self) -> None:
        if not self.pending:
            return
        text = "\n".join(f"- {self._fmt(s)}" for s in self.pending)
        if self.llm is None:
            self.summary = (self.summary + " " + text)[-1500:]
        else:
            r = self.llm.chat(
                f"Current rolling summary of earlier research steps:\n{self.summary or '(empty)'}\n\nOlder steps now "
                f"leaving working memory:\n{text}\n\nRewrite the rolling summary (max 120 words) so it keeps every "
                "finding, open question and source id that may matter later. Output only the summary.",
                purpose="t3_working_memory_summary", max_tokens=220)
            self.summary = r.text.strip()
            self.summary_calls += 1
        self.pending = []

    @staticmethod
    def _fmt(s: dict) -> str:
        return f"[{s.get('type', 'step')}] {s.get('content', '')}"[:400]

    def render(self) -> str:
        parts = []
        if self.summary or self.pending:
            pend = " ".join(self._fmt(s) for s in self.pending)
            parts.append("Summary of earlier steps: " + (self.summary + (" " + pend if pend else "")).strip())
        parts += [f"{i+1}. {self._fmt(s)}" for i, s in enumerate(self.buffer)]
        return "\n".join(parts)

    def tokens(self) -> int:
        return count_tokens(self.render())


class EpisodicMemory:
    def __init__(self, sessions: list[dict]):
        self.embedder = Embedder()
        self.sessions: list[dict] = []
        self.index = faiss.IndexFlatIP(384)
        self.add_many(sessions)

    @staticmethod
    def session_text(s: dict) -> str:
        return f"{s['user_query']} | findings: {' '.join(s.get('key_findings', []))}"

    def add_many(self, sessions: list[dict]) -> None:
        if not sessions:
            return
        vecs = self.embedder.encode([self.session_text(s) for s in sessions])
        self.index.add(vecs)
        self.sessions.extend(sessions)

    def add(self, session: dict) -> None:
        self.add_many([session])

    def retrieve(self, query: str, k: int = config.EPISODIC_TOP_K) -> list[tuple[dict, float]]:
        if not self.sessions:
            return []
        q = self.embedder.encode(query).reshape(1, -1)
        D, I = self.index.search(q, min(k, len(self.sessions)))
        return [(self.sessions[i], float(d)) for i, d in zip(I[0], D[0]) if i >= 0]

    @staticmethod
    def render(items: list[tuple[dict, float]]) -> str:
        out = []
        for s, sim in items:
            tag = "SYNTHETIC past session" if s.get("synthetic") else "past session"
            out.append(f"[{tag} {s['session_id']} | {s['date']} | cosine={sim:.2f}] Q: {s['user_query']} | "
                       f"Findings: {'; '.join(s.get('key_findings', []))} | User preference: {s.get('user_preference', '-')}")
        return "\n".join(out)
