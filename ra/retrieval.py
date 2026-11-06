"""Hybrid retrieval: BM25 (rank-bm25) + dense (all-MiniLM-L6-v2 + FAISS) fused with RRF,
optional cross-encoder re-ranking, and a multi-source wrapper with access control.

Black-box components allowed by the assignment: BM25, FAISS, RRF formula, cross-encoder.
Everything around them (tokenisation, candidate depth, fusion weights, per-source
routing, access control, caching) is ours and documented here.

How the fusion works
--------------------
BM25 scores lexical overlap (term frequency saturated by k1, normalised by document length
via b); it excels at rare keywords, names and numbers. Dense retrieval embeds query and
chunk into the same 384-d space and ranks by cosine similarity; it captures paraphrases.
Their scores live on incomparable scales, so Reciprocal Rank Fusion ignores scores and
combines *ranks*:   RRF(d) = sum_r  w_r / (k + rank_r(d)),  k = 60.
A document ranked well by either retriever rises; one ranked well by both rises most.
"""
from __future__ import annotations

import hashlib
import pickle
import re
import threading
import time
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .chunking import Chunk

_STOP = set(
    "a an the of and or to in on for with by is are was were be been being as at from that this these those it its "
    "what which who whom how why when where do does did can could should would will may might than then there their "
    "they them he she his her you your we our i me my not no yes about into over under between".split()
)
_TOK_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def bm25_tokenize(text: str) -> list[str]:
    return [t for t in _TOK_RE.findall(text.lower()) if t not in _STOP]


# ----------------------------------------------------------------- embedder
class Embedder:
    """Singleton wrapper around SentenceTransformer with a persistent embedding cache."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *a, **k):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(config.EMBED_MODEL, device=config.torch_device())
        self.path = config.CACHE_DIR / "emb_cache.pkl"
        try:
            self.cache = pickle.load(open(self.path, "rb"))
        except Exception:
            self.cache = {}
        self._dirty = 0

    @staticmethod
    def _h(t: str) -> str:
        return hashlib.md5(t.encode()).hexdigest()

    def encode(self, texts: list[str] | str) -> np.ndarray:
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        keys = [self._h(t) for t in texts]
        # The tools run in parallel threads (parallel function calling). SentenceTransformer/tokenizer calls and the
        # shared cache are not thread-safe (a race produced "IndexError: tuple index out of range"), so encoding and
        # cache access are serialised. Query encoding takes milliseconds, so the parallel speed-up is unaffected.
        with Embedder._lock:
            missing = [i for i, k in enumerate(keys) if k not in self.cache]
            if missing:
                vecs = self.model.encode([texts[i] for i in missing], batch_size=64, normalize_embeddings=True,
                                         show_progress_bar=False, convert_to_numpy=True)
                for i, v in zip(missing, vecs):
                    self.cache[keys[i]] = v.astype(np.float32)
                self._dirty += len(missing)
                if self._dirty > 500:
                    self._save_unlocked()
            out = np.stack([self.cache[k] for k in keys]).astype(np.float32)
        return out[0] if single else out

    def _save_unlocked(self):
        pickle.dump(self.cache, open(self.path, "wb"))
        self._dirty = 0

    def save(self):
        with Embedder._lock:
            self._save_unlocked()
        self._dirty = 0


# ----------------------------------------------------------------- results
@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    method: str

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def source(self) -> str:
        return self.chunk.source


def rrf_fuse(ranked_lists: list[list[int]], k: int = config.RRF_K, weights: list[float] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion over lists of item ids (best first)."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[int, float] = {}
    for w, lst in zip(weights, ranked_lists):
        for r, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + r)
    return scores


# ----------------------------------------------------------------- index
class HybridIndex:
    """BM25 + FAISS index over the chunks of one source collection."""

    def __init__(self, chunks: list[Chunk], name: str = ""):
        self.name = name
        self.chunks = chunks
        self.embedder = Embedder()
        texts = [self._index_text(c) for c in chunks]
        self.bm25 = BM25Okapi([bm25_tokenize(t) or ["_"] for t in texts]) if chunks else None
        self.emb = self.embedder.encode(texts) if chunks else np.zeros((0, 384), np.float32)
        self.faiss = faiss.IndexFlatIP(self.emb.shape[1])  # inner product on unit vectors == cosine
        if chunks:
            self.faiss.add(self.emb)

    @staticmethod
    def _index_text(c: Chunk) -> str:
        title = c.meta.get("title", "")
        return f"{title}. {c.text}" if title and not c.text.startswith(title) else c.text

    def __len__(self):
        return len(self.chunks)

    def bm25_rank(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        s = self.bm25.get_scores(bm25_tokenize(query) or ["_"])
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]

    def dense_rank(self, query: str | np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        q = self.embedder.encode(query) if isinstance(query, str) else query
        D, I = self.faiss.search(q.reshape(1, -1).astype(np.float32), min(k, len(self.chunks)))
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]

    def search(self, query: str, k: int = 5, method: str = "hybrid", k_candidates: int = 30,
               dense_query: str | np.ndarray | None = None, allowed=None) -> list[Hit]:
        """method in {bm25, dense, hybrid}. `dense_query` lets HyDE embed a different text."""
        kc = max(k_candidates, k)
        if method == "bm25":
            ranked = self.bm25_rank(query, kc)
        elif method == "dense":
            ranked = self.dense_rank(dense_query if dense_query is not None else query, kc)
        elif method == "hybrid":
            b = [i for i, _ in self.bm25_rank(query, kc)]
            d = [i for i, _ in self.dense_rank(dense_query if dense_query is not None else query, kc)]
            fused = rrf_fuse([b, d])
            ranked = sorted(fused.items(), key=lambda x: -x[1])
        else:
            raise ValueError(method)
        hits = []
        for i, s in ranked:
            c = self.chunks[i]
            if allowed is not None and not allowed(c):
                continue
            hits.append(Hit(c, s, len(hits) + 1, method))
            if len(hits) >= k:
                break
        return hits


# ----------------------------------------------------------------- reranker
class CrossEncoderReranker:
    _model = None

    def __init__(self):
        if CrossEncoderReranker._model is None:
            from sentence_transformers import CrossEncoder

            CrossEncoderReranker._model = CrossEncoder(config.CROSS_ENCODER_MODEL, device=config.torch_device())
        self.model = CrossEncoderReranker._model

    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, HybridIndex._index_text(h.chunk)) for h in hits], show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        out = []
        for r, i in enumerate(order[: top_k or len(hits)], start=1):
            h = hits[int(i)]
            out.append(Hit(h.chunk, float(scores[int(i)]), r, h.method + "+ce"))
        return out


# ------------------------------------------------------------ multi-source
ROLE_PERMISSIONS = {
    "guest": {"open"},
    "student": {"open"},
    "researcher": {"open", "paywalled"},  # institutional subscription
    "admin": {"open", "paywalled", "private"},
}


class MultiSourceRetriever:
    """One HybridIndex per knowledge source, with role-based access control."""

    SOURCES = ("academic", "wikipedia", "news", "web")

    def __init__(self, chunks_by_source: dict[str, list[Chunk]], role: str = "researcher"):
        self.indexes = {s: HybridIndex(c, s) for s, c in chunks_by_source.items()}
        self.role = role
        self.reranker = None

    def allowed(self, chunk: Chunk) -> bool:
        return chunk.meta.get("access", "open") in ROLE_PERMISSIONS[self.role]

    def search(self, query: str, sources=("academic", "wikipedia", "news"), k: int = config.TOP_K_PER_SOURCE,
               method: str = "hybrid", dense_query=None) -> dict[str, list[Hit]]:
        return {s: self.indexes[s].search(query, k, method, dense_query=dense_query, allowed=self.allowed)
                for s in sources if s in self.indexes}

    def search_flat(self, query: str, sources=("academic", "wikipedia", "news"), k: int = config.TOP_K_PER_SOURCE,
                    method: str = "hybrid", rerank: bool = False, final_k: int | None = None, dense_query=None) -> list[Hit]:
        per = self.search(query, sources, k, method, dense_query)
        # fuse sources by RRF over their per-source ranks (unweighted here; weighted fusion lives in synthesis.py)
        hits = [h for lst in per.values() for h in lst]
        hits.sort(key=lambda h: (h.rank, -h.score))
        if rerank:
            if self.reranker is None:
                self.reranker = CrossEncoderReranker()
            hits = self.reranker.rerank(query, hits)
        return hits[: final_k] if final_k else hits


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0
