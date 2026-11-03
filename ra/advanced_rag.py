"""Task 3, Requirement 14 — advanced RAG variants implemented from scratch: HyDE and CRAG
(latency = measured retrieval compute + recorded LLM latency, so cached replays report live numbers;
compared against the baseline hybrid BM25 + dense + RRF pipeline of Requirement 6).

HyDE  : the LLM writes a *hypothetical* answer passage; its embedding (which "looks like" an answer document)
        drives dense retrieval, while BM25 still uses the original question; the two are fused with RRF.
CRAG  : a retrieval evaluator (LLM) scores every retrieved chunk's relevance 0-1, then
          CORRECT   (max >= 0.7)          -> knowledge refinement: keep only chunks scoring >= 0.5
          INCORRECT (max <  0.3)          -> discard and fall back to web search with a rewritten query
          AMBIGUOUS (otherwise)           -> keep chunks >= 0.5 AND add web-search results
"""
from __future__ import annotations

import time

from . import config
from .llm import LLM
from .retrieval import Embedder, Hit, MultiSourceRetriever

SOURCES = ("academic", "wikipedia", "news")


def baseline(retriever: MultiSourceRetriever, question: str, k: int = config.TOP_K_PER_SOURCE) -> dict:
    t0 = time.perf_counter()
    hits = retriever.search_flat(question, SOURCES, k)
    return {"hits": hits, "tokens": 0, "llm_latency": 0.0, "retrieval_s": time.perf_counter() - t0, "info": {}}


def hyde(llm: LLM, retriever: MultiSourceRetriever, question: str, k: int = config.TOP_K_PER_SOURCE) -> dict:
    t0 = time.perf_counter()
    r = llm.chat(f"Write a concise, factual encyclopedia-style paragraph (about 120 words) that answers: {question}\n"
                 "Include specific mechanisms, numbers and study findings as an expert would.",
                 purpose="t3_hyde", max_tokens=220)
    hypo = r.text.strip()
    t1 = time.perf_counter()
    vec = Embedder().encode(hypo)
    hits = retriever.search_flat(question, SOURCES, k, dense_query=vec)
    return {"hits": hits, "tokens": r.input_tokens + r.output_tokens, "llm_latency": r.latency_s,
            "retrieval_s": time.perf_counter() - t1, "info": {"hypothetical_document": hypo}}


def crag(llm: LLM, retriever: MultiSourceRetriever, question: str, k: int = config.TOP_K_PER_SOURCE,
         upper: float = 0.7, lower: float = 0.3, keep: float = 0.5) -> dict:
    t0 = time.perf_counter()
    hits = retriever.search_flat(question, SOURCES, k)
    ret_s = time.perf_counter() - t0
    listing = "\n".join(f"{i}. {h.chunk.meta.get('title', '')}: {' '.join(h.chunk.text.split()[:45])}"
                        for i, h in enumerate(hits, 1))
    data, r = llm.chat_json(f"Question: {question}\n\nRetrieved passages:\n{listing}\n\nRate how useful each passage is "
                            "for answering the question (0 = irrelevant, 1 = directly answers part of it). Return JSON: "
                            "{\"scores\": [float per passage in order], \"rewritten_query\": \"a better web search query\"}",
                            default={}, purpose="t3_crag_evaluate", max_tokens=300)
    tokens, llm_lat = r.input_tokens + r.output_tokens, r.latency_s
    scores = []
    for x in (data.get("scores") or []):
        try:
            scores.append(float(x))
        except Exception:
            scores.append(0.0)
    scores = (scores + [0.0] * len(hits))[: len(hits)]
    mx = max(scores) if scores else 0.0
    kept = [h for h, s in zip(hits, scores) if s >= keep]
    action = "CORRECT" if mx >= upper else ("INCORRECT" if mx < lower else "AMBIGUOUS")
    web = []
    if action in ("INCORRECT", "AMBIGUOUS"):
        q2 = data.get("rewritten_query") or question
        t1 = time.perf_counter()
        web = retriever.indexes["web"].search(q2, k=k, allowed=retriever.allowed)
        ret_s += time.perf_counter() - t1
    final = (kept if action != "INCORRECT" else []) + web
    return {"hits": final, "tokens": tokens, "llm_latency": llm_lat, "retrieval_s": ret_s,
            "info": {"action": action, "scores": scores, "max_score": mx, "kept": len(kept), "web_added": len(web)}}
