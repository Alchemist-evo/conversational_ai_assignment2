"""Task 3 — query decomposition (CoT) and iterative query refinement.

Decomposition: a CoT prompt makes the model first reason about the *facets* of the question (definitions,
mechanisms, comparisons, evidence, disagreements) and then emit 2-4 self-contained sub-queries.

Iterative refinement: after evidence is retrieved, the model reads compact evidence notes, lists the gaps that
remain and proposes NEW sub-queries for them (or declares it is done). Capped at 3 refinement iterations.
"""
from __future__ import annotations

import time

from . import config
from .evaluation import fact_coverage, redundancy
from .llm import LLM
from .retrieval import Hit, MultiSourceRetriever

DECOMP_PROMPT = """Decompose the research question into sub-queries for a search engine.
Think step by step first: what facets does the question have (definitions, mechanisms, comparisons, evidence for/against, recent developments)? Which facets need separate searches?
Then output 2-4 self-contained, keyword-rich sub-queries (each understandable without the original question).

Research question: {q}

Return JSON: {{"reasoning": "<step-by-step facet analysis>", "sub_queries": ["...", "..."]}}"""

REFINE_PROMPT = """You are refining a literature search.
Research question: {q}
Sub-queries already searched: {done}
Evidence found so far (title: excerpt):
{notes}

Think step by step: which parts of the question are still NOT covered by the evidence (missing mechanisms, missing counter-evidence, missing numbers, unresolved disagreements)? Propose up to 2 NEW sub-queries that target those gaps (different from those already searched). If the evidence already covers everything, set done to true.
Return JSON: {{"gaps": ["..."], "new_sub_queries": ["..."], "done": false}}"""


def decompose(llm: LLM, question: str) -> tuple[list[str], dict]:
    data, r = llm.chat_json(DECOMP_PROMPT.format(q=question), default={}, max_tokens=450, purpose="t3_decompose")
    subs = [s.strip() for s in (data.get("sub_queries") or []) if isinstance(s, str) and len(s.strip()) > 3][:4]
    if len(subs) < 2:  # degrade gracefully: the question itself + a counter-evidence query
        subs = [question, f"{question} evidence limitations criticism"][: max(2, len(subs))]
    return subs, {"reasoning": data.get("reasoning", ""), "input_tokens": r.input_tokens,
                  "output_tokens": r.output_tokens, "latency_s": r.latency_s}


def refine(llm: LLM, question: str, done_queries: list[str], hits: list[Hit]) -> tuple[list[str], bool, dict]:
    seen, notes = set(), []
    for h in hits:
        if h.doc_id in seen:
            continue
        seen.add(h.doc_id)
        notes.append(f"- {h.chunk.meta.get('title', h.doc_id)}: {' '.join(h.chunk.text.split()[:30])}")
        if len(notes) >= 14:
            break
    data, r = llm.chat_json(REFINE_PROMPT.format(q=question, done=done_queries, notes="\n".join(notes)), default={},
                            max_tokens=300, purpose="t3_refine")
    new = [s for s in (data.get("new_sub_queries") or []) if isinstance(s, str) and s.strip() and s not in done_queries][:2]
    return new, bool(data.get("done", False)), {"gaps": data.get("gaps", []), "input_tokens": r.input_tokens,
                                                 "output_tokens": r.output_tokens, "latency_s": r.latency_s}


def retrieve_for(retriever: MultiSourceRetriever, queries: list[str], sources=("academic", "wikipedia", "news"),
                 k: int = config.TOP_K_PER_SOURCE, method: str = "hybrid") -> list[Hit]:
    out = []
    for sq in queries:
        for lst in retriever.search(sq, sources, k, method).values():
            out.extend(lst)
    return out


def iterative_research(llm: LLM, retriever: MultiSourceRetriever, question: str, gold_q: dict | None = None,
                       max_iters: int = config.MAX_REFINEMENT_ITERS, sources=("academic", "wikipedia", "news"),
                       k: int = config.TOP_K_PER_SOURCE) -> dict:
    """Iteration 0 = static decomposition; iterations 1..max_iters = refinement. Records per-iteration metrics."""
    trace = {"question": question, "iterations": []}
    t0 = time.perf_counter()
    subs, meta = decompose(llm, question)
    r_t0 = time.perf_counter()
    hits = retrieve_for(retriever, subs, sources, k)
    ret_lat = time.perf_counter() - r_t0
    all_queries = list(subs)
    toks = meta["input_tokens"] + meta["output_tokens"]
    llm_lat = meta["latency_s"]

    def snapshot(it, new_q, info):
        ids = [h.doc_id for h in hits]
        trace["iterations"].append({
            "iteration": it, "new_sub_queries": new_q, "n_hits": len(hits), "unique_docs": len(set(ids)),
            "coverage": fact_coverage(ids, gold_q) if gold_q else None,
            "redundancy": redundancy([h.chunk.chunk_id for h in hits]),
            "cum_tokens": toks, "cum_llm_latency_s": llm_lat, "cum_retrieval_latency_s": ret_lat, "info": info,
        })

    snapshot(0, subs, {"reasoning": meta["reasoning"]})
    for it in range(1, max_iters + 1):
        new, done, info = refine(llm, question, all_queries, hits)
        toks += info["input_tokens"] + info["output_tokens"]
        llm_lat += info["latency_s"]
        if done or not new:
            snapshot(it, [], {**info, "stopped": "done" if done else "no new sub-queries"})
            break
        r_t0 = time.perf_counter()
        hits += retrieve_for(retriever, new, sources, k)
        ret_lat += time.perf_counter() - r_t0
        all_queries += new
        snapshot(it, new, info)
    trace["hits"] = hits
    trace["all_queries"] = all_queries
    trace["wall_time_s"] = time.perf_counter() - t0
    return trace
