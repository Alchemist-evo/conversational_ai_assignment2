"""Task 6 — the integrated ResearchAssistant.

Pipeline for one research query (each stage is a component built in Tasks 1-4):

  security check (T2) -> adaptive routing (T3: Adaptive RAG idea) -> episodic recall (T3)
  -> CoT decomposition (T1/T3) -> parallel multi-source tool calls with safety + fallbacks (T2)
  -> iterative refinement (T3) -> credibility-weighted RRF fusion + cross-encoder re-ranking (T3/T4)
  -> context engineering (T3) -> conflict detection & resolution (T4) -> cited synthesis (T4)
  -> claim verification + self-correction (T4) -> confidence calibration (T4) -> citations (T2 tool)
  -> memory update (working + episodic) -> audit-trail record (T6)

Multi-turn: every user message is first classified as NEW research, a REFORMULATION of the previous answer
(e.g. "explain in simpler terms" - no retrieval needed) or a FOLLOW-UP needing new research; follow-ups are
rewritten into standalone queries using the conversation history (coreference resolution).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import config
from .chunking import Chunk
from .credibility import credibility
from .decomposition import decompose, refine
from .evaluation import percentiles
from .llm import LLM
from .memory import EpisodicMemory, WorkingMemory
from .retrieval import CrossEncoderReranker, Hit
from .security import detect_injection, redact_pii
from .synthesis import SOURCE_WEIGHT_PRESETS, SynthesisResult, full_synthesis, hits_to_evidence, synthesize, weighted_rrf
from .tools import ToolRegistry, format_citation

ROUTER_PROMPT = """Conversation so far:
{history}

New user message: "{msg}"

Classify the new message:
- "reformulate": asks to re-express the previous answer (simpler, shorter, translate, bullet points) - no new facts needed.
- "follow_up": continues the previous topic but needs NEW information (e.g. critics, comparisons, evidence for X).
- "new_research": a new research question.
Also rewrite it as a fully self-contained research query (resolve pronouns like "that" using the conversation), and rate complexity: "simple" (single fact) or "complex" (multi-faceted / contested).
Return JSON: {{"type": "...", "standalone_query": "...", "complexity": "simple|complex"}}"""


@dataclass
class AuditRecord:
    question: str
    started: str
    route: dict = field(default_factory=dict)
    sub_queries: dict = field(default_factory=lambda: {"static": [], "refinements": []})
    tool_calls: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    final_synthesis: str = ""
    overall_confidence: float = 0.0
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__

    @property
    def traceability(self) -> float:
        if not self.claims:
            return 0.0
        ok = [bool(c.get("sources")) and all(s.get("url") and s.get("evidence_snippet") for s in c["sources"]) and c.get("verdict") == "SUPPORTED"
              for c in self.claims]
        return float(np.mean(ok))

    def render_markdown(self) -> str:
        L = [f"### Research audit report", f"**Question:** {self.question}  ", f"**Started:** {self.started}  ",
             f"**Route:** {self.route.get('type')} / complexity={self.route.get('complexity')}  ",
             f"**Static sub-queries:** " + "; ".join(self.sub_queries["static"])]
        for i, r in enumerate(self.sub_queries["refinements"], 1):
            L.append(f"**Refinement {i}:** " + ("; ".join(r) if r else "(stopped - no gaps)"))
        L.append("\n**Tool calls**\n\n| # | tool | args | time | latency (s) | output (chars) | status |\n|---|---|---|---|---|---|---|")
        for i, t in enumerate(self.tool_calls, 1):
            L.append(f"| {i} | {t['tool']} | `{json.dumps(t['args'])[:60]}` | {t['timestamp']} | {t['latency_s']:.2f} | {t['output_size']} | {'ok' if t['ok'] else t['error'][:30]} |")
        L.append("\n**Sources used**\n\n| S-id | title | credibility | url |\n|---|---|---|---|")
        for s in self.sources:
            L.append(f"| {s['sid']} | {s['title'][:70]} | {s['credibility']:.2f} | {s['url']} |")
        if self.conflicts:
            L.append("\n**Conflicts resolved**")
            for c in self.conflicts:
                L.append(f"- {c['issue']} -> {c['decision']} (share {c['share']:.2f})")
        L.append("\n**Claims (evidence trail)**\n\n| # | claim | verdict | confidence | supporting source(s) |\n|---|---|---|---|---|")
        for i, c in enumerate(self.claims, 1):
            srcs = ", ".join(f"{s['sid']} {s['title'][:35]}" for s in c["sources"]) or "-"
            L.append(f"| {i} | {c['text'][:110]} | {c['verdict']} | {c['confidence']:.2f} | {srcs} |")
        L.append(f"\n**Final synthesis** (overall confidence {self.overall_confidence:.2f}):\n\n> {self.final_synthesis}")
        L.append(f"\n**Claim-to-source traceability:** {100*self.traceability:.0f}%  |  latency {self.metrics.get('latency_s', 0):.1f}s  |  "
                 f"tokens in/out {self.metrics.get('input_tokens', 0)}/{self.metrics.get('output_tokens', 0)}  |  "
                 f"est. cost ${self.metrics.get('cost_usd', 0):.4f}")
        return "\n".join(L)


class ResearchAssistant:
    def __init__(self, llm: LLM, data, retriever, episodic: EpisodicMemory, tools: ToolRegistry,
                 rerank: bool = True, refine_iters: int = config.MAX_REFINEMENT_ITERS, verify: bool = True,
                 weights: str = "credibility", top_evidence: int = 8, user: str = "researcher_1"):
        self.llm, self.data, self.retriever, self.episodic, self.tools = llm, data, retriever, episodic, tools
        self.rerank, self.refine_iters, self.verify = rerank, refine_iters, verify
        self.weights = SOURCE_WEIGHT_PRESETS[weights]
        self.top_evidence = top_evidence
        self.user = user
        self.history: list[dict] = []
        self.working = WorkingMemory(llm=llm)
        self.audits: list[AuditRecord] = []
        self.reranker = CrossEncoderReranker() if rerank else None
        self.pool = ThreadPoolExecutor(max_workers=6)
        self.last_result: SynthesisResult | None = None

    # ----------------------------------------------------------- helpers
    def _tool_batch(self, calls: list[tuple[str, dict]], audit: AuditRecord) -> list:
        results = list(self.pool.map(lambda c: self.tools.call(*c), calls))  # parallel function calling
        for r in results:
            audit.tool_calls.append({"tool": r.tool, "args": r.args, "timestamp": r.ts, "latency_s": r.latency_s,
                                     "output_size": r.output_size, "ok": r.ok, "error": r.error})
        return results

    def _search(self, queries: list[str], audit: AuditRecord) -> dict[str, list[Hit]]:
        calls = []
        for q in queries:
            calls += [("search_academic_db", {"query": q, "max_results": 5}),
                      ("search_wikipedia", {"topic": q, "max_results": 5}),
                      ("search_news", {"query": q, "max_results": 5})]
        per_source: dict[str, list[Hit]] = {"academic": [], "wikipedia": [], "news": []}
        results = self._tool_batch(calls, audit)
        failed = [r for r in results if not r.ok and not r.blocked_by]
        if failed:  # graceful degradation via fallback source
            fb = self._tool_batch([("search_web", {"query": r.args.get("query") or r.args.get("topic")}) for r in failed], audit)
            results += fb
        src_of = {"search_academic_db": "academic", "search_wikipedia": "wikipedia", "search_news": "news", "search_web": "news"}
        for r in results:
            if not r.ok:
                continue
            for rank, o in enumerate(r.output, 1):
                doc = self.data.docs.get(o["source_id"], {})
                meta = {k: v for k, v in doc.items() if k != "text"}
                ch = Chunk(o["chunk_id"], o["source_id"], o["source_type"], o["excerpt"], meta)
                per_source[src_of[r.tool]].append(Hit(ch, 0.0, rank, r.tool))
        return per_source

    def route(self, message: str) -> dict:
        if not self.history:
            hist = "(none)"
        else:
            hist = "\n".join(f"User: {t['user']}\nAssistant: {t['assistant'][:300]}" for t in self.history[-3:])
        data, _ = self.llm.chat_json(ROUTER_PROMPT.format(history=hist, msg=message), default={}, max_tokens=150,
                                     purpose="t6_route")
        typ = data.get("type") if data.get("type") in ("reformulate", "follow_up", "new_research") else "new_research"
        if not self.history:
            typ = "new_research"
        return {"type": typ, "standalone_query": data.get("standalone_query") or message,
                "complexity": data.get("complexity", "complex")}

    # ------------------------------------------------------------- core
    def research(self, question: str, mode: str = "auto", route: dict | None = None, style: str = "",
                 extra_history: bool = True) -> dict:
        t0 = time.perf_counter()
        start_calls = len(self.llm.calls)
        audit = AuditRecord(question, time.strftime("%Y-%m-%dT%H:%M:%S"))
        if detect_injection(question):
            audit.final_synthesis = "Request refused: the query contains prompt-injection patterns."
            audit.route = {"type": "blocked"}
            self.audits.append(audit)
            return {"answer": audit.final_synthesis, "audit": audit, "blocked": True}
        route = route or ({"type": "new_research", "standalone_query": question, "complexity": "complex"} if mode == "full"
                          else {"type": "new_research", "standalone_query": question, "complexity": "simple"} if mode == "fast"
                          else self.route(question))
        audit.route = route
        q = route["standalone_query"]
        complex_q = route.get("complexity", "complex") != "simple"

        # episodic recall (top-2 into the context)
        epi = self.episodic.retrieve(q, k=2)
        epi_text = EpisodicMemory.render(epi)

        # decomposition + parallel multi-source retrieval + iterative refinement
        if complex_q:
            subs, _ = decompose(self.llm, q)
        else:
            subs = [q]
        audit.sub_queries["static"] = subs
        per_source = self._search(subs, audit)
        all_q = list(subs)
        if complex_q:
            for _ in range(self.refine_iters):
                flat = [h for lst in per_source.values() for h in lst]
                new, done, _info = refine(self.llm, q, all_q, flat)
                audit.sub_queries["refinements"].append(new)
                if done or not new:
                    break
                more = self._search(new, audit)
                for s in per_source:
                    per_source[s] += more[s]
                all_q += new
        self.working.add({"type": "plan", "content": f"Q: {q} | sub-queries: {all_q}"})

        # fusion: per-source best rank -> credibility-weighted RRF -> cross-encoder re-ranking
        for s, lst in per_source.items():
            best = {}
            for h in lst:
                if h.doc_id not in best or h.rank < best[h.doc_id].rank:
                    best[h.doc_id] = h
            per_source[s] = sorted(best.values(), key=lambda h: h.rank)
        fused = [h for h, _ in weighted_rrf(per_source, self.weights, doc_level_credibility=True)][:20]
        if self.reranker is not None and fused:
            fused = self.reranker.rerank(q, fused)
        evidence = hits_to_evidence(fused[: self.top_evidence if complex_q else 4])
        # credibility tool (logged) for every source we are going to use
        self._tool_batch([("check_source_credibility", {"source_url": e["url"]}) for e in evidence if e["url"].startswith("http")], audit)

        # context engineering: episodic memory + history + working memory as additional context
        ctx_parts = []
        if epi_text:
            ctx_parts.append("Relevant past research sessions (episodic memory):\n" + epi_text)
        if extra_history and self.history:
            ctx_parts.append("Conversation so far:\n" + "\n".join(f"User: {t['user']}\nAssistant: {t['assistant'][:400]}" for t in self.history[-3:]))
        wm = self.working.render()
        if wm:
            ctx_parts.append("Working memory:\n" + wm)
        res = full_synthesis(self.llm, q, evidence, resolve=complex_q, verify=self.verify, extra_context="\n\n".join(ctx_parts),
                             style=style)
        self.last_result = res

        # citations via tool (logged)
        used = sorted({s for c in res.claims_after for s in (c.get("support") or c.get("cited") or [])})
        sid2doc = {e["sid"]: e["doc_id"] for e in evidence}
        cit_results = self._tool_batch([("generate_citation", {"source_id": sid2doc[s]}) for s in used if s in sid2doc], audit)
        citations = {r.output["source_id"]: r.output["citation"] for r in cit_results if r.ok}

        # audit record
        ev = {e["sid"]: e for e in evidence}
        audit.sources = [{"sid": e["sid"], "doc_id": e["doc_id"], "title": e["title"], "url": e["url"],
                          "credibility": e["credibility"], "source": e["source"]} for e in evidence]
        audit.conflicts = [{"issue": c["issue"], "decision": c["resolution"]["winning_stance"] or "contested",
                            "share": c["resolution"]["share"]} for c in res.conflicts]
        for c in res.claims_after:
            srcs = [{"sid": s, "title": ev[s]["title"], "url": ev[s]["url"], "evidence_snippet": ev[s]["text"][:240],
                     "citation": citations.get(ev[s]["doc_id"], format_citation(ev[s]["meta"]))}
                    for s in (c.get("support") or []) if s in ev]
            audit.claims.append({"text": c["text"], "verdict": c.get("verdict", "UNVERIFIED"), "confidence": c["confidence"],
                                 "sources": srcs})
        answer, pii = redact_pii(res.answer)  # output filter (exfiltration defence)
        audit.final_synthesis = answer
        audit.overall_confidence = res.overall_confidence
        u = self.llm.usage_since(start_calls)
        latency = time.perf_counter() - t0
        # replayed (cached) LLM calls take ~0 s, so add their recorded latency to report live-equivalent latency
        cached_lat = sum(c.latency_s for c in self.llm.calls[start_calls:] if c.cached)
        audit.metrics = {"latency_s": latency + cached_lat, "llm_calls": u["calls"], "input_tokens": u["input_tokens"],
                         "output_tokens": u["output_tokens"],
                         "cost_usd": config.cost_usd(u["input_tokens"], u["output_tokens"], "gpt-4o"),
                         "n_tool_calls": len(audit.tool_calls), "pii_redacted": pii,
                         "halluc_before": res.halluc_rate_before, "halluc_after": res.halluc_rate_after}
        self.audits.append(audit)

        # memory update
        self.working.add({"type": "finding", "content": answer[:300]})
        self.episodic.add({"session_id": f"LIVE-{len(self.audits):03d}", "synthetic": False, "date": config.RUN_DATE,
                           "user_query": q, "key_findings": [c["text"] for c in res.claims_after[:3]],
                           "user_preference": "-", "sources_consulted": [e["doc_id"] for e in evidence]})
        return {"answer": answer, "audit": audit, "result": res, "evidence": evidence, "route": route, "blocked": False}

    def chat(self, message: str) -> dict:
        route = self.route(message)
        if route["type"] == "reformulate" and self.last_result is not None:
            t0 = time.perf_counter()
            start = len(self.llm.calls)
            prev = self.history[-1]["assistant"]
            text, _ = synthesize(self.llm, f"{message} (Previous answer to rewrite: \"{prev}\")", self.last_result.evidence,
                                 extra_context="Rewrite the previous answer as requested, keeping citations.",
                                 style=message, purpose="t6_reformulate")
            u = self.llm.usage_since(start)
            cached_lat = sum(c.latency_s for c in self.llm.calls[start:] if c.cached)
            out = {"answer": text, "route": route, "evidence": self.last_result.evidence, "audit": None,
                   "metrics": {"latency_s": time.perf_counter() - t0 + cached_lat, **u}}
        else:
            out = self.research(message, route=route)
            out["metrics"] = out["audit"].metrics
        self.history.append({"user": message, "assistant": out["answer"], "route": route})
        return out
