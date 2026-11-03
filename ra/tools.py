"""Task 2 — research tools with JSON Schemas, Pydantic validation, RBAC and simulated network behaviour.

The knowledge sources are local collections (so results are reproducible), but each tool behaves like a remote
API: it has a seeded, simulated network latency and an optional seeded failure rate. This makes the
sequential-vs-parallel timing and the failure-handling experiments meaningful. Both are clearly reported as
simulated in the notebook.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from . import config
from .credibility import credibility, credibility_from_url, register_domains
from .security import (ROLE_DOC_ACCESS, SecureStore, SecurityViolation, authorize_tool, detect_injection, detect_sqli,
                       redact_pii, sanitize_tool_output)

# =========================================================== argument models


class _QueryMixin(BaseModel):
    @field_validator("query", "topic", check_fields=False)
    @classmethod
    def _clean_query(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v).strip()
        if len(v) < 3:
            raise ValueError("query too short (min 3 chars)")
        if len(v) > 300:
            raise ValueError("query too long (max 300 chars)")
        if detect_injection(v):
            raise ValueError("prompt-injection pattern detected in query")
        if detect_sqli(v):
            raise ValueError("SQL-injection pattern detected in query")
        return v


class SearchAcademicArgs(_QueryMixin):
    query: str = Field(..., description="Focused research query")
    max_results: int = Field(5, ge=1, le=10)
    year_from: Optional[int] = Field(None, ge=1900, le=2026)
    study_type: Optional[Literal["meta-analysis", "systematic review", "randomised controlled trial", "cohort study",
                                 "narrative review", "assessment report"]] = None
    sort_by: Literal["relevance", "year"] = "relevance"


class SearchWikipediaArgs(_QueryMixin):
    topic: str = Field(..., description="Encyclopedic topic name")
    max_results: int = Field(3, ge=1, le=5)


class SearchNewsArgs(_QueryMixin):
    query: str
    max_results: int = Field(5, ge=1, le=10)
    recency_days: Optional[int] = Field(None, ge=1, le=36500)


class SearchWebArgs(_QueryMixin):
    query: str
    max_results: int = Field(5, ge=1, le=10)


class CredibilityArgs(BaseModel):
    source_url: str = Field(..., pattern=r"^https?://[\w.-]+(/[\w./%#?=&+-]*)?$", max_length=300)


class CitationArgs(BaseModel):
    source_id: str = Field(..., pattern=r"^[A-Za-z0-9_#:.\-]{2,80}$")
    style: Literal["APA", "MLA"] = "APA"


class NotesArgs(BaseModel):
    owner: str = Field(..., pattern=r"^[a-z][a-z0-9_]{1,30}$")


# ============================================================== JSON schemas
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_academic_db",
        "description": ("Search peer-reviewed papers and abstracts. Use for scientific evidence, effect sizes and study "
                        "designs. GOOD query: 'intermittent fasting vs calorie restriction weight loss RCT'. BAD query: "
                        "'tell me everything about diets' (too vague) or a full chatty sentence."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 3, "maxLength": 300, "description": "Focused keyword-style research query."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "year_from": {"type": "integer", "minimum": 1900, "maximum": 2026, "description": "Only papers published in or after this year."},
            "study_type": {"type": "string", "enum": ["meta-analysis", "systematic review", "randomised controlled trial", "cohort study", "narrative review", "assessment report"]},
            "sort_by": {"type": "string", "enum": ["relevance", "year"], "default": "relevance"}},
            "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "search_wikipedia",
        "description": ("Get encyclopedic background excerpts on a topic. GOOD topic: 'C4 carbon fixation'. BAD topic: "
                        "'is fasting good or bad for me personally' (opinion question, not a topic)."),
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "minLength": 3, "maxLength": 300},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3}},
            "required": ["topic"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "search_news",
        "description": ("Search recent news articles (credibility varies - verify with check_source_credibility). GOOD: "
                        "'aspartame IARC classification 2023'. BAD: 'news' (no subject)."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 3, "maxLength": 300},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "recency_days": {"type": "integer", "minimum": 1, "maximum": 36500, "description": "Only articles from the last N days."}},
            "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "check_source_credibility",
        "description": "Return a 0-1 credibility score with authority / evidence-quality / recency components for a source URL.",
        "parameters": {"type": "object", "properties": {"source_url": {"type": "string", "format": "uri", "maxLength": 300}},
                       "required": ["source_url"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "generate_citation",
        "description": "Format a citation for a retrieved source id (e.g. 'D1-A1' or 'WIKI-Vaccine').",
        "parameters": {"type": "object", "properties": {
            "source_id": {"type": "string", "pattern": "^[A-Za-z0-9_#:.\\-]{2,80}$"},
            "style": {"type": "string", "enum": ["APA", "MLA"], "default": "APA"}},
            "required": ["source_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "General web search (forums, blogs, news mirrors). LOW trust - use as a fallback or to find what is being claimed publicly.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 3, "maxLength": 300},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}},
            "required": ["query"], "additionalProperties": False}}},
]
ARG_MODELS = {"search_academic_db": SearchAcademicArgs, "search_wikipedia": SearchWikipediaArgs,
              "search_news": SearchNewsArgs, "search_web": SearchWebArgs, "check_source_credibility": CredibilityArgs,
              "generate_citation": CitationArgs, "read_research_notes": NotesArgs}
FALLBACKS = {"search_news": "search_web", "search_academic_db": "search_wikipedia", "search_wikipedia": "search_web"}
LATENCY = {  # simulated network latency ranges (seconds)
    "search_academic_db": (0.6, 1.2), "search_wikipedia": (0.3, 0.7), "search_news": (0.4, 0.9),
    "search_web": (0.5, 1.0), "check_source_credibility": (0.1, 0.25), "generate_citation": (0.02, 0.06),
    "read_research_notes": (0.05, 0.1),
}


# ================================================================ citations
def _apa_authors(authors: list[str]) -> str:
    if not authors:
        return "Anonymous"
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + ", & " + authors[-1]


def format_citation(doc: dict, style: str = "APA") -> str:
    syn = " [Synthetic source]" if doc.get("synthetic") else ""
    title, venue, url, year = doc["title"], doc.get("venue", ""), doc.get("url", ""), doc.get("year", "n.d.")
    if style == "APA":
        if doc["source"] == "news" or doc["source"] == "web":
            import datetime as _dt

            try:
                d = _dt.date.fromisoformat(doc["date"])
                when = f"{d.year}, {d.strftime('%B')} {d.day}"
            except Exception:
                when = str(year)
            return f"{_apa_authors(doc.get('authors', []))} ({when}). {title}. {venue}. {url}{syn}"
        if doc["source"] == "wikipedia":
            return f"Wikipedia contributors. ({year}). {title}. In {venue}. {url}"
        return f"{_apa_authors(doc.get('authors', []))} ({year}). {title}. {venue}. {url}{syn}"
    # MLA 9
    a = doc.get("authors", [])
    auth = (a[0] + (", et al." if len(a) > 2 else (", and " + a[1] if len(a) == 2 else ""))) if a else ""
    return f"{auth}. \"{title}.\" {venue}, {year}, {url}.{syn}".lstrip(". ")


# ================================================================== registry
@dataclass
class ToolResult:
    tool: str
    args: dict
    ok: bool
    output: object = None
    error: str = ""
    latency_s: float = 0.0
    blocked_by: str = ""
    sanitized: int = 0
    ts: str = ""

    @property
    def output_size(self) -> int:
        return len(json.dumps(self.output, ensure_ascii=False)) if self.output is not None else 0


class ToolError(Exception):
    pass


class ToolRegistry:
    def __init__(self, data, retriever, role: str = "researcher", user: str = "researcher_1",
                 failure_rates: dict | None = None, simulate_latency: bool = True, seed: int = config.SEED):
        self.data = data
        self.retriever = retriever
        self.role = role
        self.user = user
        self.failure_rates = failure_rates or {}
        self.simulate_latency = simulate_latency
        self.seed = seed
        self.store = SecureStore([d for d in data.docs.values() if d["source"] == "academic"])
        self.docs_by_url = {d["url"]: d for d in data.docs.values()}
        register_domains(data.docs.values())
        self._attempts: dict[str, int] = {}
        self.log: list[ToolResult] = []
        self.audit = None

    # ------------------------------------------------------------ helpers
    def _rng(self, tool: str, args: dict, attempt: int) -> random.Random:
        h = hashlib.md5(f"{self.seed}|{tool}|{json.dumps(args, sort_keys=True)}|{attempt}".encode()).hexdigest()
        return random.Random(int(h, 16))

    def _doc_view(self, hit, max_words=120) -> dict:
        d = hit.chunk.meta
        text, n_removed = sanitize_tool_output(hit.chunk.text)
        words = text.split()
        return {"source_id": hit.doc_id, "title": d.get("title"), "venue": d.get("venue"), "year": d.get("year"),
                "url": d.get("url"), "source_type": hit.source, "chunk_id": hit.chunk.chunk_id,
                "excerpt": " ".join(words[:max_words]) + (" ..." if len(words) > max_words else ""),
                "sanitized_sentences": n_removed}

    # ------------------------------------------------------------- tools
    def search_academic_db(self, a: SearchAcademicArgs):
        hits = self.retriever.indexes["academic"].search(a.query, k=30, allowed=self.retriever.allowed)
        ids = [h.doc_id for h in hits]
        keep = self.store.filter_papers(ids, a.year_from, a.study_type, ROLE_DOC_ACCESS[self.role])
        hits = [h for h in hits if h.doc_id in keep][: a.max_results]
        if a.sort_by == "year":
            hits.sort(key=lambda h: -h.chunk.meta.get("year", 0))
        return [self._doc_view(h) for h in hits]

    def search_wikipedia(self, a: SearchWikipediaArgs):
        return [self._doc_view(h) for h in self.retriever.indexes["wikipedia"].search(a.topic, k=a.max_results)]

    def search_news(self, a: SearchNewsArgs):
        hits = self.retriever.indexes["news"].search(a.query, k=20)
        if a.recency_days:
            from .credibility import _age_years

            hits = [h for h in hits if _age_years(h.chunk.meta.get("date", "2000-01-01")) * 365.25 <= a.recency_days]
        return [self._doc_view(h) for h in hits[: a.max_results]]

    def search_web(self, a: SearchWebArgs):
        return [self._doc_view(h) for h in self.retriever.indexes["web"].search(a.query, k=a.max_results)]

    def check_source_credibility(self, a: CredibilityArgs):
        return credibility_from_url(a.source_url, self.docs_by_url)

    def generate_citation(self, a: CitationArgs):
        doc_id = a.source_id.split("#")[0].replace("web:", "")
        if doc_id not in self.data.docs:
            raise ToolError(f"unknown source_id {a.source_id}")
        return {"source_id": doc_id, "style": a.style, "citation": format_citation(self.data.docs[doc_id], a.style)}

    def read_research_notes(self, a: NotesArgs):
        rows = self.store.read_notes(self.user, a.owner)
        return [{"title": t, "content": redact_pii(c)[0]} for t, c in rows]

    # -------------------------------------------------------------- call
    def call(self, tool: str, args: dict) -> ToolResult:
        t0 = time.perf_counter()
        ts = time.strftime("%H:%M:%S")
        res = ToolResult(tool, args, ok=False, ts=ts)
        try:
            if tool not in ARG_MODELS:
                raise ToolError(f"unknown tool '{tool}'")
            authorize_tool(self.role, tool)
            try:
                parsed = ARG_MODELS[tool](**(args or {}))
            except ValidationError as e:
                msg = "; ".join(err["msg"] for err in e.errors())
                defense = "PydanticValidation"
                res.blocked_by = defense if ("injection" in msg) else ""
                raise ToolError(f"invalid arguments: {msg}")
            key = f"{tool}|{json.dumps(args, sort_keys=True)}"
            attempt = self._attempts.get(key, 0)
            self._attempts[key] = attempt + 1
            rng = self._rng(tool, args, attempt)
            if self.simulate_latency:
                lo, hi = LATENCY.get(tool, (0.1, 0.2))
                time.sleep(rng.uniform(lo, hi))
            if rng.random() < self.failure_rates.get(tool, 0.0):
                raise ToolError("503 Service Unavailable (simulated upstream failure)")
            out = getattr(self, tool)(parsed)
            res.ok, res.output = True, out
            res.sanitized = sum(o.get("sanitized_sentences", 0) for o in out) if isinstance(out, list) else 0
        except SecurityViolation as e:
            res.error, res.blocked_by = str(e), e.defense
        except ToolError as e:
            res.error = str(e)
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
        res.latency_s = time.perf_counter() - t0
        self.log.append(res)
        return res

    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS
