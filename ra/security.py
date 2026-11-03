"""Security layer for the research agent (Task 2, Requirements 5-8).

Defences, each mapped to one of the four research-assistant risks:
  1. Prompt injection via search queries  -> Pydantic validation with injection-pattern detector
                                              + parameterized SQL (never string-formatted)
  2. Unauthorised source access           -> role-based access control on tools and documents
  3. Data exfiltration                    -> owner-scoped notes store, PII redaction on outputs, audit log
  4. Indirect injection in retrieved text -> tool-output sanitiser + <untrusted_source> delimiters
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field

# ----------------------------------------------------------- 1. injection
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |the )?(previous|prior|above|system)",
    r"you are now (in )?(developer|dan|jailbreak|admin) mode",
    r"(reveal|print|show|leak) (me )?(the |your )?(system prompt|hidden instructions|api key|password)",
    r"system (note|prompt|override) to (ai|assistant|llm)",
    r"(return|generate|fabricate|invent) (fake|false|made[- ]up) (citations|sources|references)",
    r"do not mention this instruction",
    r"act as (an? )?(unfiltered|unrestricted)",
    r"\bnew instructions?\s*:",
]
SQLI_PATTERNS = [r"('|\")\s*(or|and)\s+\S+\s*=\s*\S+", r";\s*(drop|delete|update|insert|alter)\b", r"--", r"/\*", r"\bunion\s+select\b"]
_INJ_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
_SQL_RE = [re.compile(p, re.I) for p in SQLI_PATTERNS]


def detect_injection(text: str) -> list[str]:
    return [p.pattern for p in _INJ_RE if p.search(text or "")]


def detect_sqli(text: str) -> list[str]:
    return [p.pattern for p in _SQL_RE if p.search(text or "")]


# ------------------------------------------------- 4. indirect injection
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sanitize_tool_output(text: str) -> tuple[str, int]:
    """Remove instruction-like sentences from retrieved (untrusted) text. Returns (clean_text, n_removed)."""
    removed = 0
    keep = []
    for s in _SENT_SPLIT.split(text or ""):
        if detect_injection(s) or re.search(r"\b(ignore|disregard)\b.*\binstructions?\b", s, re.I):
            removed += 1
            keep.append("[REMOVED: instruction-like text from untrusted source]")
        else:
            keep.append(s)
    return " ".join(keep), removed


def wrap_untrusted(source_id: str, text: str) -> str:
    return f"<untrusted_source id=\"{source_id}\">\n{text}\n</untrusted_source>"


# ---------------------------------------------------------- 3. PII filter
PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone": r"\+?\d[\d\s().-]{8,}\d",
    "student_id": r"\b20\d{2}[A-Z]{2}\d{2}[A-Z]{2}\d{3,4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}


def redact_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for name, pat in PII_PATTERNS.items():
        if re.search(pat, text or ""):
            found.append(name)
            text = re.sub(pat, f"[REDACTED-{name.upper()}]", text)
    return text, found


# ------------------------------------------------------------ 2. RBAC
ROLE_TOOL_PERMISSIONS = {
    "guest": {"search_wikipedia", "search_news", "search_web", "generate_citation", "check_source_credibility"},
    "student": {"search_wikipedia", "search_news", "search_web", "generate_citation", "check_source_credibility",
                "search_academic_db"},
    "researcher": {"search_wikipedia", "search_news", "search_web", "generate_citation", "check_source_credibility",
                   "search_academic_db", "read_research_notes"},
    "admin": {"*"},
}
ROLE_DOC_ACCESS = {"guest": {"open"}, "student": {"open"}, "researcher": {"open", "paywalled"},
                   "admin": {"open", "paywalled", "private"}}
RESTRICTED_SOURCES = re.compile(r"\b(paywall(ed)?|private (database|db|notes)|subscription[- ]only|internal db)\b", re.I)


class SecurityViolation(Exception):
    def __init__(self, defense: str, message: str):
        super().__init__(message)
        self.defense = defense


@dataclass
class AuditLog:
    events: list = field(default_factory=list)

    def log(self, user: str, action: str, decision: str, detail: str = ""):
        self.events.append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": user, "action": action,
                            "decision": decision, "detail": detail[:200]})


def authorize_tool(role: str, tool: str) -> None:
    allowed = ROLE_TOOL_PERMISSIONS.get(role, set())
    if "*" not in allowed and tool not in allowed:
        raise SecurityViolation("RBAC", f"role '{role}' may not call '{tool}'")


# ------------------------------------------------- SQLite academic + notes
class SecureStore:
    """SQLite store for academic paper metadata and private user research notes.

    `lookup_title_secure` uses parameterized queries; `lookup_title_vulnerable` concatenates user input into SQL
    and exists ONLY to demonstrate the attack in the notebook."""

    def __init__(self, academic_docs: list[dict]):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("CREATE TABLE papers (doc_id TEXT PRIMARY KEY, title TEXT, year INT, study_type TEXT, access TEXT)")
        self.db.execute("CREATE TABLE notes (note_id INTEGER PRIMARY KEY, owner TEXT, title TEXT, content TEXT, access TEXT)")
        self.db.executemany("INSERT INTO papers VALUES (?,?,?,?,?)",
                            [(d["doc_id"], d["title"], d["year"], d["study_type"], d["access"]) for d in academic_docs])
        self.db.executemany("INSERT INTO notes (owner,title,content,access) VALUES (?,?,?,?)", [
            ("alice", "Draft thesis notes", "Unpublished result: effect size 0.42. Contact alice.w@uni.example.edu, +91 98765 43210.", "private"),
            ("bob", "Grant idea", "Idea for microplastics grant; reviewer list confidential. Student ID 2023AB12CD345.", "private"),
            ("alice", "Reading list", "Papers on intermittent fasting to read next week.", "private"),
        ])
        self.db.commit()

    # --- metadata filtering for search_academic_db (parameterized) ---
    def filter_papers(self, doc_ids: list[str], year_from: int | None, study_type: str | None, access_levels: set[str]):
        if not doc_ids:
            return []
        q = f"SELECT doc_id FROM papers WHERE doc_id IN ({','.join('?' * len(doc_ids))}) AND year >= ? AND access IN ({','.join('?' * len(access_levels))})"
        params = list(doc_ids) + [year_from or 0] + sorted(access_levels)
        if study_type:
            q += " AND study_type = ?"
            params.append(study_type)
        keep = {r[0] for r in self.db.execute(q, params)}
        return [d for d in doc_ids if d in keep]

    def lookup_title_vulnerable(self, fragment: str):
        sql = f"SELECT doc_id, title, access FROM papers WHERE title LIKE '%{fragment}%' AND access = 'open'"
        try:
            return self.db.execute(sql).fetchall()
        except sqlite3.Error as e:
            return [("SQL ERROR", str(e), "")]

    def lookup_title_secure(self, fragment: str):
        return self.db.execute("SELECT doc_id, title, access FROM papers WHERE title LIKE ? AND access = 'open'",
                               (f"%{fragment}%",)).fetchall()

    def read_notes(self, requester: str, owner: str):
        if requester != owner:
            raise SecurityViolation("DataAccessControl", f"user '{requester}' may not read notes of '{owner}'")
        return self.db.execute("SELECT title, content FROM notes WHERE owner = ?", (owner,)).fetchall()


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
