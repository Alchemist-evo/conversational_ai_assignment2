"""Source credibility model (Task 4, Requirement 1) — used by the credibility tool and by synthesis.

credibility(doc) = 0.50 * authority + 0.35 * evidence_quality + 0.15 * recency      (all in [0, 1])

* authority        — who published it (peer-reviewed journal > encyclopedia > reputable news > blog/tabloid/PR/forum).
* evidence_quality — study design hierarchy (meta-analysis/systematic review > RCT > observational > narrative >
                     news report > opinion > anecdote).
* recency          — 0.5 + 0.5 * 2^(-age / half_life); half-life 10 y for scholarly/encyclopedic content, 4 y for
                     news/web. Recency is deliberately a small weight: an old meta-analysis should still beat a
                     new tabloid story.
Query-dependent *relevance* is not part of the static score; it enters through the retrieval rank in weighted RRF.
The weights are design choices (justified in the notebook) — not fixed rules — and are varied in experiments.
"""
from __future__ import annotations

import math
from datetime import date
from urllib.parse import urlparse

from . import config

AUTHORITY = {
    "journal": 0.90, "encyclopedia": 0.70, "reputable": 0.55, "opinion": 0.30, "industry_pr": 0.20,
    "blog": 0.15, "tabloid": 0.15, "user_generated": 0.10, "unknown": 0.30,
}
EVIDENCE = {
    "meta-analysis": 1.0, "systematic review": 1.0, "umbrella review": 1.0, "assessment report": 0.95,
    "randomised controlled trial": 0.9, "clinical trial report": 0.85, "life-cycle assessment": 0.75,
    "cohort study": 0.7, "prospective observational study": 0.65, "natural experiment": 0.7, "empirical analysis": 0.7,
    "modelling study": 0.7, "benchmark study": 0.7, "field study": 0.7, "observational analysis": 0.7,
    "narrative review": 0.75, "guideline": 0.8, "clinical guideline": 0.75, "expert evaluation": 0.85,
    "official inquiry": 0.8, "encyclopedia": 0.6, "news report": 0.4, "opinion/unverified": 0.15, "anecdote": 0.05,
}
W_AUTH, W_EVID, W_REC = 0.50, 0.35, 0.15

KNOWN_DOMAINS = {
    "journals.example.org": "journal", "en.wikipedia.org": "encyclopedia", "huggingface.co": "encyclopedia",
    "forum.example.net": "user_generated",
}


def register_domains(docs) -> None:
    """Learn domain -> outlet type from the corpus metadata (e.g. healthbuzz.example.net -> tabloid)."""
    for d in docs:
        host = urlparse(d.get("url", "")).netloc.lower()
        if host and host not in KNOWN_DOMAINS:
            KNOWN_DOMAINS[host] = d.get("outlet_type", "unknown")


def _age_years(d: str) -> float:
    ref = date.fromisoformat(config.REFERENCE_DATE)
    try:
        dd = date.fromisoformat(d[:10])
    except Exception:
        return 5.0
    return max(0.0, (ref - dd).days / 365.25)


def recency_score(doc: dict) -> float:
    hl = 4.0 if doc.get("source") in ("news", "web") else 10.0
    return 0.5 + 0.5 * 2 ** (-_age_years(doc.get("date", "2015-01-01")) / hl)


def credibility(doc: dict) -> dict:
    otype = doc.get("outlet_type", "unknown")
    auth = AUTHORITY.get(otype, 0.3)
    evid = EVIDENCE.get(doc.get("study_type", ""), 0.4)
    rec = recency_score(doc)
    score = W_AUTH * auth + W_EVID * evid + W_REC * rec
    return {"score": round(score, 3), "authority": auth, "evidence_quality": evid, "recency": round(rec, 3),
            "outlet_type": otype, "study_type": doc.get("study_type", "")}


def credibility_from_url(url: str, docs_by_url: dict | None = None) -> dict:
    if docs_by_url and url in docs_by_url:
        return {**credibility(docs_by_url[url]), "matched": "known document"}
    host = urlparse(url).netloc.lower()
    otype = KNOWN_DOMAINS.get(host, "unknown")
    auth = AUTHORITY[otype]
    return {"score": round(W_AUTH * auth + W_EVID * 0.4 + W_REC * 0.75, 3), "authority": auth, "evidence_quality": 0.4,
            "recency": 0.75, "outlet_type": otype, "matched": "domain registry"}
