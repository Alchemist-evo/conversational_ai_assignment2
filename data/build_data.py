"""Build every data file used by the notebook (run once; outputs are committed).

1. Natural Questions subset (real data, CC-BY-SA-3.0):
   `google-research-datasets/nq_open` (question + gold short answers) joined on the question text with
   `sentence-transformers/natural-questions` (question + gold Wikipedia passage). We keep pairs whose gold
   answer string occurs in the passage, filter to the Science/Health/Economy domain with a keyword list,
   and sample 1,000 examples with seed 42.  -> data/nq_subset.json
   MODIFICATION NOTICE: the dataset was subsetted, filtered and passages truncated to 300 words.
2. Real Wikipedia articles (CC-BY-SA) for every curated topic + long-context distractors, fetched via the
   MediaWiki API and truncated.  -> data/wikipedia_articles.json
3. Curated synthetic corpus (author_corpus.py) + automatic Wikipedia support labels for key facts.
4. 50 SYNTHETIC past research sessions for episodic memory (dev topics + NQ `episodic` split only).
   -> data/episodic_sessions.json

Usage:  python data/build_data.py
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "curated"))
SEED = 42

DOMAIN_KEYWORDS = [  # matched as whole words (prefix match for stems ending in '*')
    "disease*", "virus*", "viral", "vaccin*", "cell*", "blood", "heart", "cancer*", "brain", "protein*", "gene", "genes",
    "genetic*", "dna", "rna", "bacteri*", "infection*", "drug*", "medicine*", "medical", "health", "vitamin*",
    "hormone*", "photosynth*", "species", "atom*", "chemical*", "molecule*", "planet*", "climate", "ocean*",
    "carbon", "oxygen", "evolution*", "economy", "economic*", "inflation", "banking", "central bank", "gdp", "insulin", "diabetes",
    "muscle*", "bone*", "organ", "organs", "enzyme*", "nutrient*", "diet*", "immune", "antibiotic*", "physics",
    "electron*", "chromosome*", "metabol*", "nerve*", "kidney*", "liver", "lung*", "tissue*", "mitochondria*",
    "photon*", "radiation", "nuclear", "acid*", "plant cell*", "solar system", "gravity", "neuron*", "symptom*",
]

WB = chr(92) + "b"  # regex word boundary
WORD = chr(92) + "w*"  # regex: word characters
DOMAIN_RE = re.compile(
    WB + "(" + "|".join(re.escape(k[:-1]) + WORD if k.endswith("*") else re.escape(k) for k in DOMAIN_KEYWORDS) + ")" + WB
)

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "ResearchAssistantCourseProject/1.0 (educational use; python-urllib)"
DROP_SECTIONS = ("See also", "References", "External links", "Further reading", "Notes", "Bibliography", "Sources")


def build_nq(n_total: int = 1000) -> list[dict]:
    from datasets import load_dataset

    qa = load_dataset("google-research-datasets/nq_open", split="train")
    st = load_dataset("sentence-transformers/natural-questions", split="train")
    gold = {r["question"]: r["answer"] for r in qa}
    rows, seen = [], set()
    for r in st:
        q = r["query"]
        if q not in gold or q in seen:
            continue
        passage = r["answer"]
        answers = gold[q]
        if not any(a.lower() in passage.lower() for a in answers):
            continue
        if not DOMAIN_RE.search(q.lower()):
            continue
        seen.add(q)
        words = passage.split()
        rows.append({"question": q, "answers": answers, "passage": " ".join(words[:300])})
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = rows[:n_total]
    # Task-1 simple questions: first 5 rows that are unambiguously science/health/economics
    # (no pop-culture "who played/sang" questions) -> moved to the front.
    strict = ("brain", "blood", "cell", "dna", "protein", "economic", "gdp", "bone", "enzyme", "immune", "insulin",
              "virus", "bacteria", "photosynth", "vitamin", "heart rate", "kidney", "liver")
    pick = [i for i, r in enumerate(rows) if any(k in r["question"] for k in strict)
            and not r["question"].startswith(("who ", "where was", "when was"))][:5]
    rows = [rows[i] for i in pick] + [r for i, r in enumerate(rows) if i not in set(pick)]
    # split assignment (documented in the notebook)
    splits = ["prompt_simple"] * 5 + ["episodic"] * 30 + ["retrieval_dev"] * 100 + ["heldout"] * 50
    splits += ["distractor"] * (len(rows) - len(splits))
    for i, (r, s) in enumerate(zip(rows, splits)):
        r["nq_id"] = f"NQ-{i:04d}"
        r["split"] = s
        r["doc_id"] = f"WIKI-NQ-{i:04d}"
        r["answer_present"] = True
    print(f"NQ: {len(rows)} examples after domain filter + sampling")
    return rows


def fetch_wiki(title: str, max_words: int = 4000) -> dict | None:
    params = {"action": "query", "format": "json", "prop": "extracts|info", "explaintext": 1, "redirects": 1,
              "inprop": "url", "titles": title}
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
            break
        except Exception as e:  # noqa: BLE001
            print("  retry", title, e)
            time.sleep(5 * (attempt + 1))
    else:
        return None
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    if not text:
        return None
    # cut trailing reference sections
    for sec in DROP_SECTIONS:
        m = re.search(rf"\n==+ {re.escape(sec)} ==+\n", text)
        if m:
            text = text[: m.start()]
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    words = text.split(" ")
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return {"title": page["title"], "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title}"), "text": text}


def build_wikipedia(topics: list[dict], extra: list[str]) -> dict:
    path = HERE / "wikipedia_articles.json"
    out = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    have = {d["title"] for d in out.values()}
    wanted = [(t["topic_id"], t["wiki_title"]) for t in topics] + [("EXTRA", x) for x in extra]
    for tid, title in wanted:
        if title in have or title.replace("–", "-") in have:
            continue
        art = fetch_wiki(title)
        if not art:
            print("  !! could not fetch", title)
            continue
        doc_id = "WIKI-" + re.sub(r"[^A-Za-z0-9]+", "_", art["title"]).strip("_")
        out[doc_id] = {"doc_id": doc_id, "topic_id": tid if tid != "EXTRA" else None, "source": "wikipedia",
                       "title": art["title"], "authors": ["Wikipedia contributors"], "year": 2026,
                       "date": "2026-11-01", "venue": "Wikipedia, The Free Encyclopedia", "url": art["url"],
                       "study_type": "encyclopedia", "peer_reviewed": False, "outlet_type": "encyclopedia",
                       "access": "open", "synthetic": False, "text": art["text"]}
        print(f"  wiki {art['title']}: {len(art['text'].split())} words")
        time.sleep(1.5)
    return out


def augment_support(topics: list[dict], wiki: dict) -> int:
    """Add a topic's Wikipedia article to a fact's support set if it contains all anchor keywords."""
    added = 0
    by_topic = {d["topic_id"]: d for d in wiki.values() if d["topic_id"]}
    for t in topics:
        art = by_topic.get(t["topic_id"])
        if not art:
            continue
        low = art["text"].lower()
        for f in t["facts"].values():
            kws = f.get("wiki_keywords") or []
            if kws and all(k.lower() in low for k in kws):
                if art["doc_id"] not in f["support"]:
                    f["support"].append(art["doc_id"])
                    added += 1
    return added


PREFERENCES = [
    "Prefers concise bullet-point summaries with APA citations.",
    "Wants both sides of contested evidence presented explicitly.",
    "Asks for peer-reviewed sources over news where possible.",
    "Prefers plain-language explanations for a non-expert audience.",
    "Wants numbers and effect sizes quoted precisely.",
]


def build_episodic(topics: list[dict], nq: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    sessions = []
    # 20 sessions from DEV topics only (held-out topics are never stored in memory)
    dev_topics = [t for t in topics if t["split"] == "dev"]
    templates = [
        "Background reading on {name}", "Key findings about {name}", "Controversies around {name}",
        "Recent evidence on {name}", "Explain {name} for a class presentation",
    ]
    i = 0
    while len(sessions) < 20:
        t = dev_topics[i % len(dev_topics)]
        tmpl = templates[(i // len(dev_topics)) % len(templates)]
        facts = list(t["facts"].values())
        rng.shuffle(facts)
        chosen = facts[:2]
        docs = sorted({d for f in chosen for d in f["support"] if not d.startswith("WIKI")})
        sessions.append({
            "session_id": f"EP-{len(sessions)+1:03d}", "synthetic": True, "record_type": "SYNTHETIC research session",
            "date": f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}", "topic_id": t["topic_id"],
            "user_query": tmpl.format(name=t["name"].lower()),
            "sub_queries": [f"What is known about {t['name'].lower()}?", f"What evidence supports: {chosen[0]['text'][:60]}?"],
            "sources_consulted": docs,
            "key_findings": [f["text"] for f in chosen],
            "user_preference": PREFERENCES[i % len(PREFERENCES)],
            "outcome": rng.choice(["answered", "answered", "answered", "partially answered"]),
            "user_rating": rng.choice([3, 4, 4, 5, 5]),
        })
        i += 1
    # 30 sessions from NQ `episodic` split
    for r in [x for x in nq if x["split"] == "episodic"]:
        sent = next((s for s in re.split(r"(?<=[.!?])\s+", r["passage"]) if any(a.lower() in s.lower() for a in r["answers"])),
                    r["passage"][:200])
        sessions.append({
            "session_id": f"EP-{len(sessions)+1:03d}", "synthetic": True, "record_type": "SYNTHETIC research session",
            "date": f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}", "topic_id": None,
            "user_query": r["question"].capitalize() + "?",
            "sub_queries": [r["question"]],
            "sources_consulted": [r["doc_id"]],
            "key_findings": [f"Answer: {r['answers'][0]}. Evidence: {sent[:300]}"],
            "user_preference": PREFERENCES[len(sessions) % len(PREFERENCES)],
            "outcome": "answered", "user_rating": rng.choice([3, 4, 5]),
        })
    assert len(sessions) == 50
    return sessions


def main() -> None:
    import author_corpus

    author_corpus.main()
    corpus = json.load(open(HERE / "curated" / "topics.json", encoding="utf-8"))
    topics = corpus["topics"]

    nq_path = HERE / "nq_subset.json"
    nq = json.load(open(nq_path, encoding="utf-8")) if nq_path.exists() else build_nq()
    json.dump(nq, open(HERE / "nq_subset.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    wiki = build_wikipedia(topics, corpus["long_context_distractors"] + corpus["extra_wiki"])
    json.dump(wiki, open(HERE / "wikipedia_articles.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    n = augment_support(topics, wiki)
    print(f"added {n} automatic Wikipedia support labels")
    json.dump(corpus, open(HERE / "curated" / "topics.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    sessions = build_episodic(topics, nq)
    json.dump(sessions, open(HERE / "episodic_sessions.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {len(sessions)} synthetic episodic sessions")


if __name__ == "__main__":
    main()
