"""Loading the datasets, building the knowledge collections and the documented splits."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import cached_property

from . import config
from .chunking import Chunk, chunk_document


@dataclass
class ResearchData:
    topics: dict
    docs: dict
    nq: list
    sessions: list
    long_context_distractors: list = field(default_factory=list)

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, data_dir=None) -> "ResearchData":
        d = data_dir or config.DATA_DIR
        corpus = json.load(open(d / "curated" / "topics.json", encoding="utf-8"))
        wiki = json.load(open(d / "wikipedia_articles.json", encoding="utf-8"))
        nq = json.load(open(d / "nq_subset.json", encoding="utf-8"))
        sessions = json.load(open(d / "episodic_sessions.json", encoding="utf-8"))
        docs = {x["doc_id"]: x for x in corpus["documents"]}
        docs.update(wiki)
        for r in nq:
            first = " ".join(r["passage"].split()[:6])
            docs[r["doc_id"]] = {
                "doc_id": r["doc_id"], "topic_id": None, "source": "wikipedia", "title": f"Wikipedia passage: {first}...",
                "authors": ["Wikipedia contributors"], "year": 2018, "date": "2018-12-01",
                "venue": "Wikipedia (via Natural Questions)", "url": "https://huggingface.co/datasets/sentence-transformers/natural-questions",
                "study_type": "encyclopedia", "peer_reviewed": False, "outlet_type": "encyclopedia", "access": "open",
                "synthetic": False, "text": r["passage"], "nq_id": r["nq_id"],
            }
        topics = {t["topic_id"]: t for t in corpus["topics"]}
        wiki_titles = {v["title"]: k for k, v in wiki.items()}
        distractors = [wiki_titles[t] for t in corpus["long_context_distractors"] if t in wiki_titles]
        return cls(topics, docs, nq, sessions, distractors)

    # ----------------------------------------------------------- questions
    @cached_property
    def questions(self) -> list[dict]:
        out = []
        for t in self.topics.values():
            for q in t["questions"]:
                qq = dict(q)
                qq["topic_id"] = t["topic_id"]
                qq["split"] = t["split"]
                qq["fact_texts"] = [t["facts"][f]["text"] for f in q["facts"]]
                qq["fact_support"] = {f: list(t["facts"][f]["support"]) for f in q["facts"]}
                qq["conflicts"] = t["conflicts"]
                qq["gold_answers"] = None
                out.append(qq)
        return out

    def by_use(self, use: str) -> list[dict]:
        return [q for q in self.questions if use in q["use"]]

    def nq_split(self, split: str) -> list[dict]:
        return [r for r in self.nq if r["split"] == split]

    def nq_as_question(self, r: dict, category: str = "simple") -> dict:
        return {"qid": r["nq_id"], "question": r["question"][0].upper() + r["question"][1:] + "?", "category": category,
                "topic_id": None, "split": r["split"], "facts": ["F1"],
                "fact_texts": [f"The answer is: {r['answers'][0]}"], "fact_support": {"F1": [r["doc_id"]]},
                "conflicts": [], "gold_answers": r["answers"], "source_profile": "single", "nature": "factual",
                "use": [], "gold_doc": r["doc_id"]}

    def task1_questions(self) -> list[dict]:
        simple = [self.nq_as_question(r) for r in self.nq_split("prompt_simple")]
        cur = self.by_use("task1")
        order = {"medium": 1, "complex": 2, "ambiguous": 3}
        return simple + sorted(cur, key=lambda q: (order[q["category"]], q["qid"]))

    def relevant_docs(self, q: dict) -> set[str]:
        return {d for ds in q["fact_support"].values() for d in ds}

    def topic_docs(self, topic_id: str, sources=("academic", "news", "web")) -> list[dict]:
        return [d for d in self.docs.values() if d.get("topic_id") == topic_id and d["source"] in sources]

    # ---------------------------------------------------------- collections
    def collection_docs(self) -> dict[str, list[dict]]:
        """Knowledge collections (sources). `web` is the general-web fallback index."""
        cols = {"academic": [], "wikipedia": [], "news": [], "web": []}
        for d in self.docs.values():
            if d["source"] == "academic":
                cols["academic"].append(d)
            elif d["source"] == "wikipedia":
                cols["wikipedia"].append(d)
            elif d["source"] == "news":
                cols["news"].append(d)
            elif d["source"] == "web":
                cols["web"].append(d)
        # general web crawl = user-generated posts + news + NQ wiki passages (lower-trust mirror)
        cols["web"] = cols["web"] + cols["news"] + [d for d in cols["wikipedia"] if d.get("nq_id")]
        return cols

    def build_chunks(self, strategy: str = "recursive", size: int = config.CHUNK_SIZE_DEFAULT,
                     overlap: float = config.CHUNK_OVERLAP_DEFAULT, embedder=None, sources=None) -> dict[str, list[Chunk]]:
        out = {}
        for s, ds in self.collection_docs().items():
            if sources and s not in sources:
                continue
            chunks = []
            for d in ds:
                # short synthetic docs and NQ passages are already passage-sized
                strat = strategy if len(d["text"].split()) > 120 else "none"
                for c in chunk_document(d, strat, size, overlap, embedder):
                    if s == "web":
                        c = Chunk("web:" + c.chunk_id, c.doc_id, "web", c.text, {**c.meta, "mirrored_from": d["source"]})
                    chunks.append(c)
            out[s] = chunks
        return out

    # ------------------------------------------------------- evidence packs
    def evidence_pack(self, q: dict, n: int = 6, seed: int = config.SEED) -> list[dict]:
        """Fixed evidence bundle for Task 1 so that every prompting technique sees identical input.

        Curated questions: all topic docs (academic/news/web) + the topic's Wikipedia lead (first 250 words).
        NQ questions: the gold passage + 3 random distractor passages."""
        rng = random.Random(f"{seed}-{q['qid']}")
        if q.get("gold_doc"):
            others = [r["doc_id"] for r in self.nq_split("distractor")]
            ids = [q["gold_doc"]] + rng.sample(others, 3)
            rng.shuffle(ids)
            return [self._pack_item(self.docs[i]) for i in ids]
        docs = self.topic_docs(q["topic_id"])
        wiki = [d for d in self.docs.values() if d["source"] == "wikipedia" and d.get("topic_id") == q["topic_id"]]
        items = [self._pack_item(d) for d in docs] + [self._pack_item(w, 250) for w in wiki]
        rng.shuffle(items)
        return items[:n] if len(items) > n else items

    @staticmethod
    def _pack_item(d: dict, max_words: int = 160) -> dict:
        words = d["text"].split()
        return {"doc_id": d["doc_id"], "source": d["source"], "title": d["title"], "venue": d["venue"],
                "year": d["year"], "text": " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")}

    def split_table(self):
        import pandas as pd

        rows = []
        for s in ["prompt_simple", "episodic", "retrieval_dev", "heldout", "distractor"]:
            use = {"prompt_simple": "Task 1 simple questions", "episodic": "Seeds for 30 synthetic episodic sessions",
                   "retrieval_dev": "Task 3 chunking / Recall@k experiments", "heldout": "Task 6 final 50-question evaluation",
                   "distractor": "Wikipedia collection noise (never queried directly)"}[s]
            rows.append({"dataset": "Natural Questions (subset)", "split": s, "size": len(self.nq_split(s)), "used_for": use})
        dev = [t for t in self.topics.values() if t["split"] == "dev"]
        ho = [t for t in self.topics.values() if t["split"] == "heldout"]
        rows.append({"dataset": "Curated topics", "split": "dev", "size": f"{len(dev)} topics / {sum(len(t['questions']) for t in dev)} questions",
                     "used_for": "Task 1 medium/complex/ambiguous + APE dev; Task 3 experiments; Task 4 weight tuning; 20 episodic sessions"})
        rows.append({"dataset": "Curated topics", "split": "heldout", "size": f"{len(ho)} topics / {sum(len(t['questions']) for t in ho)} questions",
                     "used_for": "Task 4 synthesis + RAG Triad; Task 6 10-query evaluation (never in memory)"})
        return pd.DataFrame(rows)
