"""Evaluation utilities: LLM-as-a-Judge rubrics, correctness criterion, RAG Triad,
Cohen's kappa, Expected Calibration Error and simple retrieval metrics.

All judges use the `judge` role (a larger model *different* from the generator) at
temperature 0, return strict JSON and are cached, so scores are reproducible.
"""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

from .llm import LLM, item_value, list_field, truthy

# ------------------------------------------------------------------ helpers


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def gold_match(answer: str, golds: list[str]) -> bool:
    a = normalize_answer(answer)
    return any(normalize_answer(g) in a for g in golds if g)


def format_evidence(items: list[dict], max_words: int = 120, label: str = "S") -> str:
    out = []
    for i, it in enumerate(items, 1):
        words = it["text"].split()
        txt = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
        out.append(f"[{label}{i}] ({it.get('source', '?')}, {it.get('venue', '')}, {it.get('year', '')}) "
                   f"{it.get('title', '')}\n{txt}")
    return "\n\n".join(out)


def final_answer_section(text: str) -> str:
    m = re.search(r"final answer\s*[:\-]\s*(.*)", text, flags=re.I | re.S)
    return m.group(1).strip() if m else text.strip()


# --------------------------------------------------------- correctness judge
CORRECTNESS_SYSTEM = (
    "You are a strict, impartial grader of research answers. You compare a candidate answer with reference key "
    "facts. A fact is 'covered' only if the answer states it (paraphrase is fine; vaguer or partial statements are "
    "not). Respond with JSON only."
)


def judge_correctness(llm: LLM, q: dict, answer: str, purpose: str = "judge_correctness") -> dict:
    """Correctness criterion (fixed BEFORE running any technique, identical for all):

    * simple (NQ) questions: correct  <=>  the gold short answer appears in / is equivalent to the answer
      AND the answer does not contradict it.
    * medium / complex: correct  <=>  key-fact coverage >= 60 % AND no contradiction of a reference fact.
    * ambiguous: as above AND the answer explicitly acknowledges the conflicting evidence / multiple views.
    """
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(q["fact_texts"], 1))
    gold = f"\nGold short answer(s): {q['gold_answers']}" if q.get("gold_answers") else ""
    prompt = (
        f"Question: {q['question']}{gold}\n\nReference key facts:\n{facts}\n\nCandidate answer:\n\"\"\"{answer}\"\"\"\n\n"
        "Return JSON: {\"covered\": [list of booleans, one per key fact, in order], "
        "\"contradiction\": true if the answer asserts something that contradicts a reference fact, "
        "\"acknowledges_uncertainty\": true if the answer explicitly notes conflicting evidence, uncertainty or "
        "multiple valid interpretations, \"gold_match\": true if the answer contains the gold short answer or an "
        "equivalent (false if no gold given), \"rationale\": one sentence}"
    )
    data, resp = llm.chat_json(prompt, default={}, role="judge", system=CORRECTNESS_SYSTEM, max_tokens=300,
                               purpose=purpose)
    covered = [truthy(c) for c in list_field(data, "covered")][: len(q["fact_texts"])]
    covered += [False] * (len(q["fact_texts"]) - len(covered))
    coverage = float(np.mean(covered)) if covered else 0.0
    contradiction = truthy(data.get("contradiction", False))
    ack = truthy(data.get("acknowledges_uncertainty", False))
    if q.get("gold_answers"):
        gm = gold_match(answer, q["gold_answers"]) or truthy(data.get("gold_match", False))
        correct = gm and not contradiction
        coverage = 1.0 if gm else 0.0
    elif q["category"] == "ambiguous":
        correct = coverage >= 0.6 and not contradiction and ack
    else:
        correct = coverage >= 0.6 and not contradiction
    return {"correct": bool(correct), "coverage": coverage, "covered": covered, "contradiction": contradiction,
            "acknowledges_uncertainty": ack, "rationale": data.get("rationale", ""), "judge_model": resp.model}


# ---------------------------------------------------- synthesis quality judge
QUALITY_RUBRIC = """Score each criterion from 1 to 5 using these anchors:
COMPLETENESS - 1: misses almost all key facts; 3: covers about half of the key facts; 5: covers essentially all key facts.
ACCURACY - 1: several statements contradict the reference/evidence; 3: mostly accurate with one notable error or overstatement; 5: no errors or overstatements.
BALANCE - 1: one-sided, ignores conflicting evidence or presents weak sources as settled; 3: mentions other views superficially; 5: fairly weighs conflicting evidence by credibility and states uncertainty where it exists (for purely factual questions, 5 = no unwarranted hedging or bias).
CITATION_QUALITY - 1: no citations or citations to irrelevant sources; 3: some claims cited, some unsupported or mis-cited; 5: every substantive claim cites a source that supports it."""


def judge_synthesis_quality(llm: LLM, q: dict, answer: str, evidence: list[dict] | None = None,
                            purpose: str = "judge_quality") -> dict:
    facts = "\n".join(f"- {f}" for f in q["fact_texts"])
    conflicts = "\n".join(f"- {c['issue']}" for c in q.get("conflicts", [])) or "- none"
    ev = format_evidence(evidence, max_words=40) if evidence else "(not provided)"
    prompt = (
        f"{QUALITY_RUBRIC}\n\nQuestion: {q['question']}\n\nReference key facts:\n{facts}\n\nKnown points of "
        f"disagreement in the literature:\n{conflicts}\n\nSources the answer could cite (abbreviated):\n{ev}\n\n"
        f"Answer to grade:\n\"\"\"{answer}\"\"\"\n\nReturn JSON: {{\"completeness\": int, \"accuracy\": int, "
        f"\"balance\": int, \"citation_quality\": int, \"rationale\": \"one sentence\"}}"
    )
    data, resp = llm.chat_json(prompt, default={}, role="judge", max_tokens=250, purpose=purpose,
                               system="You are an expert research-quality assessor. Respond with JSON only.")
    scores = {}
    for k in ("completeness", "accuracy", "balance", "citation_quality"):
        try:
            scores[k] = int(np.clip(int(data.get(k, 1)), 1, 5))
        except Exception:
            scores[k] = 1
    scores["overall"] = float(np.mean(list(scores.values())))
    scores["rationale"] = data.get("rationale", "")
    return scores


# ------------------------------------------------------------------ RAG Triad
TRIAD_RUBRIC = """You evaluate a retrieval-augmented answer on three independent criteria, each on an anchored 1-5 scale.

CONTEXT RELEVANCE (is the retrieved evidence relevant to the question?)
 1 = nearly all retrieved passages are off-topic.
 3 = about half the passages are relevant; important aspects of the question have no supporting passage.
 5 = nearly all passages are relevant and together they cover every aspect of the question.
GROUNDEDNESS (is every claim in the answer supported by the retrieved context?)
 1 = most claims are not found in the context or contradict it.
 2 = several core claims are unsupported.
 3 = core claims are supported but some details/numbers are not in the context.
 4 = exactly one minor detail (a number, date, name or qualifier) is not in the context.
 5 = every factual claim can be traced to the context. Background knowledge that is true but absent from the
     context still counts as unsupported.
ANSWER RELEVANCE (does the answer address the question that was asked?)
 1 = answers a different question or refuses.
 3 = partially answers; a main part of the question is ignored.
 5 = directly and completely addresses every part of the question.

WORKED EXAMPLES (question: "How do vaccines work and why are boosters needed?")
 Level 1 example - context: passages about the history of smallpox and vaccine prices (context relevance 1); answer: "Vaccines were invented by Jenner and cost $20; boosters prevent autism" (groundedness 1 - unsupported/false; answer relevance 1 - never explains mechanism or boosters).
 Level 3 example - context: one passage on immune memory, one on vaccine logistics, one off-topic (context relevance 3); answer explains antibodies and memory cells correctly but says "boosters are needed because vaccines expire in the body after exactly one year" (groundedness 3 - invented detail); it skips why immunity wanes (answer relevance 3).
 Level 5 example - context: passages on immune memory, antibody waning and boosters (context relevance 5); answer: "Vaccines expose the immune system to an antigen, producing antibodies and memory cells [S1]; antibody levels wane over months and variants evolve, so boosters restore protection [S2][S3]" - every claim supported (groundedness 5), both parts answered (answer relevance 5)."""


def rag_triad(llm: LLM, question: str, contexts: list[dict], answer: str, purpose: str = "rag_triad") -> dict:
    ctx = format_evidence(contexts, max_words=90)
    prompt = (f"{TRIAD_RUBRIC}\n\nNow evaluate.\nQuestion: {question}\n\nRetrieved context:\n{ctx}\n\nAnswer:\n\"\"\"{answer}\"\"\"\n\n"
              "Procedure: (1) go through the answer claim by claim and list every claim, number or detail that is NOT "
              "stated in the retrieved context; (2) list the parts of the question with no relevant passage and the "
              "parts the answer does not address; (3) only then assign the three scores from the anchors above. "
              "Groundedness may be 5 only if the unsupported list is empty.\n"
              "Return JSON: {\"unsupported_claims\": [str], \"missing_aspects\": [str], \"context_relevance\": int, "
              "\"groundedness\": int, \"answer_relevance\": int, "
              "\"rationale\": {\"context_relevance\": str, \"groundedness\": str, \"answer_relevance\": str}}")
    data, resp = llm.chat_json(prompt, default={}, role="judge", max_tokens=900, purpose=purpose,
                               system="You are a strict, meticulous RAG evaluator. Respond with JSON only.")
    out = {}
    for k in ("context_relevance", "groundedness", "answer_relevance"):
        try:
            out[k] = int(np.clip(int(data.get(k, 1)), 1, 5))
        except Exception:
            out[k] = 1
    out["rationale"] = data.get("rationale", {})
    out["unsupported_claims"] = data.get("unsupported_claims", []) if isinstance(data.get("unsupported_claims"), list) else []
    return out


def triad_failure_stage(scores: dict, threshold: int = 4) -> str:
    """First broken stage along the pipeline order retrieval -> grounding -> answer. A stage fails below 4: for a
    research assistant a 3 already means invented details (groundedness) or an ignored part of the question."""
    if scores["context_relevance"] < threshold:
        return "retrieval"
    if scores["groundedness"] < threshold:
        return "grounding"
    if scores["answer_relevance"] < threshold:
        return "answer"
    return "none"


# ------------------------------------------------------- claim-level judges
def judge_claims_correct(llm: LLM, q: dict, claims: list[str], reference_docs: list[dict],
                         purpose: str = "judge_claims") -> list[bool]:
    """Ground-truth correctness of individual claims (used for calibration). Reference = key facts + the
    topic's credible documents. A claim is correct if it is true according to the reference."""
    if not claims:
        return []
    facts = "\n".join(f"- {f}" for f in q["fact_texts"])
    refs = format_evidence(reference_docs, max_words=70, label="R")
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    prompt = (f"Question: {q['question']}\n\nReference key facts:\n{facts}\n\nReference documents:\n{refs}\n\n"
              f"Claims to check:\n{numbered}\n\nFor each claim decide if it is CORRECT according to the reference "
              "(true and not overstated). Claims that state uncertainty accurately are correct. Claims that repeat a "
              "low-credibility source's assertion as fact are incorrect.\n"
              "Return JSON: {\"labels\": [true/false per claim in order]}")
    data, _ = llm.chat_json(prompt, default={}, role="judge", max_tokens=200, purpose=purpose,
                            system="You are a careful fact-checker. Respond with JSON only.")
    labels = [truthy(x) for x in list_field(data, "labels")][: len(claims)]
    return labels + [False] * (len(claims) - len(labels))


def judge_citation_support(llm: LLM, pairs: list[tuple[str, str]], purpose: str = "judge_citation") -> list[bool]:
    """pairs = [(claim, cited source text)]. Accurate only if the source DIRECTLY supports the claim."""
    if not pairs:
        return []
    blocks = "\n\n".join(f"Pair {i}:\nCLAIM: {c}\nSOURCE: {s[:700]}" for i, (c, s) in enumerate(pairs, 1))
    prompt = (f"For each pair, does the SOURCE directly support the CLAIM (not merely mention the topic)?\n\n{blocks}\n\n"
              "Return JSON: {\"supports\": [true/false per pair in order]}")
    data, _ = llm.chat_json(prompt, default={}, role="judge", max_tokens=150 + 10 * len(pairs), purpose=purpose,
                            system="You are a strict citation auditor. Respond with JSON only.")
    labels = [truthy(x) for x in list_field(data, "supports")][: len(pairs)]
    return labels + [False] * (len(pairs) - len(labels))


# ------------------------------------------------------------ statistics
def cohen_kappa(a: list, b: list, weights: str | None = None, labels: list | None = None) -> float:
    """Cohen's kappa implemented from scratch. weights in {None, 'linear', 'quadratic'}."""
    assert len(a) == len(b) and len(a) > 0
    labels = labels or sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n, k = len(a), len(labels)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= n
    E = np.outer(O.sum(1), O.sum(0))
    if weights is None:
        W = 1 - np.eye(k)
    else:
        d = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / max(k - 1, 1)
        W = d if weights == "linear" else d**2
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom > 0 else 1.0


def expected_calibration_error(conf: list[float], correct: list[bool], n_bins: int = 5):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        if m.sum() == 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "mean_conf": np.nan, "accuracy": np.nan, "gap": np.nan})
            continue
        mc, acc = conf[m].mean(), correct[m].mean()
        ece += m.sum() / len(conf) * abs(mc - acc)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_conf": mc, "accuracy": acc, "gap": mc - acc})
    return float(ece), rows


def recall_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    top = []
    for d in retrieved_doc_ids:
        if d not in top:
            top.append(d)
        if len(top) >= k:
            break
    return len(set(top) & relevant) / len(relevant)


def fact_coverage(retrieved_doc_ids, q: dict) -> float:
    """Evidence Coverage = fraction of the question's key facts for which at least one supporting document
    was retrieved (doc-level relevance labels from the curated corpus)."""
    got = set(retrieved_doc_ids)
    sup = q["fact_support"]
    return float(np.mean([bool(set(v) & got) for v in sup.values()])) if sup else float("nan")


def redundancy(chunk_ids: list[str]) -> float:
    """Retrieval Redundancy = share of retrieved chunks that duplicate an already-retrieved chunk."""
    if not chunk_ids:
        return 0.0
    return 1 - len(set(chunk_ids)) / len(chunk_ids)


def percentiles(values, ps=(50, 95)) -> dict:
    v = np.asarray(values, float)
    return {f"P{p}": float(np.percentile(v, p)) for p in ps} if len(v) else {f"P{p}": float("nan") for p in ps}


# --------------------------------------------------- reasoning depth rule
CONNECTIVES = ["because", "therefore", "thus", "hence", "so that", "as a result", "consequently", "which means",
               "this means", "since", "however", "but ", "whereas", "although", "in contrast", "on the other hand",
               "first", "second", "third", "finally", "next", "then", "overall", "in conclusion", "this suggests",
               "this indicates", "implies", "leads to", "due to", "while "]
_ENUM_RE = re.compile(r"^\s*(\d+[.)]|step\s*\d+|[-*•]|path\s*[a-c0-9])", re.I)


def reasoning_depth(text: str) -> int:
    """Consistent step-counting rule applied identically to every technique:
    split the delivered response into lines, and each line into sentences; a sentence of >= 5 words is a
    reasoning step if it (a) starts an enumerated line (1., 2), Step 3, bullet) or (b) contains an inferential /
    contrastive connective from a fixed lexicon."""
    steps = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        enum = bool(_ENUM_RE.match(line))
        sents = re.split(r"(?<=[.!?])\s+", line.strip())
        for j, s in enumerate(sents):
            if len(s.split()) < 5:
                continue
            low = " " + s.lower() + " "
            if (enum and j == 0) or any(c in low for c in CONNECTIVES):
                steps += 1
    return steps


def sources_referenced(text: str) -> int:
    return len(set(re.findall(r"\[S(\d+)\]", text)) | set(re.findall(r"\bS(\d+)\b", text)))
