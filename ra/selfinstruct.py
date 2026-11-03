"""Task 5 Part A — Self-Instruct pipeline (Wang et al., 2023) for research-assistant tasks, from scratch.

seed tasks -> instruction generation -> classification-task identification -> instance generation
(output-first for classification, input-first otherwise) -> filtering (ROUGE-L dedup, heuristics, LLM quality).
"""
from __future__ import annotations

import random
import re

import numpy as np

from . import config
from .llm import LLM

SEED_TASKS = [  # written manually for this assignment
    {"instruction": "Classify the type of evidence a study abstract provides.", "is_classification": True,
     "input": "We randomly assigned 400 adults to receive either a vitamin supplement or a placebo for two years and compared fracture rates.",
     "output": "Randomised controlled trial"},
    {"instruction": "Summarise the main finding of a research abstract in one sentence for a non-expert.", "is_classification": False,
     "input": "In a cohort of 20,000 nurses followed for 12 years, those sleeping fewer than 6 hours had a 15% higher risk of hypertension after adjustment for confounders.",
     "output": "Nurses who regularly slept less than six hours were somewhat more likely to develop high blood pressure over twelve years."},
    {"instruction": "Decide whether two source statements agree, conflict, or are unrelated.", "is_classification": True,
     "input": "Source A: Moderate coffee intake is linked to lower mortality. Source B: Coffee drinking raises the risk of early death.",
     "output": "Conflict"},
    {"instruction": "Break a complex research question into 2-4 searchable sub-questions.", "is_classification": False,
     "input": "How do microplastics affect marine food chains and human health?",
     "output": "1. How do microplastics enter marine food chains? 2. Do microplastics accumulate in fish and shellfish? 3. What evidence links dietary microplastics to human health effects?"},
    {"instruction": "Write an APA 7 reference for the given source details.", "is_classification": False,
     "input": "Authors: Lee, J. and Patel, R.; Year: 2021; Title: Sleep and memory consolidation; Journal: Journal of Sleep Science, 12(3), 45-60.",
     "output": "Lee, J., & Patel, R. (2021). Sleep and memory consolidation. Journal of Sleep Science, 12(3), 45-60."},
]

BANNED = ["image", "picture", "photo", "graph", "chart", "draw", "video", "audio", "file", "upload", "map of"]


# ---------------------------------------------------------- ROUGE-L
def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(a: str, b: str) -> float:
    ta, tb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    if not ta or not tb:
        return 0.0
    l = _lcs(ta, tb)
    if l == 0:
        return 0.0
    p, r = l / len(tb), l / len(ta)
    return 2 * p * r / (p + r)


# ----------------------------------------------------- generation steps
GEN_PROMPT = """Come up with {n} diverse task instructions for an AI RESEARCH ASSISTANT (tasks a researcher would delegate: evaluating evidence, summarising, comparing sources, classifying study designs or claims, extracting data, formulating search queries, critiquing methods, citation formatting, etc.).
Requirements: each instruction is one or two sentences, self-contained, text-only (no images/files), and different in verb and topic from the examples. Include both classification tasks (finite label set) and open-ended tasks.

Examples:
{examples}

Return JSON: {{"instructions": ["...", "..."]}}"""


def generate_instructions(llm: LLM, target: int = 50, per_call: int = 8, seed: int = config.SEED,
                          max_calls: int = 12) -> list[dict]:
    rng = random.Random(seed)
    pool: list[dict] = []
    calls = 0
    while len(pool) < target and calls < max_calls:
        demos = rng.sample(SEED_TASKS, 3) + (rng.sample(pool, min(2, len(pool))) if pool else [])
        ex = "\n".join(f"- {d['instruction']}" for d in demos)
        data, _ = llm.chat_json(GEN_PROMPT.format(n=per_call, examples=ex), default={}, temperature=0.9,
                                seed=seed + calls, max_tokens=700, purpose="t5_si_generate")
        calls += 1
        for ins in data.get("instructions") or []:
            if isinstance(ins, str) and ins.strip():
                pool.append({"instruction": ins.strip(), "origin": f"call{calls}"})
    return pool[:target]


def classify_instructions(llm: LLM, items: list[dict], batch: int = 10) -> None:
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        listing = "\n".join(f"{j+1}. {x['instruction']}" for j, x in enumerate(chunk))
        data, _ = llm.chat_json(
            "Is each task a CLASSIFICATION task (the output is one label from a small finite set, e.g. "
            "'agree/conflict/unrelated', 'RCT/cohort/...')? Answer Yes or No for each.\n\n" + listing +
            "\n\nReturn JSON: {\"labels\": [\"Yes\"|\"No\", ...]}", default={}, max_tokens=200, purpose="t5_si_classify")
        labels = data.get("labels") or []
        for j, x in enumerate(chunk):
            lab = str(labels[j]).strip().lower() if j < len(labels) else "no"
            x["is_classification"] = lab.startswith("y")


OUTPUT_FIRST = """Task: {instruction}
This is a classification task. OUTPUT-FIRST generation:
1) List the possible class labels for this task.
2) Take the label at position {pos} of your list (counting from 1; wrap around if the list is shorter) as the TARGET label.
3) Only then write a realistic input whose correct label is the TARGET label.
Return JSON: {{"labels": ["..."], "target_label": "...", "input": "..."}}"""

INPUT_FIRST = """Task: {instruction}
INPUT-FIRST generation: first write a realistic, specific input for this task (if the task needs no input, use ""), then write the ideal output for that input.
Return JSON: {{"input": "...", "output": "..."}}"""


def generate_instances(llm: LLM, items: list[dict]) -> list[dict]:
    out = []
    for i, x in enumerate(items):
        if x.get("is_classification"):
            data, _ = llm.chat_json(OUTPUT_FIRST.format(instruction=x["instruction"], pos=(i % 3) + 1), default={},
                                    temperature=0.7, max_tokens=350, purpose="t5_si_output_first")
            out.append({**x, "method": "output-first", "labels": data.get("labels", []),
                        "input": str(data.get("input", "")), "output": str(data.get("target_label", ""))})
        else:
            data, _ = llm.chat_json(INPUT_FIRST.format(instruction=x["instruction"]), default={}, temperature=0.7,
                                    max_tokens=450, purpose="t5_si_input_first")
            out.append({**x, "method": "input-first", "input": str(data.get("input", "")),
                        "output": str(data.get("output", ""))})
    return out


# ------------------------------------------------------------ filtering
def heuristic_issues(x: dict) -> list[str]:
    issues = []
    ins = x["instruction"]
    if not (3 <= len(ins.split()) <= 150):
        issues.append("instruction length")
    if any(b in ins.lower() for b in BANNED):
        issues.append("non-text modality")
    if not x.get("output") or len(x["output"].split()) < 1:
        issues.append("empty output")
    if x.get("input") and x.get("output") and x["input"].strip() == x["output"].strip():
        issues.append("output copies input")
    if x.get("is_classification") and x.get("labels") and x["output"] not in x["labels"]:
        issues.append("label not in label set")
    if not x.get("is_classification") and len(x.get("output", "").split()) < 5:
        issues.append("output too short")
    return issues


def llm_quality(llm: LLM, items: list[dict], batch: int = 8) -> None:
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        listing = "\n\n".join(f"#{j+1}\nINSTRUCTION: {x['instruction']}\nINPUT: {x.get('input','')[:300]}\nOUTPUT: {x.get('output','')[:300]}"
                              for j, x in enumerate(chunk))
        data, _ = llm.chat_json(
            "Rate each instruction-tuning example 1-5 (5 = clear, useful research task with a correct, high-quality "
            "output; 3 = usable but flawed; 1 = wrong, vague or useless).\n\n" + listing +
            "\n\nReturn JSON: {\"scores\": [int, ...]}", default={}, role="judge", max_tokens=150, purpose="t5_si_quality")
        sc = data.get("scores") or []
        for j, x in enumerate(chunk):
            try:
                x["quality"] = int(sc[j])
            except Exception:
                x["quality"] = 1


def filter_dataset(items: list[dict], rouge_threshold: float = 0.7, min_quality: int = 3) -> tuple[list[dict], dict]:
    kept, reasons = [], {"duplicate (ROUGE-L>=0.7)": 0, "heuristic": 0, "low LLM quality": 0}
    pool = [s["instruction"] for s in SEED_TASKS]
    for x in items:
        x["max_rouge"] = max(rouge_l(x["instruction"], p) for p in pool)
        if x["max_rouge"] >= rouge_threshold:
            reasons["duplicate (ROUGE-L>=0.7)"] += 1
            x["kept"] = False
            continue
        x["issues"] = heuristic_issues(x)
        if x["issues"]:
            reasons["heuristic"] += 1
            x["kept"] = False
            continue
        if x.get("quality", 5) < min_quality:
            reasons["low LLM quality"] += 1
            x["kept"] = False
            continue
        x["kept"] = True
        kept.append(x)
        pool.append(x["instruction"])
    return kept, reasons


def diversity(items: list[dict]) -> dict:
    verbs = [re.findall(r"[a-z]+", x["instruction"].lower())[:1] for x in items]
    verbs = [v[0] for v in verbs if v]
    rouges = [rouge_l(a["instruction"], b["instruction"]) for i, a in enumerate(items) for b in items[i + 1 :]]
    return {"unique_first_verbs": len(set(verbs)), "mean_pairwise_rougeL": float(np.mean(rouges)) if rouges else 0.0}
