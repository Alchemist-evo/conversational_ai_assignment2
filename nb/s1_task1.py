from common import code, md, module

CELLS = [
md(r"""
---
# Module 1 · Task 1 — Multi-Step Reasoning Strategy Evaluation (1 mark)

**Objective.** Find the prompting strategy that reasons most reliably about research questions of different complexity.

### 1.1 Design
**Research domain: Science & Health, with Economics.** These are areas where the evidence base is large, where credible and non-credible sources visibly disagree (nutrition, vaccines, sweeteners), and where questions span single facts up to contested multi-source syntheses. It also covers the sample questions given in the assignment.

**Test set: 20 questions, 5 per complexity level.**
| Level | Definition | Source |
|---|---|---|
| Simple | Single-source answerable, one gold short answer | 5 real NQ questions (split `prompt_simple`) |
| Medium | Needs synthesis across several paragraphs | Curated dev topics |
| Complex | Needs synthesis across *multiple source types* | Curated dev topics |
| Ambiguous | Conflicting evidence or several valid interpretations | Curated dev topics with gold conflicts |

**Controlled comparison.** Every technique gets the *same* question and the *same evidence pack*: the topic's academic/news/forum documents plus the Wikipedia lead, in the same order; for NQ, the gold passage plus 3 random distractors. The same system prompt is used, generation is at temperature 0 (except ToT exploration), and the same generator model runs all of it. Any difference therefore comes from the prompting strategy alone.

**The five techniques**
* **Zero-Shot**: a direct answer with citations.
* **Few-Shot**: 3 hand-written, well-reasoned exemplars (vitamin C & colds, why the sky is blue, nuclear vs. coal safety). All lie outside the test topics, and each models evidence weighting and source criticism.
* **Chain-of-Thought (CoT)**: a structured 5-step scaffold. (1) restate, (2) extract evidence per source, (3) find agreement/conflict, (4) weigh credibility, (5) conclude with uncertainty.
* **Tree-of-Thought (ToT)**: 3 reasoning paths with different starting strategies (credibility-first, mechanism-first, conflict-first) at T = 0.7. An evaluator call scores each path's **evidence strength** (1–10), and the best path is returned. That is 4 LLM calls per question.
* **APE for research**: the LLM is shown expert demonstrations and **proposes 5 candidate instructions**. Each candidate is scored on a **separate APE-dev set** (4 questions never used in the test set) with the same correctness judge. The best instruction is then used on the 20 test questions.

### 1.2 Metrics (criteria fixed *before* running any technique and applied identically to all five)
* **Reasoning correctness.** An LLM judge (gpt-oss-120b on Groq, a different and larger model family than the Gemini generator) marks each reference key fact as covered or not, and flags contradictions and acknowledged uncertainty. A response is **correct** if:
  * *simple:* it contains the gold short answer (normalised string match, or judged equivalent) and does not contradict it;
  * *medium/complex:* it covers ≥ 60 % of key facts with no contradiction;
  * *ambiguous:* as for medium/complex, **and** it explicitly acknowledges the conflicting evidence or multiple interpretations.
  The judge sees the full delivered response (for ToT, the selected path).
* **Reasoning depth.** A fixed lexical rule (`ra.evaluation.reasoning_depth`). A sentence of at least 5 words counts as one reasoning step if it starts an enumerated line (`1.`, `Step 2`, a bullet) **or** contains an inferential/contrastive connective from a fixed lexicon (*because, therefore, however, which means, …*). The same rule is applied to every technique's delivered response.
* **Sources referenced**: the number of distinct `[S#]` sources cited.
* **Tokens & latency**: exact API usage and wall-clock latency, summed over all calls a technique makes per query (ToT = 4 calls).
* **Cost per 1,000 queries**: tokens × the assignment's GPT-4o pricing ($5 / 1M input, $15 / 1M output).
"""),
module("prompting.py"),
code(r"""
from ra.prompting import (TECHNIQUES, ape_generate_candidates, ape_select, with_instruction, ZERO_SHOT_INSTR, COT_INSTR,
                          FEW_SHOT_EXAMPLES, TOT_PATHS)
from ra.evaluation import judge_correctness, reasoning_depth, sources_referenced
T1Q = D.task1_questions()
display(Markdown("**Table 1.1 – Task 1 test set (20 questions).**"))
display(pd.DataFrame([{"qid": q["qid"], "level": q["category"], "question": q["question"], "#key facts": len(q["fact_texts"]),
                       "gold answer": (q["gold_answers"] or [""])[0]} for q in T1Q]))
assert pd.Series([q["category"] for q in T1Q]).value_counts().to_dict() == {"simple": 5, "medium": 5, "complex": 5, "ambiguous": 5}
"""),
md(r"""
### 1.3 APE: automatic prompt generation and selection
APE treats the instruction as a parameter to optimise. The LLM writes candidate instructions from demonstrations of ideal behaviour. Each candidate is scored on held-aside APE-dev questions; the human-written Zero-Shot and CoT instructions are scored as baselines, and the top-scoring LLM candidate is kept.
"""),
code(r"""
ape_dev = D.by_use("ape_dev")
judge_fn = lambda q, ans: judge_correctness(llm, q, ans, purpose="t1_judge_ape")
candidates = ape_generate_candidates(llm, n=5)
all_cands = candidates + [ZERO_SHOT_INSTR, COT_INSTR]
best_instr, ape_rows = ape_select(llm, all_cands, ape_dev, D.evidence_pack, judge_fn)
ape_df = pd.DataFrame(ape_rows)
ape_df["origin"] = ["LLM candidate"] * len(candidates) + ["human: zero-shot", "human: CoT"]
llm_rows = ape_df[ape_df.origin == "LLM candidate"].sort_values(["dev_correct", "dev_coverage"], ascending=False)
APE_BEST = llm_rows.iloc[0]["instruction"] if len(llm_rows) else COT_INSTR
display(Markdown(f"**Table 1.2 – APE candidates scored on the APE-dev set ({len(ape_dev)} questions).**"))
display(ape_df[["origin", "dev_correct", "dev_coverage", "instruction"]])
print("\nSelected APE instruction:\n", textwrap.fill(APE_BEST, 120))
"""),
md(r"""
### 1.4 Running the five techniques on the 20 test questions
"""),
code(r"""
t1_rows, t1_outputs = [], {}
for qi, q in enumerate(T1Q, 1):
    ev = D.evidence_pack(q)
    outs = {name: fn(llm, q["question"], ev) for name, fn in TECHNIQUES.items()}
    outs["APE"] = with_instruction(llm, q["question"], ev, APE_BEST)
    t1_outputs[q["qid"]] = outs
    for name, o in outs.items():
        j = judge_correctness(llm, q, o.response, purpose="t1_judge")
        t1_rows.append({"qid": q["qid"], "level": q["category"], "technique": name, "correct": j["correct"],
                        "coverage": j["coverage"], "contradiction": j["contradiction"], "ack_uncertainty": j["acknowledges_uncertainty"],
                        "depth": reasoning_depth(o.response), "sources_ref": sources_referenced(o.response),
                        "in_tok": o.input_tokens, "out_tok": o.output_tokens, "latency_s": o.latency_s, "calls": o.n_calls})
    print(f"\r{qi}/{len(T1Q)} questions done", end="")
t1 = pd.DataFrame(t1_rows)
ORDER = ["Zero-Shot", "Few-Shot", "CoT", "ToT", "APE"]
summ = t1.groupby("technique").agg(correctness=("correct", "mean"), key_fact_coverage=("coverage", "mean"), depth=("depth", "mean"),
                                   sources_ref=("sources_ref", "mean"), in_tok=("in_tok", "mean"), out_tok=("out_tok", "mean"),
                                   latency_s=("latency_s", "mean"), calls=("calls", "mean")).reindex(ORDER)
summ["correctness"] *= 100
summ["cost_per_1k_usd"] = [config.cost_usd(r.in_tok, r.out_tok) * 1000 for r in summ.itertuples()]
by_level = (t1.pivot_table(index="technique", columns="level", values="correct", aggfunc="mean") * 100).reindex(ORDER)[["simple", "medium", "complex", "ambiguous"]]
print()
display(Markdown("**Table 1.3 – Technique comparison (means per query).**")); display(summ.round(2))
display(Markdown("**Table 1.4 – Reasoning correctness (%) by question level.**")); display(by_level.round(0))
RESULTS["task1"] = {"summary": summ.round(3).to_dict(), "by_level": by_level.round(1).to_dict(), "ape_instruction": APE_BEST}
"""),
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axes[0]
ax.bar(summ.index, summ["correctness"], color=PALETTE[:5]); bar_labels(ax, "{:.0f}%")
ax.set_ylabel("Reasoning correctness (%)"); ax.set_title("Fig 1.1 – Reasoning correctness vs technique"); ax.set_ylim(0, 105)
ax = axes[1]
for i, tname in enumerate(ORDER):
    s = t1[t1.technique == tname]
    ax.scatter(s.latency_s, s.depth, alpha=0.35, color=PALETTE[i], s=18)
    ax.scatter(s.latency_s.mean(), s.depth.mean(), color=PALETTE[i], s=160, edgecolor="k", label=tname, marker="D")
ax.set_xlabel("Latency per query (s)"); ax.set_ylabel("Reasoning depth (steps)"); ax.set_title("Fig 1.2 – Reasoning depth vs latency"); ax.legend(fontsize=8)
ax = axes[2]
for i, tname in enumerate(ORDER):
    s = t1[t1.technique == tname].groupby("sources_ref")["depth"].mean()
    ax.plot(s.index, s.values, marker="o", color=PALETTE[i], label=tname)
ax.set_xlabel("Number of sources referenced"); ax.set_ylabel("Mean reasoning depth"); ax.set_title("Fig 1.3 – Depth vs number of sources referenced"); ax.legend(fontsize=8)
plt.tight_layout(); save(fig, "fig1_task1"); plt.show()
fig, ax = plt.subplots(figsize=(9, 3.8))
by_level.plot(kind="bar", ax=ax, color=PALETTE[:4], rot=0); ax.set_ylabel("Correct (%)"); ax.set_title("Fig 1.4 – Correctness by question level and technique"); ax.legend(title="level", fontsize=8)
save(fig, "fig1_4_levels"); plt.show()
"""),
md(r"""
### 1.5 Deep analysis
**(a) Three examples where CoT reasoned better than Zero-Shot.** These are selected automatically as the questions with the largest key-fact coverage gain of CoT over Zero-Shot, with correctness as the tie-breaker.
"""),
code(r"""
piv = t1.pivot_table(index="qid", columns="technique", values=["coverage", "correct"])
gain = (piv[("coverage", "CoT")] - piv[("coverage", "Zero-Shot")]) + 0.5 * (piv[("correct", "CoT")].astype(float) - piv[("correct", "Zero-Shot")].astype(float))
examples = gain.sort_values(ascending=False).head(3).index.tolist()
for qid in examples:
    q = next(x for x in T1Q if x["qid"] == qid); o = t1_outputs[qid]
    zs, cot = t1[(t1.qid == qid) & (t1.technique == "Zero-Shot")].iloc[0], t1[(t1.qid == qid) & (t1.technique == "CoT")].iloc[0]
    display(Markdown(f"**{qid} ({q['category']}) – {q['question']}**  \nZero-Shot: coverage {zs.coverage:.2f}, correct={zs.correct}, depth={zs.depth} · "
                     f"CoT: coverage {cot.coverage:.2f}, correct={cot.correct}, depth={cot.depth}"))
    print("ZERO-SHOT:", textwrap.shorten(o["Zero-Shot"].response, 600))
    print("COT      :", textwrap.shorten(o["CoT"].response, 900), "\n")
"""),
md(r"""
**(b) ToT: valuable for ambiguous questions, overkill for factual ones.** The table compares ToT with CoT and Zero-Shot on cost and correctness by question level. The evaluator's path scores show how it picked between competing interpretations.
"""),
code(r"""
cmp = t1[t1.technique.isin(["Zero-Shot", "CoT", "ToT"])].copy()
cmp["kind"] = np.where(cmp.level == "ambiguous", "ambiguous", np.where(cmp.level == "simple", "simple factual", "medium/complex"))
tot_tab = cmp.groupby(["kind", "technique"]).agg(correct_pct=("correct", lambda s: 100 * s.mean()), coverage=("coverage", "mean"),
                                                 tokens=("in_tok", lambda s: s.mean()), latency_s=("latency_s", "mean")).round(2)
display(Markdown("**Table 1.5 – ToT vs CoT vs Zero-Shot by question kind.**")); display(tot_tab)
amb = [q for q in T1Q if q["category"] == "ambiguous"][0]
tot = t1_outputs[amb["qid"]]["ToT"]
print(f"ToT evaluator on '{amb['question']}': scores={tot.extra['scores']} -> chose path {tot.extra['best']}: {tot.extra['reason']}")
"""),
md(r"""
**(c) How APE finds prompts a human might miss.** The human prompts optimise for what the author *thinks* matters. APE searches the space of instructions empirically and keeps whatever scores best on held-aside questions. The table below compares the selected APE instruction with the human-written ones on the test set.
"""),
code(r"""
display(summ.loc[["Zero-Shot", "CoT", "APE"], ["correctness", "key_fact_coverage", "depth", "sources_ref", "in_tok", "cost_per_1k_usd"]].round(2))
print("APE instruction:", textwrap.fill(APE_BEST, 120))
"""),
md(r"""
### 1.6 Inference — Task 1
<!--T1_INFERENCE-->
"""),
]
