from common import code, md, module

CELLS = [
md(r"""
---
# Module 3 · Task 5 — Instruction Tuning, RLHF, PEFT & Paradigm Differentiation (2 marks)

## Part A — Instruction tuning data via Self-Instruct (0.6 marks)
Self-Instruct (Wang et al., 2023) bootstraps an instruction-tuning dataset from a handful of human-written seed tasks:
1. **Seed tasks.** 5 research tasks written by hand.
2. **Instruction generation.** The LLM writes new instructions from 3 sampled seed tasks plus 2 previously generated ones (in-context bootstrapping), at T = 0.9.
3. **Classification identification.** Is the output a label from a finite set (Yes) or open text (No)?
4. **Instance generation**, in two modes:
   * **Output-first** for classification tasks: list the label set, pick a *target label* (we rotate through label positions), then write an input that has that label.
   * **Input-first** for everything else: write the input, then the output.
5. **Filtering.**
   * ROUGE-L ≥ 0.7 against the existing pool counts as a duplicate. ROUGE-L is implemented from scratch with LCS dynamic programming.
   * Heuristics reject non-text modalities, empty or too-short outputs, output = input, and labels outside the label set.
   * An LLM-judge quality score below 3/5 is rejected.

**Why output-first for classification?** With input-first generation, the model writes "natural" inputs that cluster on the majority or most typical label (for example, most claims it writes are "supported"), which gives a skewed label distribution. Fixing the label *before* writing the input lets us choose the class mix, here by rotating label positions. This balances the classes and forces the model to produce minority-class and borderline examples.
"""),
module("selfinstruct.py"),
code(r"""
from ra.selfinstruct import SEED_TASKS, generate_instructions, classify_instructions, generate_instances, llm_quality, filter_dataset, rouge_l, diversity, heuristic_issues
display(Markdown("**Table 5.1 – The 5 manually written seed tasks.**")); display(pd.DataFrame(SEED_TASKS))
gen = generate_instructions(llm, target=50)
classify_instructions(llm, gen)
inst = generate_instances(llm, gen)
llm_quality(llm, inst)
for x in inst: x["heuristic_ok"] = not heuristic_issues(x)
kept, reasons = filter_dataset([dict(x) for x in inst])
si = pd.DataFrame(inst).reindex(columns=["instruction", "is_classification", "method", "input", "output", "quality", "heuristic_ok", "labels"])
print(f"Generated instructions: {len(gen)} | instances: {len(inst)} | kept after filtering: {len(kept)}")
print("Removed by reason:", reasons)
display(si[["instruction", "is_classification", "method", "input", "output", "quality"]].head(10))
cls_df = si[si.is_classification]
if len(cls_df): print("\nOutput-first label distribution (classification instances):", cls_df.output.value_counts().to_dict())
div_before, div_after = diversity(inst), diversity(kept)
print("Diversity before:", div_before, "| after:", div_after)
import json as _json
with open("outputs/self_instruct_dataset.jsonl", "w", encoding="utf-8") as f:
    for x in kept: f.write(_json.dumps({k: x.get(k) for k in ("instruction", "input", "output", "is_classification", "method", "quality")}, ensure_ascii=False) + "\n")
RESULTS["task5"] = {"si_generated": len(inst), "si_kept": len(kept), "si_removed": reasons, "si_cls_share": float(si.is_classification.mean()),
                    "si_quality_before": float(si.quality.mean()), "si_quality_after": float(np.mean([x["quality"] for x in kept])) if kept else None}
"""),
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
vc = si.is_classification.map({True: "Classification (Yes)", False: "Non-classification (No)"}).value_counts()
if len(vc): vc.plot(kind="pie", ax=axes[0], autopct="%1.0f%%", colors=[PALETTE[0], PALETTE[1]])
axes[0].set_ylabel(""); axes[0].set_title("Fig 5.1 – Task types")
kq = pd.DataFrame(kept).reindex(columns=["instruction", "is_classification", "method", "input", "output", "quality", "heuristic_ok", "labels"])
qb = pd.DataFrame({"before filtering": [si.quality.mean(), 100 * si.heuristic_ok.mean(), len(si)],
                   "after filtering": [kq.quality.mean() if len(kq) else 0, 100.0, len(kq)]}, index=["mean LLM quality (1-5)", "heuristic pass rate (%)", "# examples"])
qb.iloc[:1].plot(kind="bar", ax=axes[1], rot=0, color=[PALETTE[5], PALETTE[2]]); bar_labels(axes[1], "{:.2f}"); axes[1].set_ylim(0, 5.5); axes[1].set_title("Fig 5.2 – Quality before/after filtering")
qb.iloc[2:].plot(kind="bar", ax=axes[2], rot=0, color=[PALETTE[5], PALETTE[2]]); bar_labels(axes[2], "{:.0f}"); axes[2].set_title("Fig 5.3 – Dataset size before/after filtering")
plt.tight_layout(); save(fig, "fig5_selfinstruct"); plt.show()
display(qb.round(2))
"""),
md(r"""
### A.7 Instruction tuning vs traditional fine-tuning
| Feature | Traditional fine-tuning | Instruction tuning |
|---|---|---|
| Task scope | Single task (e.g. only "classify study design") | Many tasks phrased as natural-language instructions (our 50 research tasks) |
| Training data | Task-specific (input, label) pairs | Diverse (instruction, input, output) triples, often synthetic as here |
| Generalisation | Limited to the trained task and format | Potential transfer to *unseen* instructions, because the model learns to follow instructions in general |
| Zero-shot capability | Limited; new tasks need new data | May improve zero-shot instruction following (FLAN, T0, Self-Instruct results) |
| Best use case | A narrow, stable, high-volume task | A general research assistant that must handle open-ended, varied requests |

## Part B — RLHF, DPO & PEFT (0.8 marks)
### B.1 DPO loss calculation (0.3 marks)
DPO optimises the policy directly on preference pairs, using the closed-form optimal policy of the KL-regularised RLHF objective:

$$\mathcal{L}_{DPO} = -\log \sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\big]\Big)$$
"""),
module("alignment.py"),
code(r"""
from ra.alignment import dpo_loss, peft_comparison, memory_for_model, lora_params, grpo_advantages
d = dpo_loss(pw_theta=0.60, pl_theta=0.15, pw_ref=0.35, pl_ref=0.45, beta=1.0)
steps = pd.DataFrame([
  ("1. Likelihood ratio (preferred)",  "π_θ(y_w)/π_ref(y_w) = 0.60 / 0.35", d["ratio_w"]),
  ("1. Likelihood ratio (rejected)",   "π_θ(y_l)/π_ref(y_l) = 0.15 / 0.45", d["ratio_l"]),
  ("2. Log ratio (preferred)",         "ln(1.7143)", d["log_ratio_w"]),
  ("2. Log ratio (rejected)",          "ln(0.3333)", d["log_ratio_l"]),
  ("3. Difference",                    "0.5390 − (−1.0986)", d["difference"]),
  ("4. × β (β = 1)",                   "1 × 1.6376", d["beta_x_diff"]),
  ("5. Sigmoid",                       "σ(1.6376) = 1/(1+e^−1.6376)", d["sigmoid"]),
  ("6. DPO loss",                      "−ln σ(·)", d["loss"]),
], columns=["step", "computation", "value"])
display(Markdown("**Table 5.2 – DPO loss, step by step.**")); display(steps.round(4))
RESULTS["task5"]["dpo_loss"] = d["loss"]
"""),
md(r"""
**Interpretation.**
* The current policy has **raised** the probability of the balanced 3-source summary relative to the reference (ratio 1.71, implicit reward +0.54) and **lowered** the one-sided summary (ratio 0.33, implicit reward −1.10). The preference margin is 1.64 nats.
* σ(1.64) = 0.84: under the Bradley-Terry model the policy now prefers y_w with 84 % probability.
* The loss of **0.178** is small (it would be ln 2 = 0.693 for a policy identical to the reference), so this pair is already mostly learned. Its gradient is scaled by (1 − σ) = 0.16, so the update pressure from this pair is now weak.
* For a research assistant, DPO is pushing the model towards *balanced, multi-source* summaries, which is exactly the alignment target.

### B.2 PEFT analysis (0.3 marks)
Assumptions for a Llama-2-7B-class model (6.74 B parameters, 32 layers, d = 4096):
* LoRA rank r = 16 on the q, k, v, o projections, so each adapted d×d matrix adds r·(d + d) parameters.
* Full fine-tuning with AdamW in mixed precision needs about 16 bytes/parameter: bf16 weights + bf16 gradients + fp32 master weights + two fp32 Adam moments.
* QLoRA stores the frozen base in 4-bit NF4 (0.5 bytes/parameter).
* About 4 GB of activations is added for every method (sequence length 2k, gradient checkpointing).
"""),
code(r"""
pt = peft_comparison(16); display(Markdown("**Table 5.3 – Full FT vs LoRA vs QLoRA on a 7B model.**")); display(pt)
mm = memory_for_model(70e9); display(Markdown("**Table 5.4 – Memory for a 70B model, M = P × B × 1.2.**")); display(mm)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].bar(pt.Method, pt["GPU VRAM (GB, approx.)"], color=PALETTE[:3]); bar_labels(axes[0], "{:.1f}"); axes[0].set_ylabel("GB"); axes[0].set_title("Fig 5.4 – Approx. training VRAM, 7B model")
axes[1].bar(mm.Precision, mm["Memory M = P x B x 1.2 (GB)"], color=PALETTE[3:7]); bar_labels(axes[1], "{:.0f}"); axes[1].set_ylabel("GB"); axes[1].set_title("Fig 5.5 – 70B model memory by precision")
plt.tight_layout(); save(fig, "fig5_peft"); plt.show()
RESULTS["task5"]["lora_trainable_pct"] = float(pt["Trainable %"].iloc[1]); RESULTS["task5"]["mem70b"] = mm.set_index("Precision")["Memory M = P x B x 1.2 (GB)"].to_dict()
"""),
md(r"""
**QLoRA's three innovations** (Dettmers et al., 2023)
1. **4-bit NormalFloat (NF4).** Pre-trained weights are roughly normally distributed. NF4 places its 16 quantisation levels at the quantiles of a standard normal, which is information-theoretically optimal for such data. It beats uniform INT4 at the same 0.5 bytes/parameter.
2. **Double quantisation.** The per-block (64-weight) fp32 scaling constants are themselves quantised to 8-bit in blocks of 256. Constant overhead drops from about 0.5 to about 0.127 bits per parameter, roughly 3 GB saved on a 65B model.
3. **Paged optimisers.** Optimizer states live in NVIDIA unified memory and are paged to CPU RAM during memory spikes (long sequences, gradient checkpointing), which avoids out-of-memory crashes.

Gradients flow through the frozen 4-bit base, which is de-quantised to bf16 on the fly, into bf16 LoRA adapters. This is how a 65B model can be fine-tuned on a single 48 GB GPU.

### B.3 PPO vs GRPO, and advanced alignment (0.2 marks)
**PPO (actor-critic).** RLHF with PPO trains a **reward model** on human preferences, then optimises the policy (actor) with a clipped surrogate objective. Advantages are computed against a **learned value function (critic)** of similar size to the policy, under a KL penalty to the reference model. That means four models in memory (policy, reference, reward, critic). It is unstable and sensitive to hyper-parameters.

**GRPO (Group Relative Policy Optimisation, DeepSeekMath 2024).** GRPO **drops the critic**. For each prompt it samples a *group* of G responses, scores them with the reward function, and uses the **group-normalised reward as the advantage**, Âᵢ = (rᵢ − mean(r)) / std(r). The group itself is the baseline. This roughly halves memory and works well with verifiable rewards (maths, code), which is the recipe behind DeepSeek-R1-style reasoning models.
"""),
code(r"""
rewards = [4.5, 2.0, 3.0, 4.0]   # e.g. judge scores of 4 sampled syntheses for one research prompt
adv = grpo_advantages(rewards)
display(pd.DataFrame({"response": ["balanced, 3 cited sources", "one-sided, 1 source", "uncited summary", "balanced, 2 sources"],
                      "reward": rewards, "GRPO advantage (r-mean)/std": adv.round(3)}))
print("Responses above the group mean get positive advantage (reinforced); no value network is needed.")
"""),
md(r"""
**How advanced alignment methods address RLHF's limitations**
| Method | Key idea | RLHF limitation addressed |
|---|---|---|
| **Constitutional AI / RLAIF** (Bai et al., 2022) | A written *constitution* of principles. The model critiques and revises its own outputs (supervised stage), then an AI preference model trained on AI feedback replaces human labels (RL stage) | Cost and scale of human labelling; inconsistent annotators; makes the values explicit and auditable |
| **SPIN: Self-Play fine-tuning** (Chen et al., 2024) | The model plays against its previous iteration. It learns to tell its own generations from human SFT data, using a DPO-like loss, and iterates until it cannot | Needs no new preference data or reward model; squeezes more out of the existing SFT data |
| **Process Reward Models** (Lightman et al., 2023) | Reward each *reasoning step*, not just the final outcome | Outcome rewards are sparse and reward lucky wrong reasoning (reward hacking); step-level credit assignment gives better multi-step reasoning, which matters for research synthesis chains |
| **DPO** (Rafailov et al., 2023) | Closed-form re-parameterisation; a classification loss on preference pairs | Removes the separate reward model and the unstable PPO loop |
| **GRPO** | Group-relative advantages | Removes the critic (memory, instability) |

## Part C — Paradigm differentiation: Prompting vs RAG vs Fine-tuning (0.6 marks)
### C.1 Conceptual comparison for research tasks
| Criterion | Prompting (zero/few-shot) | RAG | Fine-tuning |
|---|---|---|---|
| Knowledge freshness | Frozen at the pre-training cut-off | **Live**: update the index, not the model | Frozen at training time; re-train to update |
| Citation / provenance | None (or hallucinated) | **Native**: every claim maps to a retrieved source | None; knowledge is baked into the weights |
| Handling conflicting evidence | Only what the model memorised, with no credibility signal | Retrieves both sides with metadata (credibility, date) | Learns the majority view in the training data |
| Hallucination risk | High on specifics (numbers, studies) | Lower; verifiable against context | Medium; can memorise but also confabulate |
| Behaviour / style control | Limited by prompt length | Same as prompting | **Strong**: tone, format and domain conventions learned |
| Upfront cost | None | Index build (cheap) | Data + GPU training |
| Per-query cost | Lowest | + retrieval and longer prompts | Low (short prompts) |
| Best for | Quick, general questions | **Evidence-based research questions** | Fixed formats, domain jargon, consistent style |

### C.2 Empirical comparison on 5 research questions
The same 5 contested held-out research questions are answered three ways:
* **Zero-Shot prompting:** closed-book, no evidence.
* **RAG:** the Task 4 evidence and synthesis.
* **Fine-tuning (simulated).** We cannot train a model through the Gemini API, so the *behaviour* of a fine-tuned model is approximated by conditioning on 8 of our filtered Self-Instruct examples (an in-context proxy for the learned style), with **no retrieval**. Cost is reported **without** the demonstration tokens, because a truly fine-tuned model would not need them. This is a stated approximation, not a real fine-tune.

**Metrics.** *Accuracy* uses the Task 1 correctness criterion. *Hallucination rate* is the share of claims the verifier cannot support from the topic's full credible document set.
"""),
code(r"""
from ra.synthesis import verify_claims, split_claims, docs_to_evidence
pq = [T4[k]["q"] for k in T4 if T4[k]["q"]["nature"] == "contested"][:5]
demos = (kept or inst)[:8]
demo_msgs = []
for x in demos:
    demo_msgs += [{"role": "user", "content": f"{x['instruction']}\n{x.get('input','')}"}, {"role": "assistant", "content": x.get("output", "")}]
par_rows = []
for q in pq:
    ref_ev = docs_to_evidence([d for d in D.topic_docs(q["topic_id"]) if credibility(d)["score"] >= 0.5])
    r = llm.chat(f"Research question: {q['question']}\nAnswer in 120-200 words.", purpose="t5_paradigm_zeroshot")
    zs = (r.text, r.input_tokens, r.output_tokens, r.latency_s)
    rr = T4[q["qid"]]["res"]
    rag = (rr.answer, rr.tokens_in, rr.tokens_out, rr.llm_latency)
    msgs = [{"role": "system", "content": "You are a research assistant fine-tuned on research workflows."}] + demo_msgs + \
           [{"role": "user", "content": f"Answer the research question in 120-200 words.\n{q['question']}"}]
    r = llm.chat(msgs, purpose="t5_paradigm_ft_sim")
    demo_tok = sum(count_tokens(m["content"]) for m in demo_msgs)
    ft = (r.text, r.input_tokens - demo_tok, r.output_tokens, r.latency_s)
    for name, (ans, tin, tout, lat) in {"Zero-Shot prompting": zs, "RAG (Task 4 pipeline)": rag, "Fine-tuning (simulated)": ft}.items():
        j = judge_correctness(llm, q, ans, purpose="t5_paradigm_judge")
        cl = split_claims(ans) or [{"text": ans[:300], "cited": []}]
        v = verify_claims(llm, cl, ref_ev, purpose="t5_paradigm_verify")
        par_rows.append({"qid": q["qid"], "paradigm": name, "correct": j["correct"], "coverage": j["coverage"], "halluc_rate": np.mean([c["flagged"] for c in v]),
                         "latency_s": lat, "cost_usd": config.cost_usd(tin, tout)})
pdm = pd.DataFrame(par_rows).groupby("paradigm").agg(accuracy_pct=("correct", lambda s: 100 * s.mean()), key_fact_coverage=("coverage", "mean"),
                                                      hallucination_pct=("halluc_rate", lambda s: 100 * s.mean()), latency_s=("latency_s", "mean"), cost_per_query_usd=("cost_usd", "mean"))
display(Markdown("**Table 5.5 – Paradigm comparison on 5 contested research questions.**")); display(pdm.round(4))
fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
for ax, col, ttl in zip(axes, ["accuracy_pct", "hallucination_pct", "cost_per_query_usd"], ["Accuracy (%)", "Hallucination rate (%)", "Cost per query ($, GPT-4o pricing)"]):
    pdm[col].plot(kind="bar", ax=ax, color=PALETTE[:3], rot=10); ax.set_title(ttl); bar_labels(ax, "{:.3g}")
fig.suptitle("Fig 5.6 – Prompting vs RAG vs Fine-tuning (simulated)", fontweight="bold"); plt.tight_layout(); save(fig, "fig5_6_paradigms"); plt.show()
RESULTS["task5"]["paradigms"] = pdm.round(4).to_dict()
"""),
md(r"""
### C.3 Decision framework: given a new research task, should I use Prompting, RAG or Fine-tuning?
"""),
code(r"""
N = {"s": (6, 7.4, "New research task", PALETTE[0], 2.4),
     "q1": (6, 6.0, "Needs facts beyond the model's training\n(recent, niche, proprietary, citable)?", PALETTE[5], 4.6, 0.9),
     "q2": (2.6, 4.4, "Must every claim be\ncited / auditable?", PALETTE[5], 3.2, 0.9),
     "q3": (9.4, 4.4, "Need a consistent output format, style\nor domain behaviour the prompt can't fix?", PALETTE[5], 4.4, 0.9),
     "rag": (1.2, 2.6, "RAG", PALETTE[2], 1.6), "ragp": (4.0, 2.6, "RAG (+ light prompting)", PALETTE[2], 2.4),
     "q4": (9.4, 2.7, "Enough quality examples (≥ 1k) and\nstable task & budget for training?", PALETTE[5], 4.2, 0.9),
     "ft": (7.6, 0.9, "Fine-tune (LoRA/QLoRA);\n+ RAG if facts change = HYBRID", PALETTE[1], 3.4, 0.9),
     "pr": (11.0, 0.9, "Prompting\n(few-shot / CoT / APE)", PALETTE[4], 2.4, 0.9)}
E = [("s", "q1", ""), ("q1", "q2", "yes"), ("q1", "q3", "no"), ("q2", "rag", "yes"), ("q2", "ragp", "no"), ("q3", "q4", "yes"), ("q3", "pr", "no"),
     ("q4", "ft", "yes"), ("q4", "pr", "no")]
fig, ax = viz.diagram(N, E, "Figure 5.7 – Decision tree: Prompting vs RAG vs Fine-tuning for a research task", figsize=(13, 7), xlim=(0, 12.5), ylim=(0.2, 8))
save(fig, "fig5_7_decision_tree"); plt.show()
"""),
md(r"""
### C.4 The hybrid approach: RAG for knowledge, fine-tuning for behaviour
RAG and fine-tuning solve *different* problems, so they compose:
* **RAG supplies knowledge.** It is fresh, attributable and conflict-aware: the latest trials, retracted papers flagged, credibility metadata attached.
* **Fine-tuning supplies behaviour**, with LoRA/QLoRA on Self-Instruct data like Part A's, aligned with DPO preferences like B.1's. The model learns to *always* produce structured evidence syntheses: claims ↔ citations, an explicit "evidence for / against" section, calibrated hedging, APA formatting, and refusal to treat tabloids as evidence.

**Research example.** A systematic-review assistant for a hospital's evidence team:
1. RAG retrieves from PubMed, guideline repositories and the hospital's own protocols, updated weekly.
2. A QLoRA adapter, trained on about 2k reviewer-approved summaries and DPO-tuned to prefer balanced multi-source summaries (exactly the y_w vs y_l pair in B.1), writes the synthesis in the team's GRADE-style format.

Neither half is enough alone. Fine-tuning alone would go stale and could not cite. RAG alone would need a long instruction prompt on every query and would still drift in format.
"""),
md(r"""
### Inference — Task 5
<!--T5_INFERENCE-->
"""),
]
