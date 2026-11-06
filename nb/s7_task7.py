from common import code, md, module

CELLS = [
md(r"""
---
# Module 4 · Task 7 — RAG & Agent Architecture Decision & Cost Engineering (1 mark)

## Part A — Architecture decision table (0.4 marks)
Every row is filled **programmatically from the experiment results stored in `RESULTS`**. This keeps the cited numbers identical to those measured above. Where two options are within the noise of an 8-question evaluation (|Δ| < 0.15 on the 1–5 scale), the row says so and explains the tie-break.
"""),
code(r"""
T3, T4r = RESULTS["task3"], RESULTS["task4"]
ret = pd.DataFrame(T3["retrieval"]); ret_nq = ret[ret.benchmark.str.startswith("NQ")].set_index("method"); ret_cur = ret[ret.benchmark.str.startswith("Curated")].set_index("method")
memq = pd.DataFrame(T3["memory_ablation"]); varq = pd.DataFrame(T3["rag_variants"]); abl3 = pd.DataFrame(T3["retrieval_ablation"])
def close(a, b, tol=0.15): return abs(a - b) < tol
rows = []
best_ret = ret_nq["Recall@10"].drop("hybrid+CE").idxmax()
# The pipeline passes the top-5 to the LLM, so Recall@5 *after* re-ranking is what matters for synthesis.
best_r5 = ret_nq["Recall@5"].idxmax()
rows.append(("Retrieval method", "Hybrid RRF (BM25 + dense) + cross-encoder re-ranking" if best_r5 == "hybrid+CE" else best_r5,
             f"Table 3.4: NQ Recall@10 BM25 {ret_nq.loc['bm25','Recall@10']:.2f} / dense {ret_nq.loc['dense','Recall@10']:.2f} / hybrid {ret_nq.loc['hybrid','Recall@10']:.2f}; "
             f"curated multi-source Recall@10 {ret_cur.loc['bm25','Recall@10']:.2f} / {ret_cur.loc['dense','Recall@10']:.2f} / {ret_cur.loc['hybrid','Recall@10']:.2f}. "
             f"Dense alone has the best Recall@10 ({best_ret}), but only the top-5 reaches the prompt: NQ Recall@5 is highest for hybrid + cross-encoder "
             f"({ret_nq.loc['hybrid+CE','Recall@5']:.2f} vs dense {ret_nq.loc['dense','Recall@5']:.2f}), and BM25 keeps exact names/numbers retrievable. "
             f"On the small curated set dense is ahead (Recall@5 {ret_cur.loc['dense','Recall@5']:.2f} vs {ret_cur.loc['hybrid+CE','Recall@5']:.2f}), so the two are close; "
             f"we keep hybrid + CE for robustness and accept ~{ret_nq.loc['hybrid+CE','latency_ms']/1000:.1f}s re-ranking latency."))
bc = T3["best_chunking"]
rows.append(("Chunking strategy", f"{bc['strategy'].capitalize()} {bc['size']} tokens, overlap {bc['overlap']}",
             f"Table 3.3 / Fig 3.1: best Answer Recall@5 = {bc['recall5']:.2f} among 12 strategy × overlap configurations; ties broken by fewer context tokens."))
qd, qe = memq.loc["D: full context engineering", "quality"], memq.loc["E: D + cross-encoder", "quality"]
ld, le = memq.loc["D: full context engineering", "total_latency_s"], memq.loc["E: D + cross-encoder", "total_latency_s"]
rer = "Cross-encoder ON" if qe - qd >= 0.15 or (close(qe, qd) and ret_nq.loc["hybrid+CE", "Recall@5"] > ret_nq.loc["hybrid", "Recall@5"]) else "Cross-encoder OFF"
rows.append(("Re-ranking", rer, f"Table 3.13 Config E vs D: quality {qe:.2f} vs {qd:.2f} (Δ {qe-qd:+.2f}), latency {le:.2f}s vs {ld:.2f}s (+{le-ld:.2f}s); "
             f"Table 3.4: Recall@5 hybrid {ret_nq.loc['hybrid','Recall@5']:.2f} → +CE {ret_nq.loc['hybrid+CE','Recall@5']:.2f}." + (" Quality difference within noise; kept because of the recall gain at sub-second cost." if close(qe, qd) else "")))
bv = varq["quality"].idxmax()
rows.append(("Advanced RAG variant", bv, "Table 3.14: quality " + ", ".join(f"{k} {v:.2f}" for k, v in varq["quality"].items()) + "; coverage " +
             ", ".join(f"{k} {v:.2f}" for k, v in varq["coverage"].items()) + "; latency " + ", ".join(f"{k} {v:.1f}s" for k, v in varq["latency_s"].items()) +
             (". Within noise of the baseline -> prefer the cheaper option per query, use the variant adaptively." if close(varq.loc[bv, "quality"], varq.loc["Baseline hybrid", "quality"]) and bv != "Baseline hybrid" else "")))
bm = memq["quality"].idxmax()
rows.append(("Memory design", "Combined (working + episodic + semantic) with full context engineering",
             "Table 3.13: quality " + ", ".join(f"{k.split(':')[0]} {v:.2f}" for k, v in memq["quality"].items()) + f"; best = {bm}."))
t2r = RESULTS["task2"]
rows.append(("Agent loop", "ReAct with parallel function calling", f"§2.3: parallel calls cut tool latency by {t2r['parallel_reduction_pct']:.0f}% with identical results "
             f"(same calls, same outputs); tool-selection accuracy {100*t2r['tool_selection_accuracy']:.0f}%, task completion {100*t2r['task_completion']:.0f}%."))
sq = T4r["synthesis_quality"]
rows.append(("Synthesis strategy", "Multi-source with conflict resolution", "Table 4.8: quality " + ", ".join(f"{k} {v:.2f}" for k, v in sq.items()) +
             f"; conflict gold-agreement (credibility) {T4r['conflicts_dev']['gold_agreement_pct'].get('credibility', float('nan')):.0f}% vs majority {T4r['conflicts_dev']['gold_agreement_pct'].get('majority', float('nan')):.0f}% (Table 4.4)."))
rows.append(("Hallucination control", "Verification ON (+ self-correction)", f"Fig 4.3: hallucination rate {100*T4r['halluc_before']:.1f}% → {100*T4r['halluc_after']:.1f}% "
             f"after the verification + self-correction pass; detector F1 {T4r['detector']['f1']:.2f} (Fig 4.2)."))
tri = T4r["triad"]; fails = {k: v for k, v in T4r["triad_failures"].items() if k != "none"}
stage_name = {"retrieval": "Context Relevance (retrieval)", "grounding": "Groundedness (grounding)", "answer": "Answer Relevance (answer)"}
weakest = stage_name[max(fails, key=fails.get)] if fails else min(tri, key=tri.get).replace("_", " ").title()
rows.append(("RAG Triad weak stage", weakest, "Table 4.9: stage that failed most often (score < 4) " + (f"= {max(fails, key=fails.get)} in {max(fails.values())} of {sum(T4r['triad_failures'].values())} queries" if fails else "none; lowest mean used") +
             "; means " + ", ".join(f"{k} {v:.2f}" for k, v in tri.items()) + "."))
dec = pd.DataFrame(rows, columns=["Design choice", "Selected option", "Justification (from our experiments)"])
display(Markdown("**Table 7.1 – RAG & agent architecture decision table.**")); display(dec)
RESULTS["task7"] = {"decisions": dec.to_dict("records")}
"""),
md(r"""
## Part B — API-level cost engineering (0.4 marks)
**Assumptions, used consistently** (from the assignment):
* 2,000 input + 500 output tokens per query
* 10,000 queries/day, 30-day month
* prices per 1M tokens: GPT-4o $5 in / $15 out; small LM $0.10 / $0.30; MoE mid $0.60 / $1.10

Per-query cost:
* GPT-4o = 2000·5/10⁶ + 500·15/10⁶ = **$0.0175**
* small LM = **$0.00035**
* MoE = **$0.00175**

**Strategy B (routing 80/15/5):** 0.8·0.00035 + 0.15·0.00175 + 0.05·0.0175 = **$0.0014175 per query**.
"""),
module("cost.py"),
code(r"""
from ra.cost import routing_table, optimisation_table, per_query, monthly
rt = routing_table(); display(Markdown("**Table 7.2 – Flat GPT-4o vs intelligent routing (10,000 queries/day, 30 days).**")); display(rt.round(5))
# cacheable prefix share and redundant share come from the measured Task 3 context breakdown / classification
tbk = pd.DataFrame(RESULTS["task3"]["token_breakdown"]).set_index("Component")["Tokens"]
input_total = tbk.drop(["TOTAL", "Reserved for Output"], errors="ignore").sum()
cacheable = (tbk.get("System Prompt", 0) + tbk.get("Tool Definitions", 0)) / input_total
redundant = RESULTS["task3"]["context_categories_pct"].get("potentially redundant", 0) / 100
# Context reduction. Table 3.11 found no 'potentially redundant' component in this query (every piece is >= 0.45 similar to
# the query), so dropping redundant context saves nothing here. The largest *operational* component is the tool schemas
# (Table 3.10), so the measured reduction is: send compact schemas (name, one-line description, parameter names/types/enums,
# required list) instead of the full schemas with bad/good examples and per-parameter prose.
import json as _json
from ra.tools import TOOL_SCHEMAS
from ra.tokens import count_tokens
def _compact(sch):
    f = sch["function"]; props = f.get("parameters", {}).get("properties", {})
    return {"name": f["name"], "description": f.get("description", "").split(". ")[0][:120],
            "parameters": {"type": "object", "required": f.get("parameters", {}).get("required", []),
                           "properties": {k: {kk: vv for kk, vv in v.items() if kk in ("type", "enum", "minimum", "maximum")} for k, v in props.items()}}}
full_tool_tok = count_tokens(_json.dumps([s["function"] for s in TOOL_SCHEMAS], separators=(",", ":")))
compact_tool_tok = count_tokens(_json.dumps([_compact(s) for s in TOOL_SCHEMAS], separators=(",", ":")))
reduction = (full_tool_tok - compact_tool_tok + redundant * input_total) / input_total
print(f"Tool schemas: full {full_tool_tok} tokens -> compact {compact_tool_tok} tokens; input context {input_total:.0f} tokens "
      f"-> context reduction {reduction:.0%} of input tokens (redundant share {redundant:.0%} + schema compaction)")
ot = optimisation_table(cacheable_input_share=float(cacheable), context_reduction_share=float(reduction))
display(Markdown(f"**Table 7.3 – Cost optimisation strategies.** Cached prefix = system prompt + tool schemas = {cacheable:.0%} of input tokens (Table 3.10); "
                 f"context reduction = 'potentially redundant' share ({redundant:.0%}, Table 3.11) + compact tool schemas "
                 f"({full_tool_tok}→{compact_tool_tok} tokens) = {reduction:.0%} of input tokens."))
display(ot.round(2))
RESULTS["task7"].update({"routing": rt.round(4).to_dict("records"), "optimisations": ot.round(2).to_dict("records")})
"""),
md(r"""
## Part C — Final recommendation & model comparison dashboard (0.2 marks)
To ground routing in *measured* quality rather than price alone, three model tiers available to us synthesise answers to the 8 dev questions from **identical evidence** (Config D evidence from Task 3):
* **Small LM:** Qwen2.5-14B-Instruct (local, Q4_K_M), priced as "Small LM".
* **Mid tier:** gemini-3.1-flash-lite, priced as "MoE Mid".
* **Large tier:** gpt-oss-120b (Groq, 120B MoE), priced as GPT-4o, the "dense LLM". (Gemini Pro has a quota of 0 on the free tier.)

Each answer is judged with the same rubric. *Caveat:* the dense tier is the same model as the judge, which may favour it (self-preference); treat its score as an upper bound.
"""),
code(r"""
tier_rows = []
for q in dev_cq:
    tr = traces[q["qid"]]
    per = {s: [] for s in ("academic", "wikipedia", "news")}
    for h in tr["hits"]: per[h.source].append(h)
    for s in per:
        b = {}
        for h in per[s]:
            if h.doc_id not in b or h.rank < b[h.doc_id].rank: b[h.doc_id] = h
        per[s] = sorted(b.values(), key=lambda h: h.rank)
    ev = hits_to_evidence([h for h, _ in weighted_rrf(per, SOURCE_WEIGHT_PRESETS["credibility"], doc_level_credibility=True)][:8], 140)
    for tier in ("tier_small", "tier_moe", "tier_dense"):
        ans, r = synthesize(llm, q["question"], ev, purpose=f"t7_{tier}", role=tier)
        qual = judge_synthesis_quality(llm, q, ans, ev, purpose="t7_tier_judge")
        tier_rows.append({"qid": q["qid"], "tier": tier, "model": r.model, "quality": qual["overall"], "in": r.input_tokens, "out": r.output_tokens, "latency_s": r.latency_s,
                          "cost_usd": config.cost_usd(r.input_tokens, r.output_tokens, config.TIER_TO_PRICE[tier])})
tdf = pd.DataFrame(tier_rows)
tm = tdf.groupby("tier").agg(model=("model", "first"), quality=("quality", "mean"), latency_s=("latency_s", "mean"), cost_per_query=("cost_usd", "mean")).reindex(["tier_small", "tier_moe", "tier_dense"])
display(Markdown("**Table 7.4 – Model tiers on identical evidence (8 dev questions).**")); display(tm.round(5))
# Routing simulation using the measured router decisions of Task 6 (simple vs complex) + contested -> dense
share_simple = RESULTS["task6"]["nq50"]["routed_simple_%"] / 100
mix_policy = {"tier_small": 0.80, "tier_moe": 0.15, "tier_dense": 0.05}
routed_q = sum(mix_policy[t] * tm.loc[t, "quality"] for t in mix_policy); routed_c = sum(mix_policy[t] * tm.loc[t, "cost_per_query"] for t in mix_policy)
points = pd.concat([tm[["quality", "cost_per_query"]], pd.DataFrame({"quality": [routed_q], "cost_per_query": [routed_c]}, index=["routed 80/15/5"])])
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
names = {"tier_small": "Small\n(Qwen2.5-14B local)", "tier_moe": "Mid\n(gemini-3.1-flash-lite)", "tier_dense": "Large\n(gpt-oss-120b ≈ GPT-4o)", "routed 80/15/5": "Routed\n80/15/5"}
axes[0].bar([names[i] for i in points.index], points.cost_per_query * 1000, color=PALETTE[:4]); bar_labels(axes[0], "{:.3f}")
axes[0].set_ylabel("$ per 1,000 queries (measured tokens)"); axes[0].set_title("Fig 7.1 – Cost per query vs model"); axes[0].tick_params(axis="x", labelsize=8)
for i, (idx, rrow) in enumerate(points.iterrows()):
    axes[1].scatter(rrow.cost_per_query * 1000, rrow.quality, s=180, color=PALETTE[i], edgecolor="k", zorder=3); axes[1].annotate(names[idx], (rrow.cost_per_query * 1000, rrow.quality), xytext=(6, 6), textcoords="offset points")
pareto = points.sort_values("cost_per_query"); front, best_q = [], -1
for idx, rrow in pareto.iterrows():
    if rrow.quality > best_q: front.append(idx); best_q = rrow.quality
axes[1].plot(points.loc[front].cost_per_query * 1000, points.loc[front].quality, "--", color="grey", label="Pareto frontier")
axes[1].set_xscale("log"); axes[1].set_xlabel("$ per 1,000 queries (log)"); axes[1].set_ylabel("Synthesis quality (1-5)"); axes[1].legend(); axes[1].set_title("Fig 7.2 – Quality vs cost Pareto")
plt.tight_layout(); save(fig, "fig7_dashboard"); plt.show()
print(f"Routed policy: quality {routed_q:.2f} at ${routed_c*1000:.3f}/1k queries vs dense-only {tm.loc['tier_dense','quality']:.2f} at ${tm.loc['tier_dense','cost_per_query']*1000:.3f}/1k "
      f"({100*(1-routed_c/tm.loc['tier_dense','cost_per_query']):.0f}% cheaper). Router sent {100*share_simple:.0f}% of NQ queries down the simple path.")
RESULTS["task7"]["tiers"] = tm.round(5).to_dict(); RESULTS["task7"]["routed"] = {"quality": routed_q, "cost": routed_c}
save_results()
"""),
md(r"""
### Model comparison summary & final recommendation
<!--T7_RECOMMENDATION-->

### Inference — Task 7
<!--T7_INFERENCE-->
"""),
]
