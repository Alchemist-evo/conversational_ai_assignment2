from common import code, md, module

CELLS = [
md(r"""
---
# Module 4 · Task 6 — Full System Integration, Multi-Turn Conversation, Research Audit Trail & Evaluation (1.5 marks)

`ResearchAssistant` combines every component built so far:
| Stage | Component | From |
|---|---|---|
| Input security | Injection detection; Pydantic-validated tools; RBAC; PII output filter | Task 2 |
| Routing | LLM router: new research / follow-up / reformulation + simple/complex (Adaptive-RAG idea) | Task 3 |
| Prompting | CoT decomposition and refinement prompts; evidence-weighting synthesis instructions | Tasks 1, 3 |
| Retrieval | Parallel multi-source tool calls with fallbacks → credibility-weighted RRF → cross-encoder | Tasks 2, 3, 4 |
| Memory | Episodic top-2 recall; working memory + rolling summary; episodic **write-back** after every query | Task 3 |
| Context | Structured context: episodic, history, working memory, evidence; query last | Task 3 |
| Synthesis | Conflict detection & resolution → cited synthesis → verification → self-correction → confidence → APA citations | Task 4 |
| Audit | JSON audit record per query, rendered as a report; claim-to-source traceability | Task 6 |

The configuration choices (re-ranking on, credibility weights, ≤ 3 refinement iterations, verification on) come from the Task 3 and Task 4 experiments and are justified in the Task 7 decision table.
"""),
module("assistant.py"),
code(r"""
from ra.assistant import ResearchAssistant, AuditRecord
def make_assistant():
    return ResearchAssistant(llm, D, R, EpisodicMemory(D.sessions), ToolRegistry(D, R, role="researcher", failure_rates={"search_news": 0.15}),
                             rerank=True, refine_iters=3, verify=True, weights="credibility")
RA = make_assistant()
print("ResearchAssistant ready:", {k: getattr(RA, k) for k in ("rerank", "refine_iters", "verify", "top_evidence")})
"""),
md(r"""
### 6.1 Multi-turn research conversation (4 turns)
**Metrics**
* **Context Retention (1–5):** the judge's score for whether the response correctly carries over the entities and topic from earlier turns (e.g. what "that" refers to).
* **Follow-up Accuracy (1–5):** the judge's score for whether the follow-up itself is answered correctly and completely.
* **Source Freshness:** the mean age (years, relative to 2026-11) of the sources cited in the turn, and the share cited from the last 5 years.
"""),
code(r"""
from ra.credibility import _age_years
turns = ["What causes inflation?", "Can you explain that in simpler terms?", "What do critics say about that explanation?", "Compare that to the monetary theory view."]
MT = make_assistant()
mt_rows, mt_outs = [], []
for i, msg in enumerate(turns, 1):
    out = MT.chat(msg); mt_outs.append(out)
    ev = out["evidence"]; used = {s for s in re.findall(r"S\d+", out["answer"])}
    ages = [_age_years(e["meta"].get("date", "2015-01-01")) for e in ev if e["sid"] in used] or [np.nan]
    hist = "\n".join(f"User: {t['user']}\nAssistant: {t['assistant'][:500]}" for t in MT.history[:-1]) or "(first turn)"
    data, _ = llm.chat_json(f"Conversation so far:\n{hist}\n\nCurrent user message: {msg}\nAssistant response:\n{out['answer']}\n\n"
                            "Rate 1-5: context_retention (does the response correctly use and carry over the earlier context - entities, topic, "
                            "what pronouns refer to; 5 = perfectly; for the first turn rate whether it stays on the asked topic) and followup_accuracy "
                            "(does it correctly and completely answer the current message; 5 = fully). Return JSON: {\"context_retention\": int, "
                            "\"followup_accuracy\": int, \"rationale\": str}", default={}, role="judge", max_tokens=200, purpose="t6_multiturn_judge")
    mt_rows.append({"turn": i, "user": msg, "route": out["route"]["type"], "standalone": out["route"]["standalone_query"][:70],
                    "context_retention": data.get("context_retention"), "followup_accuracy": data.get("followup_accuracy"),
                    "freshness_mean_age_y": np.nanmean(ages), "share_last_5y": np.nanmean([a <= 5 for a in ages]) if not np.isnan(ages[0]) else np.nan,
                    "latency_s": out["metrics"]["latency_s"]})
    print(f"--- Turn {i}: {msg}  [{out['route']['type']}]\n{textwrap.fill(out['answer'], 120)}\n")
mt = pd.DataFrame(mt_rows)
display(Markdown("**Table 6.1 – Multi-turn evaluation.**")); display(mt.round(2))
RESULTS["task6"] = {"multiturn": mt[["turn", "route", "context_retention", "followup_accuracy", "freshness_mean_age_y"]].to_dict("records")}
"""),
code(r"""
N = {}; E = []
for i, rrow in enumerate(mt.itertuples()):
    x = 1.6 + i * 3.0
    N[f"u{i}"] = (x, 6.6, f"Turn {rrow.turn} user:\n" + textwrap.fill(rrow.user, 24), PALETTE[0], 2.6, 1.0)
    N[f"r{i}"] = (x, 4.8, f"Router: {rrow.route}\n" + textwrap.fill(rrow.standalone[:48], 26), PALETTE[4], 2.7, 1.2)
    N[f"a{i}"] = (x, 2.8, ("No retrieval:\nrewrite previous answer" if rrow.route == "reformulate" else "Decompose → parallel tools\n→ fuse → synthesise → verify"), PALETTE[1] if rrow.route != "reformulate" else PALETTE[5], 2.7, 1.0)
    N[f"m{i}"] = (x, 1.1, f"Retention {rrow.context_retention}/5\nFollow-up acc. {rrow.followup_accuracy}/5", PALETTE[2], 2.4, 0.9)
    E += [(f"u{i}", f"r{i}", ""), (f"r{i}", f"a{i}", ""), (f"a{i}", f"m{i}", "")]
    if i: E.append((f"m{i-1}", f"u{i}", "history + memory", "--"))
fig, ax = viz.diagram(N, E, "Figure 6.1 – Multi-turn research conversation flow", figsize=(14, 7.5), xlim=(0, 12.2), ylim=(0.3, 7.4))
save(fig, "fig6_1_multiturn_flow"); plt.show()
fig, ax = plt.subplots(figsize=(8, 3.4)); x = np.arange(len(mt))
ax.bar(x - 0.2, mt.context_retention.astype(float), 0.4, label="Context retention", color=PALETTE[0]); ax.bar(x + 0.2, mt.followup_accuracy.astype(float), 0.4, label="Follow-up accuracy", color=PALETTE[2])
bar_labels(ax, "{:.0f}"); ax.set_xticks(x, [f"Turn {t}" for t in mt.turn]); ax.set_ylim(0, 5.8); ax.legend(); ax.set_title("Fig 6.2 – Context retention vs turn number")
save(fig, "fig6_2_retention"); plt.show()
"""),
md(r"""
### 6.2 Research audit trail
Each query produces a structured `AuditRecord`, stored as JSON in `outputs/audit/`. It contains:
* the original question and route
* the static sub-queries and every refinement iteration
* every tool call: name, arguments, timestamp, latency and output size
* every source: title, URL and credibility
* every claim: evidence snippet, supporting source, citation, verdict and confidence
* the final synthesis and overall confidence

**Claim-to-source traceability** is the percentage of final claims that are verified SUPPORTED *and* have at least one supporting source with a URL and an evidence snippet.
"""),
code(r"""
AT = make_assistant()
aq = next(q for q in heldout if q["qid"] == "H7-Q1")
out = AT.research(aq["question"], mode="full")
audit = out["audit"]
os.makedirs("outputs/audit", exist_ok=True)
audit_path = "outputs/audit/" + aq["qid"] + ".json"
json.dump(audit.to_dict(), open(audit_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(f"Audit JSON written to {audit_path} ({os.path.getsize(audit_path)/1024:.1f} KB)")
display(Markdown(audit.render_markdown()))
"""),
code(r"""
# Audit trail visualisation: tool-call timeline + claim -> source evidence graph
import networkx as nx
fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1.2]})
tc = pd.DataFrame(audit.tool_calls)
colors = {"search_academic_db": PALETTE[0], "search_wikipedia": PALETTE[1], "search_news": PALETTE[2], "search_web": PALETTE[3], "check_source_credibility": PALETTE[4], "generate_citation": PALETTE[5]}
for i, rrow in enumerate(tc.itertuples()):
    axes[0].barh(i, rrow.latency_s, color=colors.get(rrow.tool, "grey"), edgecolor="k" if not rrow.ok else None)
axes[0].set_yticks(range(len(tc)), [f"{t[:22]}" for t in tc.tool], fontsize=6); axes[0].invert_yaxis(); axes[0].set_xlabel("latency (s)")
axes[0].set_title(f"Fig 6.3a – Audit trail: {len(tc)} tool calls (black edge = failed)")
G = nx.DiGraph()
for i, c in enumerate(audit.claims[:10], 1):
    G.add_node(f"C{i}", kind="claim", conf=c["confidence"])
    for s in c["sources"]:
        G.add_node(s["sid"], kind="source"); G.add_edge(f"C{i}", s["sid"])
pos = {}
claims_ = [n for n, d in G.nodes(data=True) if d["kind"] == "claim"]; srcs_ = [n for n, d in G.nodes(data=True) if d["kind"] == "source"]
for i, n in enumerate(claims_): pos[n] = (0, -i)
for i, n in enumerate(srcs_): pos[n] = (1, -i * max(1, len(claims_)) / max(1, len(srcs_)))
nx.draw_networkx_nodes(G, pos, nodelist=claims_, node_color=[plt.cm.RdYlGn(G.nodes[n]["conf"]) for n in claims_], node_size=600, ax=axes[1])
nx.draw_networkx_nodes(G, pos, nodelist=srcs_, node_color=PALETTE[0], node_shape="s", node_size=600, ax=axes[1])
nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=axes[1]); nx.draw_networkx_edges(G, pos, ax=axes[1], arrows=True, alpha=0.6)
src_title = {s["sid"]: s["title"][:45] for s in audit.sources}
for n in srcs_: axes[1].text(1.08, pos[n][1], src_title.get(n, ""), fontsize=7, va="center")
axes[1].set_xlim(-0.3, 2.2); axes[1].axis("off"); axes[1].set_title(f"Fig 6.3b – Claim → source evidence trail (colour = confidence), traceability {100*audit.traceability:.0f}%")
plt.tight_layout(); save(fig, "fig6_3_audit"); plt.show()
"""),
md(r"""
### 6.3 Evaluation on 10 complex held-out research queries
Every query goes through `ResearchAssistant.research(mode="full")`. Queries are grouped by **source profile** (single-source: answerable from one collection; multi-source) and **nature** (factual vs contested). The metric definitions are those from Task 4. The Conflict Resolution Rate uses the assistant's credibility-weighted decisions on the conflicts it detected.
"""),
code(r"""
EV = make_assistant()
ev_rows, ev_cal = [], []
for i, q in enumerate(heldout, 1):
    out = EV.research(q["question"], mode="full")
    a, res, evd = out["audit"], out["result"], {e["sid"]: e for e in out["evidence"]}
    qual = judge_synthesis_quality(llm, q, out["answer"], out["evidence"], purpose="t6_eval_quality")
    tri_s = rag_triad(llm, q["question"], out["evidence"], out["answer"], purpose="t6_eval_triad")
    pairs = [(c["text"], evd[s]["text"]) for c in res.claims_after for s in c["cited"] if s in evd]
    cit_ok = judge_citation_support(llm, pairs, purpose="t6_eval_citation")
    ref_docs = [{"text": d["text"], "source": d["source"], "venue": d["venue"], "year": d["year"], "title": d["title"]} for d in D.topic_docs(q["topic_id"]) if credibility(d)["score"] >= 0.5]
    labels = judge_claims_correct(llm, q, [c["text"] for c in res.claims_after], ref_docs, purpose="t6_eval_claims")
    ev_cal += [{"qid": q["qid"], "conf": c["confidence"], "correct": l} for c, l in zip(res.claims_after, labels)]
    resolved, agree = [], []
    for c in res.conflicts:
        gold, mapping = match_gold_conflict(c, q["conflicts"])
        dec = c["resolution"]["decision"]
        if dec == "contested":
            resolved.append(sum(any(evd[s]["credibility"] >= 0.5 for s in p["sids"]) for p in c["positions"]) >= 2); pred = "contested"
        else:
            resolved.append(np.mean([evd[s]["credibility"] for s in c["resolution"]["supporting_sids"]]) >= 0.5); pred = mapping.get(dec, -1)
        if gold: agree.append(pred == gold["gold"])
    ev_rows.append({"qid": q["qid"], "profile": q["source_profile"], "nature": q["nature"], "quality": qual["overall"],
                    "conflicts_found": len(res.conflicts), "conflicts_resolved": int(np.sum(resolved)), "gold_matched": len(agree), "gold_agree": int(np.sum(agree)),
                    "citation_pairs": len(cit_ok), "citations_accurate": int(np.sum(cit_ok)), "halluc_before": res.halluc_rate_before, "halluc_after": res.halluc_rate_after,
                    "context_relevance": tri_s["context_relevance"], "groundedness": tri_s["groundedness"], "answer_relevance": tri_s["answer_relevance"],
                    "traceability": a.traceability, "latency_s": a.metrics["latency_s"], "input_tokens": a.metrics["input_tokens"],
                    "output_tokens": a.metrics["output_tokens"], "cost_usd": a.metrics["cost_usd"], "llm_calls": a.metrics["llm_calls"],
                    "tool_calls": a.metrics["n_tool_calls"], "completed": bool(out["answer"]) and a.traceability > 0})
    json.dump(a.to_dict(), open(f"outputs/audit/{q['qid']}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\r{i}/{len(heldout)} evaluated", end="")
evd_df = pd.DataFrame(ev_rows); evcal = pd.DataFrame(ev_cal)
ece6, _ = expected_calibration_error(evcal.conf, evcal.correct, 5)
def agg(g):
    return pd.Series({"n": len(g), "synthesis_quality": g.quality.mean(),
                      "conflict_resolution_rate_%": 100 * g.conflicts_resolved.sum() / max(1, g.conflicts_found.sum()),
                      "gold_agreement_%": 100 * g.gold_agree.sum() / max(1, g.gold_matched.sum()),
                      "citation_accuracy_%": 100 * g.citations_accurate.sum() / max(1, g.citation_pairs.sum()),
                      "halluc_before_%": 100 * g.halluc_before.mean(), "halluc_after_%": 100 * g.halluc_after.mean(),
                      "context_rel": g.context_relevance.mean(), "groundedness": g.groundedness.mean(), "answer_rel": g.answer_relevance.mean(),
                      "traceability_%": 100 * g.traceability.mean(), "latency_P50_s": g.latency_s.median(), "tokens_in": g.input_tokens.mean(),
                      "tokens_out": g.output_tokens.mean(), "cost_per_query_$": g.cost_usd.mean()})
overall = agg(evd_df).rename("ALL (10)")
groups = pd.concat([overall.to_frame().T, evd_df.groupby("profile").apply(agg).rename(index=lambda x: f"{x}-source"), evd_df.groupby("nature").apply(agg)])
print(); display(Markdown("**Table 6.2 – Evaluation of the integrated assistant on 10 complex held-out queries.**")); display(groups.round(3))
from ra.evaluation import percentiles
lat_p = percentiles(evd_df.latency_s)
print(f"ECE (claim-level, {len(evcal)} claims): {ece6:.3f} | latency P50 {lat_p['P50']:.1f}s, P95 {lat_p['P95']:.1f}s | task completion {100*evd_df.completed.mean():.0f}%")
RESULTS["task6"].update({"eval10": overall.round(4).to_dict(), "ece": ece6, "latency": lat_p, "completion": float(evd_df.completed.mean())})
"""),
md(r"""
### 6.4 End-to-end evaluation on the 50 held-out NQ questions
The 50 NQ `heldout` questions were never used for tuning or memory. They run through `research(mode="auto")`, where the **router decides** between the fast single-hop path and the full multi-hop path (Adaptive RAG).
* **Accuracy** is the gold short answer in the response, by normalised string match, or judged equivalent when the string match fails.
* **Task completion** is a non-blocked answer containing at least one verified, cited claim.
"""),
code(r"""
NQA = make_assistant()
nq_rows = []
for i, r in enumerate(D.nq_split("heldout"), 1):
    q = D.nq_as_question(r)
    out = NQA.research(q["question"], mode="auto")
    a = out["audit"]
    ok = gold_match(out["answer"], r["answers"])
    if not ok:
        ok = judge_correctness(llm, q, out["answer"], purpose="t6_nq_judge")["correct"]
    nq_rows.append({"qid": q["qid"], "route_complexity": a.route.get("complexity"), "correct": ok, "completed": (not out["blocked"]) and a.traceability > 0,
                    "recall_gold_doc": float(r["doc_id"] in {e["doc_id"] for e in out["evidence"]}), "latency_s": a.metrics["latency_s"],
                    "cost_usd": a.metrics["cost_usd"], "input_tokens": a.metrics["input_tokens"], "output_tokens": a.metrics["output_tokens"],
                    "halluc_after": a.metrics["halluc_after"]})
    print(f"\r{i}/50", end="")
nqd = pd.DataFrame(nq_rows)
lp = percentiles(nqd.latency_s)
nq_summary = {"accuracy_%": 100 * nqd.correct.mean(), "task_completion_%": 100 * nqd.completed.mean(), "gold_passage_in_evidence_%": 100 * nqd.recall_gold_doc.mean(),
              "hallucination_after_%": 100 * nqd.halluc_after.mean(), "latency_P50_s": lp["P50"], "latency_P95_s": lp["P95"],
              "cost_per_query_$": nqd.cost_usd.mean(), "routed_simple_%": 100 * (nqd.route_complexity == "simple").mean()}
print(); display(pd.Series(nq_summary).round(3).to_frame("50 held-out NQ questions"))
RESULTS["task6"]["nq50"] = nq_summary
"""),
md(r"""
### 6.5 Cost analysis (GPT-4o pricing: $5 / 1M input, $15 / 1M output)
Measured token usage for each of the 10 complex queries is priced at the assignment's GPT-4o rates. The generator actually used was gemini-3.1-flash-lite (free tier); the GPT-4o price is applied for comparability, as the assignment requires.
"""),
code(r"""
evd_df["cost_gpt4o"] = [config.cost_usd(a, b) for a, b in zip(evd_df.input_tokens, evd_df.output_tokens)]
fig, axes = plt.subplots(1, 2, figsize=(15, 3.8))
axes[0].plot(range(1, len(evd_df) + 1), evd_df.cost_gpt4o * 100, marker="o", color=PALETTE[0], label="per query"); axes[0].axhline(evd_df.cost_gpt4o.mean() * 100, ls="--", color=PALETTE[3], label="mean")
axes[0].set_xticks(range(1, len(evd_df) + 1), evd_df.qid, rotation=30, fontsize=8); axes[0].set_ylabel("cost (US cents)"); axes[0].legend(); axes[0].set_title("Fig 6.4 – Cost per query over 10 complex queries")
axes[1].bar(evd_df.qid, evd_df.input_tokens, label="input", color=PALETTE[0]); axes[1].bar(evd_df.qid, evd_df.output_tokens, bottom=evd_df.input_tokens, label="output", color=PALETTE[1])
axes[1].set_ylabel("tokens"); axes[1].tick_params(axis="x", rotation=30, labelsize=8); axes[1].legend(); axes[1].set_title("Fig 6.5 – Tokens per query")
plt.tight_layout(); save(fig, "fig6_cost"); plt.show()
print(f"Mean cost per complex research query: ${evd_df.cost_gpt4o.mean():.4f} ({evd_df.input_tokens.mean():.0f} in / {evd_df.output_tokens.mean():.0f} out tokens, "
      f"{evd_df.llm_calls.mean():.1f} LLM calls) | per simple NQ query: ${nqd.cost_usd.mean():.4f}")
calls_df = pd.DataFrame([c.__dict__ for c in EV.llm.calls])
RESULTS["task6"]["cost_per_complex_query"] = float(evd_df.cost_gpt4o.mean())
"""),
md(r"""
**Three cost optimisation strategies (quantified in Task 7):**
1. **Model routing.** Our router already classifies queries (simple vs complex). Sending simple look-ups and helper calls (routing, decomposition, verification) to a small model, and only final contested syntheses to the large model, cuts per-query cost by about 90 % under the Task 7 price sheet.
2. **Prompt caching.** The system prompt, tool schemas and synthesis instructions are an identical prefix on every call. With provider prefix caching (50–90 % discount on cached tokens) this prefix becomes almost free.
3. **Context reduction.** Task 3's classification shows which context is *potentially redundant* (low-similarity episodic sessions and history turns). Dropping it, and capping evidence at about 140 words per source, reduces input tokens with no measured quality loss.

### 6.6 Production architecture
"""),
code(r"""
N = {"user": (1.0, 7.2, "User / UI\n(chat, API)", PALETTE[5], 1.8), "gw": (3.4, 7.2, "API gateway\nauth · RBAC · rate limit", PALETTE[3], 2.4),
     "orch": (6.4, 7.2, "Orchestrator\n(ResearchAssistant, ReAct)", PALETTE[0], 2.8), "cache": (9.6, 7.2, "Caching layer\nprompt / response / embedding", PALETTE[6], 2.8),
     "router": (6.4, 5.5, "Model router\nsmall | MoE | dense", PALETTE[4], 2.6), "llm": (9.6, 5.5, "LLM endpoints\n(Gemini / GPT-4o)", PALETTE[0], 2.6),
     "apis": (1.4, 5.5, "Source APIs\nacademic · wiki · news · web", PALETTE[1], 2.6), "vdb": (3.9, 5.5, "Vector DB (FAISS/HNSW)\n+ BM25 index", PALETTE[1], 2.4),
     "conf": (2.2, 3.6, "Conflict resolver\n(credibility-weighted)", PALETTE[3], 2.6), "cit": (5.2, 3.6, "Citation generator\n(APA / MLA)", PALETTE[2], 2.4),
     "ver": (8.2, 3.6, "Verifier +\nconfidence calibrator", PALETTE[3], 2.4), "mem": (2.2, 1.6, "Memory store\nworking · episodic (vector)", PALETTE[2], 2.8),
     "audit": (5.6, 1.6, "Audit log store\n(JSON, immutable)", PALETTE[5], 2.4), "mon": (9.0, 1.6, "Monitoring & alerts\n(metrics, judge sampling)", PALETTE[5], 2.8)}
E = [("user", "gw", ""), ("gw", "orch", ""), ("orch", "cache", "lookup"), ("orch", "router", ""), ("router", "llm", ""), ("cache", "llm", "miss"),
     ("orch", "apis", "parallel tools"), ("orch", "vdb", "hybrid search"), ("orch", "conf", ""), ("conf", "cit", ""), ("cit", "ver", ""), ("orch", "mem", "recall / update"),
     ("orch", "audit", "trace"), ("audit", "mon", "")]
fig, ax = viz.diagram(N, E, "Figure 6.6 – Production architecture", figsize=(14, 8), xlim=(0, 11.2), ylim=(0.6, 8))
save(fig, "fig6_6_architecture"); plt.show()
"""),
md(r"""
### 6.7 Monitoring & alerts
| # | Metric | How measured in production | Alert threshold |
|---|---|---|---|
| 1 | **Hallucination rate (post-verification)** | Share of claims the verifier flags, logged per query | > 10 % over a rolling 1-hour window |
| 2 | **Latency P95** | End-to-end time from the audit log | > 1.5× the baseline P95 measured in §6.3, sustained 15 min |
| 3 | **Tool failure / fallback rate** | Failed tool calls ÷ all tool calls (audit log) | > 10 % for any source over 5 min (circuit-breaker storm) |
| 4 | **Groundedness (sampled LLM judge)** | RAG-Triad judge on a 2 % sample | Daily mean < 3.5, or drop > 0.5 vs the 7-day mean |
| 5 | **Cost per query** | Token usage × price sheet | > 1.5× budget (e.g. > $0.05 for a complex query) for 1 hour |
| + | Security blocks | Pydantic / RBAC / sanitiser events | Spike > 5× baseline (possible attack campaign) |

### 6.8 Limitations & future work
| # | Limitation (observed) | Proposed improvement |
|---|---|---|
| 1 | Most of the multi-source corpus is synthetic and small (75 documents), so retrieval is easier than on the open web | Connect real APIs (Semantic Scholar, PubMed, news APIs) and re-run the same evaluation harness |
| 2 | A single LLM judge (gpt-oss-120b) scores everything and may have its own biases; κ needs human labels; κ needs human labels | More human labels, a judge from a different model family, judge ensembles |
| 3 | The verifier is the same small generator model, which misses subtle numeric hallucinations (see the Task 4 confusion matrix) | A dedicated NLI verifier (e.g. DeBERTa-MNLI) or a stronger model for verification only |
| 4 | Conflict detection relies on one LLM pass, so missed conflicts are invisible to the resolver | Claim-level clustering (embed claims, then pairwise NLI contradiction) as a second detector |
| 5 | Latency: complex queries need about 10 sequential LLM calls | Parallelise conflict detection with synthesis, cache decompositions, route helper calls to small models, stream answers |

### Inference — Task 6
<!--T6_INFERENCE-->
"""),
]
