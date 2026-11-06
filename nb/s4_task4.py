from common import code, md, module

CELLS = [
md(r"""
---
# Module 2 · Task 4 — Multi-Source Synthesis, Conflict Resolution, Hallucination Detection, Confidence Calibration & RAG Triad (1.5 marks)

**Objective.** Synthesise conflicting multi-source evidence into an answer with accurate citations, detect and correct hallucinations, calibrate confidence, and evaluate the system with the RAG Triad.

**Pipeline (`ra/synthesis.py`):**
1. evidence (deduplicated, with credibility scores)
2. LLM **conflict detection**: issues and competing positions, with source ids
3. **conflict resolution** in code, using majority vote, credibility weighting or evidence strength
4. **cited synthesis** that includes the conflict analysis
5. **claim splitting**
6. **verification pass**: each claim is checked as SUPPORTED, UNSUPPORTED or CONTRADICTED with its evidence ids
7. **self-correction** of flagged claims using only supported evidence
8. re-verification
9. per-claim **confidence**
10. APA **citations**
"""),
module("synthesis.py"),
md(r"""
### 4.1 Source ranking: credibility weights
`credibility = 0.50·authority + 0.35·evidence_quality + 0.15·recency`

* **Authority (0.50)** reflects the publisher's accountability and editorial review: peer-reviewed journal 0.90 > encyclopedia 0.70 > reputable news 0.55 > opinion 0.30 > industry PR 0.20 > blog / tabloid 0.15 > forum 0.10. It gets the largest weight because it is the hardest property for a bad source to fake.
* **Evidence quality (0.35)** follows the study-design hierarchy of evidence-based medicine: meta-analysis / systematic review 1.0 > RCT 0.9 > cohort / observational 0.65–0.7 > narrative review 0.75 > news report 0.4 > opinion 0.15 > anecdote 0.05.
* **Recency (0.15)** is `0.5 + 0.5·2^(−age/half-life)`, with a 10-year half-life for scholarly sources and 4 years for news. It has the smallest weight on purpose: a 2017 umbrella review should still beat a 2024 tabloid. For time-sensitive questions, such as "how high did inflation get", recency matters more and the weight could be raised.

These categories are **examples, not fixed rules**. A reputable-news report of a preliminary study (D1-N1) correctly scores in the middle. Relevance to the query is not part of the static score; it enters through the retrieval rank inside weighted RRF.
"""),
code(r"""
from ra.credibility import credibility
from ra.synthesis import (weighted_rrf, SOURCE_WEIGHT_PRESETS, detect_conflicts, resolve_conflict, match_gold_conflict, docs_to_evidence,
                          synthesize, split_claims, verify_claims, self_correct, claim_confidence, full_synthesis, citations_for)
rep = ["D1-A1", "D1-A2", "D5-A3", "WIKI-Intermittent_fasting", "D1-N1", "H9-N1", "D7-N3", "D1-N2", "D1-N3", "D1-F1"]
cred_tab = pd.DataFrame([{"doc": d, "type": f"{D.docs[d]['source']}/{D.docs[d]['outlet_type']}", "study_type": D.docs[d]["study_type"], "date": D.docs[d]["date"],
                          **{k: v for k, v in credibility(D.docs[d]).items() if k in ("authority", "evidence_quality", "recency", "score")}} for d in rep if d in D.docs])
display(Markdown("**Table 4.1 – Credibility components for representative sources.**")); display(cred_tab)
"""),
md(r"""
### 4.2 Weighted fusion: experimenting with source-weight combinations
For each dev question we retrieve the **top-3 snippets per source** and fuse them with **weighted RRF**: `score = w_source × 1/(60 + rank)`. The `+doc-cred` variants also multiply by each document's own credibility, so a tabloid inside the "news" source is down-weighted relative to a reputable outlet. We report the precision and coverage of the fused top-5 against the gold support documents, plus the share of low-credibility documents (credibility < 0.35) that reach the top-5.
"""),
code(r"""
fus_rows = []
for q in dev_cq:
    per = R.search(q["question"], ("academic", "wikipedia", "news"), 3)
    rel = D.relevant_docs(q)
    for preset, w in SOURCE_WEIGHT_PRESETS.items():
        for dc in (False, True):
            top = [h for h, _ in weighted_rrf(per, w, doc_level_credibility=dc)][:5]
            ids = [h.doc_id for h in top]
            fus_rows.append({"weights": preset + (" +doc-cred" if dc else ""), "precision@5": np.mean([i in rel for i in ids]),
                             "coverage@5": fact_coverage(ids, q), "low_cred_share@5": np.mean([credibility(h.chunk.meta)["score"] < 0.35 for h in top])})
fus = pd.DataFrame(fus_rows).groupby("weights").mean().sort_values("precision@5", ascending=False)
display(Markdown("**Table 4.2 – Weighted RRF with different weight combinations (8 dev questions).**")); display(fus.round(3))
RESULTS["task4"] = {"fusion": fus.round(4).to_dict()}
"""),
md(r"""
### 4.3 Conflict detection and resolution
**Strategies (implemented in code, applied to the positions the LLM detects):**
* **majority vote:** each source counts as one vote.
* **credibility-weighted:** each source votes with its credibility score.
* **evidence-strength:** each source votes with authority × evidence quality × recency.

A position wins if it gets ≥ 60 % of the total score. Otherwise the conflict is declared **"contested"**, which the synthesis must then present as genuinely open.

**Metric definitions**
* **Conflict Resolution Rate (CRR)** is the percentage of *identified* conflicts for which the system reaches a final determination supported by credible selected evidence. That means either (i) a winning position whose supporting sources average credibility ≥ 0.5, or (ii) an explicit "contested" verdict where *both* sides are backed by credible sources (credibility ≥ 0.5).
* **Gold agreement** is the percentage of detected conflicts that match an adjudicated gold conflict (by document overlap) *and* whose decision equals the gold label (winning position or "contested").

The evidence for each dev topic is the full set of its curated documents, so every side of each conflict is present.
"""),
code(r"""
def conflict_eval(qs, strategies=("majority", "credibility", "evidence"), weight_override=None, cache={}):
    rows = []
    for q in qs:
        key = q["qid"]
        if key not in cache:
            ev = docs_to_evidence(D.topic_docs(q["topic_id"]))
            cache[key] = (ev, detect_conflicts(llm, q["question"], ev))
        ev, confs = cache[key]
        evd = {e["sid"]: e for e in ev}
        for c in confs:
            gold, mapping = match_gold_conflict(c, q["conflicts"])
            for strat in strategies:
                res = resolve_conflict(c, ev, strat, weight_override=weight_override if strat == "credibility" else None)
                if res["decision"] == "contested":
                    credible_sides = sum(any(evd[s]["credibility"] >= 0.5 for s in p["sids"]) for p in c["positions"])
                    resolved = credible_sides >= 2
                    pred = "contested"
                else:
                    resolved = np.mean([evd[s]["credibility"] for s in res["supporting_sids"]]) >= 0.5
                    pred = mapping.get(res["decision"], -1) if gold else None
                agree = (gold is not None) and (pred == gold["gold"])
                rows.append({"qid": q["qid"], "issue": c["issue"][:70], "strategy": strat, "matched_gold": gold["cid"] if gold else None,
                             "decision": res["winning_stance"] or "contested", "share": round(res["share"], 2), "resolved": resolved, "gold_agree": agree})
    return pd.DataFrame(rows, columns=["qid", "issue", "strategy", "matched_gold", "decision", "share", "resolved", "gold_agree"])
contested_qs = [q for q in D.questions if q["split"] == "dev" and q["conflicts"]]
contested_qs = list({q["topic_id"]: q for q in contested_qs}.values())   # one question per topic (evidence = whole topic)
cf = conflict_eval(contested_qs)
display(Markdown("**Table 4.3 – Detected conflicts and decisions per strategy (dev topics).**")); display(cf)
crr = cf.groupby("strategy").agg(conflicts=("issue", "size"), CRR_pct=("resolved", lambda s: 100 * s.mean()),
                                 gold_agreement_pct=("gold_agree", lambda s: 100 * s.mean()),
                                 matched=("matched_gold", lambda s: s.notna().sum()))
display(Markdown("**Table 4.4 – Conflict Resolution Rate and gold agreement by strategy.**")); display(crr.round(1))
RESULTS["task4"]["conflicts_dev"] = crr.round(3).to_dict()
"""),
md(r"""
**Example where conflict resolution changes the final answer.** For intermittent fasting, three low-credibility sources (tabloid, blog, forum) claim IF is *superior* to calorie restriction, while two peer-reviewed trials find it *equivalent*. Majority vote follows the crowd; credibility weighting follows the evidence. We synthesise the answer under each resolution.
"""),
code(r"""
qf = next(q for q in D.questions if q["qid"] == "D1-Q1")
ev_f = docs_to_evidence(D.topic_docs("D1"))
gold_c = qf["conflicts"][0]
sid_of = {e["doc_id"]: e["sid"] for e in ev_f}
c_manual = {"issue": gold_c["issue"], "positions": [{"stance": p["stance"], "sids": [sid_of[d] for d in p["docs"]], "docs": p["docs"]} for p in gold_c["positions"]]}
outs = {}
for strat in ("majority", "credibility"):
    res = resolve_conflict(c_manual, ev_f, strat)
    c2 = {**c_manual, "resolution": res}
    ans, _ = synthesize(llm, "Is intermittent fasting more effective than continuous calorie restriction for weight loss?", ev_f, [c2], purpose="t4_conflict_example")
    outs[strat] = ans
    print(f"=== {strat.upper()} -> decision: {res['winning_stance'] or 'contested'} (scores {np.round(res['scores'], 2)}, share {res['share']:.2f})")
    print(textwrap.fill(ans, 120), "\n")
"""),
code(r"""
# Conflict Resolution Rate vs source-credibility weight (multiplier on the academic source's weight)
mults = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
curve_rows = []
for m in mults:
    d = conflict_eval(contested_qs, strategies=("credibility",), weight_override={"academic": m})
    curve_rows.append({"academic_weight_multiplier": m, "CRR_pct": 100 * d.resolved.mean(), "gold_agreement_pct": 100 * d.gold_agree.mean()})
crr_curve = pd.DataFrame(curve_rows)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(crr_curve.academic_weight_multiplier, crr_curve.CRR_pct, marker="o", label="Conflict Resolution Rate", color=PALETTE[0])
ax.plot(crr_curve.academic_weight_multiplier, crr_curve.gold_agreement_pct, marker="s", label="Agreement with gold", color=PALETTE[2])
ax.set_xlabel("Weight multiplier on academic sources"); ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend(); ax.set_title("Fig 4.1 – Conflict Resolution Rate vs source credibility weight")
save(fig, "fig4_1_crr_weight"); plt.show(); display(crr_curve.round(1))
"""),
md(r"""
### 4.4 Full pipeline on the 10 held-out research questions
For each held-out question:
1. CoT decomposition
2. multi-source hybrid retrieval per sub-query
3. credibility-weighted RRF and cross-encoder re-ranking, keeping the top-8 evidence documents
4. `full_synthesis`, which runs conflict detection → resolution → cited synthesis → verification → self-correction → confidence

We then measure:
* **Citation accuracy:** the judge checks each (claim, cited source) pair to see whether the source *directly* supports the claim.
* **Hallucination rate:** the share of claims the verifier flags as not supported by the retrieved evidence, before and after self-correction.
* **Calibration:** the judge labels each final claim as correct or incorrect against the gold key facts and the topic's credible documents (ground truth). This gives a reliability diagram and the ECE.
* **Synthesis quality:** the 1–5 rubric.
"""),
code(r"""
heldout = D.by_use("heldout")
def heldout_evidence(q, sources=("academic", "wikipedia", "news"), n=8):
    subs, _ = decompose(llm, q["question"])
    per = {s: [] for s in sources}
    for sq in subs:
        for s, hl in R.search(sq, sources, 5).items(): per[s] += hl
    for s in per:
        b = {}
        for h in per[s]:
            if h.doc_id not in b or h.rank < b[h.doc_id].rank: b[h.doc_id] = h
        per[s] = sorted(b.values(), key=lambda h: h.rank)
    fused = [h for h, _ in weighted_rrf(per, SOURCE_WEIGHT_PRESETS["credibility"], doc_level_credibility=True)]
    return hits_to_evidence(ce.rerank(q["question"], fused[:20])[:n], 140), subs
T4 = {}
for i, q in enumerate(heldout, 1):
    ev, subs = heldout_evidence(q)
    res = full_synthesis(llm, q["question"], ev, resolve=True, verify=True)
    T4[q["qid"]] = {"q": q, "ev": ev, "res": res, "subs": subs}
    print(f"\r{i}/{len(heldout)} held-out questions synthesised", end="")
q_show = heldout[0]["qid"]; r0 = T4[q_show]["res"]
print(f"\n\nExample ({q_show}): {T4[q_show]['q']['question']}\n--- draft ---\n{textwrap.fill(r0.draft, 120)}\n--- after verification + self-correction ---\n{textwrap.fill(r0.answer, 120)}")
print("\nClaims with confidence and APA citations:")
for c in r0.claims_after[:5]:
    print(f" [{c.get('verdict','?')}, conf={c['confidence']:.2f}, n_src={c['n_sources']}] {c['text'][:110]}")
    for cit in c["citations"][:2]: print("      ->", cit)
"""),
code(r"""
from ra.evaluation import judge_citation_support, judge_claims_correct, expected_calibration_error
cal_rows, cit_rows = [], []
for qid, t in T4.items():
    q, ev, res = t["q"], t["ev"], t["res"]
    evd = {e["sid"]: e for e in ev}
    pairs, meta = [], []
    for c in res.claims_after:
        for s in c["cited"]:
            if s in evd: pairs.append((c["text"], evd[s]["text"])); meta.append(s)
    sup = judge_citation_support(llm, pairs, purpose="t4_citation_judge")
    cit_rows += [{"qid": qid, "accurate": x} for x in sup]
    ref_docs = [{"text": d["text"], "source": d["source"], "venue": d["venue"], "year": d["year"], "title": d["title"]}
                for d in D.topic_docs(q["topic_id"]) if credibility(d)["score"] >= 0.5]
    labels = judge_claims_correct(llm, q, [c["text"] for c in res.claims_after], ref_docs, purpose="t4_claim_truth")
    for c, lab in zip(res.claims_after, labels):
        cal_rows.append({"qid": qid, "nature": q["nature"], "claim": c["text"][:90], "confidence": c["confidence"], "n_sources": c["n_sources"],
                         "verdict": c.get("verdict"), "correct": lab})
cal = pd.DataFrame(cal_rows); cit = pd.DataFrame(cit_rows)
ece, bins = expected_calibration_error(cal.confidence, cal.correct, 5)
bins = pd.DataFrame(bins)
hb = np.mean([t["res"].halluc_rate_before for t in T4.values()]); ha = np.mean([t["res"].halluc_rate_after for t in T4.values()])
print(f"Citation accuracy: {100*cit.accurate.mean():.1f}% of {len(cit)} claim-citation pairs")
print(f"Hallucination rate: before self-correction {100*hb:.1f}% -> after {100*ha:.1f}%")
print(f"Expected Calibration Error (5 bins): {ece:.3f} | mean confidence {cal.confidence.mean():.2f} vs accuracy {cal.correct.mean():.2f} -> "
      f"{'OVER' if cal.confidence.mean() > cal.correct.mean() else 'UNDER'}-confident on average")
display(Markdown("**Table 4.5 – Calibration bins.**")); display(bins.round(3))
by_nat = cal.groupby("nature").apply(lambda g: pd.Series({"claims": len(g), "mean_conf": g.confidence.mean(), "accuracy": g.correct.mean(),
                                                          "ECE": expected_calibration_error(g.confidence, g.correct, 5)[0]}))
display(Markdown("**Table 4.6 – Calibration by query type.**")); display(by_nat.round(3))
by_ns = cal.groupby("n_sources").agg(claims=("correct", "size"), mean_conf=("confidence", "mean"), accuracy=("correct", "mean"))
display(Markdown("**Table 4.7 – Confidence and correctness by number of supporting sources (claim type).**")); display(by_ns.round(3))
RESULTS["task4"].update({"citation_accuracy": float(cit.accurate.mean()), "halluc_before": float(hb), "halluc_after": float(ha), "ece": float(ece),
                         "mean_conf": float(cal.confidence.mean()), "claim_accuracy": float(cal.correct.mean())})
"""),
md(r"""
#### 4.4.1 Hallucination detector: confusion matrix against constructed ground truth
To test the **detector itself** we need claims whose labels are known. For each held-out question we take 3 gold key facts whose supporting document *is* in the retrieved evidence (label: supported). We add 2 *fabricated* claims (label: hallucinated): one key fact from a different topic, and one gold fact with its numbers altered (×2 + 1). The verifier then labels all of them blind.
"""),
code(r"""
def perturb(t):
    out = re.sub(r"\d+(?:\.\d+)?", lambda m: str(round(float(m.group()) * 2 + 1, 1)).rstrip("0").rstrip("."), t)
    return out if out != t else "It is firmly established that " + t[0].lower() + t[1:].replace("not ", "")
lab_rows = []
qs_list = list(T4.values())
for i, t in enumerate(qs_list):
    q, ev = t["q"], t["ev"]; got = {e["doc_id"] for e in ev}
    sup = [f for fid, f in zip(q["facts"], q["fact_texts"]) if set(q["fact_support"][fid]) & got][:3]
    other = qs_list[(i + 1) % len(qs_list)]["q"]["fact_texts"][0]
    numeric = next((f for f in q["fact_texts"] if re.search(r"\d", f)), q["fact_texts"][0])
    claims = [{"text": s, "cited": []} for s in sup] + [{"text": other, "cited": []}, {"text": perturb(numeric), "cited": []}]
    truth = [False] * len(sup) + [True, True]
    ver = verify_claims(llm, claims, ev, purpose="t4_detector_test")
    lab_rows += [{"qid": q["qid"], "claim": c["text"][:80], "truth_hallucinated": tr, "pred_flagged": v["flagged"]} for c, tr, v in zip(claims, truth, ver)]
lab = pd.DataFrame(lab_rows)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm = confusion_matrix(lab.truth_hallucinated, lab.pred_flagged, labels=[False, True])
p, rcl, f1 = precision_score(lab.truth_hallucinated, lab.pred_flagged), recall_score(lab.truth_hallucinated, lab.pred_flagged), f1_score(lab.truth_hallucinated, lab.pred_flagged)
print(f"Detector on {len(lab)} labelled claims: precision {p:.2f}, recall {rcl:.2f}, F1 {f1:.2f}")
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=["pred supported", "pred hallucinated"], yticklabels=["true supported", "true hallucinated"], cbar=False)
axes[0].set_title("Fig 4.2 – Hallucination detector confusion matrix")
axes[1].bar(["before self-correction", "after self-correction"], [100 * hb, 100 * ha], color=[PALETTE[3], PALETTE[2]]); bar_labels(axes[1], "{:.1f}%")
axes[1].set_ylabel("Hallucination rate (%)"); axes[1].set_title("Fig 4.3 – Hallucination reduction (10 held-out questions)")
b = bins.dropna()
axes[2].plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
axes[2].plot(b.mean_conf, b.accuracy, marker="o", color=PALETTE[0], label=f"assistant (ECE={ece:.3f})")
for rr in b.itertuples(): axes[2].annotate(f"n={rr.n}", (rr.mean_conf, rr.accuracy), fontsize=7, xytext=(3, -10), textcoords="offset points")
axes[2].set_xlabel("Predicted confidence"); axes[2].set_ylabel("Actual correctness"); axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1.05); axes[2].legend(fontsize=8)
axes[2].set_title("Fig 4.4 – Calibration plot")
plt.tight_layout(); save(fig, "fig4_halluc_calibration"); plt.show()
RESULTS["task4"]["detector"] = {"precision": p, "recall": rcl, "f1": f1, "cm": cm.tolist()}
"""),
md(r"""
### 4.5 Multi-source vs single-source synthesis quality
Each held-out question is answered with evidence restricted to one source (Wikipedia, academic or news), two sources (Wikipedia + academic), and all three. The same synthesis prompt and the same judge are used, and quality is the mean of the four rubric criteria (completeness, accuracy, balance, citation quality).
"""),
code(r"""
src_cfgs = {"Wikipedia only": ("wikipedia",), "Academic only": ("academic",), "News only": ("news",), "Wikipedia + Academic": ("wikipedia", "academic")}
sq_rows = []
for qid, t in T4.items():
    q = t["q"]
    for name, srcs in src_cfgs.items():
        per = {s: [] for s in srcs}
        for sq in t["subs"]:
            for s, hl in R.search(sq, srcs, 5).items(): per[s] += hl
        fused = [h for h, _ in weighted_rrf(per, SOURCE_WEIGHT_PRESETS["credibility"], doc_level_credibility=True)]
        ev = hits_to_evidence(ce.rerank(q["question"], fused[:20])[:8], 140)
        ans, _ = synthesize(llm, q["question"], ev, purpose="t4_single_source_synth")
        qual = judge_synthesis_quality(llm, q, ans, ev, purpose="t4_single_source_judge")
        sq_rows.append({"qid": qid, "config": name, "n_sources": len(srcs), **{k: qual[k] for k in ("completeness", "accuracy", "balance", "citation_quality", "overall")}})
    qual = judge_synthesis_quality(llm, q, t["res"].answer, t["ev"], purpose="t4_multi_source_judge")
    t["quality"] = qual
    sq_rows.append({"qid": qid, "config": "Multi-source + conflict resolution", "n_sources": 3, **{k: qual[k] for k in ("completeness", "accuracy", "balance", "citation_quality", "overall")}})
sqdf = pd.DataFrame(sq_rows)
sq_mean = sqdf.groupby("config")[["n_sources", "completeness", "accuracy", "balance", "citation_quality", "overall"]].mean().sort_values("overall")
display(Markdown("**Table 4.8 – Synthesis quality (1–5) by evidence source configuration (10 held-out questions).**")); display(sq_mean.round(2))
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
sqdf.groupby("n_sources").overall.mean().plot(kind="bar", ax=axes[0], color=PALETTE[:3], rot=0); bar_labels(axes[0], "{:.2f}")
axes[0].set_xlabel("Number of source types used"); axes[0].set_ylabel("Synthesis quality (1-5)"); axes[0].set_ylim(0, 5.5); axes[0].set_title("Fig 4.5 – Synthesis quality vs number of sources")
sq_mean[["completeness", "accuracy", "balance", "citation_quality"]].plot(kind="barh", ax=axes[1], color=PALETTE[:4]); axes[1].set_xlim(0, 5.5); axes[1].set_title("Fig 4.6 – Rubric criteria per configuration")
plt.tight_layout(); save(fig, "fig4_sources"); plt.show()
RESULTS["task4"]["synthesis_quality"] = sq_mean.overall.round(3).to_dict()
"""),
md(r"""
### 4.6 RAG Triad evaluation on the 10 held-out questions (LLM-as-a-Judge)
The judge (gemini-3.5-flash-lite, temperature 0) scores **Context Relevance**, **Groundedness** and **Answer Relevance** on anchored 1–5 rubrics. The prompt includes worked examples for levels 1, 3 and 5, shown in full below. To stop the judge from handing out 5s by default, it must first **list every claim that is not stated in the retrieved context** (true background knowledge counts as unsupported), and only then score; groundedness 5 is allowed only when that list is empty. A query *fails* at the first stage, in pipeline order (retrieval → grounding → answer), that scores **< 4**: for a research assistant a 3 already means invented details or an ignored part of the question.
"""),
code(r"""
from ra.evaluation import rag_triad, triad_failure_stage, TRIAD_RUBRIC, cohen_kappa
print(TRIAD_RUBRIC)
tri_rows = []
for qid, t in T4.items():
    s = rag_triad(llm, t["q"]["question"], t["ev"], t["res"].answer, purpose="t4_rag_triad")
    t["triad"] = s
    tri_rows.append({"qid": qid, "question": t["q"]["question"][:60], "context_relevance": s["context_relevance"], "groundedness": s["groundedness"],
                     "answer_relevance": s["answer_relevance"], "failed_stage": triad_failure_stage(s),
                     "why": (s["rationale"] or {}).get({"retrieval": "context_relevance", "grounding": "groundedness", "answer": "answer_relevance"}.get(triad_failure_stage(s), "groundedness"), "") if isinstance(s["rationale"], dict) else ""})
tri = pd.DataFrame(tri_rows)
tri_mean = tri[["context_relevance", "groundedness", "answer_relevance"]].mean()
display(Markdown("**Table 4.9 – RAG Triad scores and failure diagnosis per query.**")); display(tri)
print("Means:", tri_mean.round(2).to_dict(), "| failures by stage:", tri.failed_stage.value_counts().to_dict())
fig, ax = plt.subplots(figsize=(7, 3.5)); tri_mean.plot(kind="bar", ax=ax, color=PALETTE[:3], rot=0); bar_labels(ax, "{:.2f}")
ax.set_ylim(0, 5.5); ax.set_ylabel("mean score (1-5)"); ax.set_title("Fig 4.7 – RAG Triad (10 held-out questions)"); save(fig, "fig4_7_triad"); plt.show()
RESULTS["task4"]["triad"] = tri_mean.round(3).to_dict(); RESULTS["task4"]["triad_failures"] = tri.failed_stage.value_counts().to_dict()
"""),
md(r"""
#### 4.6.1 Judge reliability: Cohen's κ against human labels
Human annotators score the same (question, context, answer) items with the same rubric, without seeing the judge's scores. We export a template with all 30 judge items (10 questions × 3 criteria). **At least 15 must be labelled by the team** in `data/human_labels.csv`, column `human_score`. We report unweighted κ, quadratic-weighted κ on the 1–5 scale, and κ on the binarised pass/fail decision (≥ 4, the same threshold as the failure diagnosis in Table 4.9).
"""),
code(r"""
items = []
for qid, t in T4.items():
    ctx = " || ".join(f"[{e['sid']}] {e['title']}: {e['text'][:200]}" for e in t["ev"][:5])
    for crit in ("context_relevance", "groundedness", "answer_relevance"):
        items.append({"item_id": f"{qid}:{crit}", "criterion": crit, "question": t["q"]["question"], "context_excerpt": ctx,
                      "answer": t["res"].answer, "judge_score": t["triad"][crit]})
items = pd.DataFrame(items)
try:  # the blank template for annotators (may be open in a spreadsheet program, which locks it on Windows)
    items.drop(columns="judge_score").assign(human_score="").to_csv("data/human_labels_template.csv", index=False)
except PermissionError:
    print("note: data/human_labels_template.csv is open in another program - template not rewritten")
hp = Path("data/human_labels.csv")
if hp.exists():
    hl = pd.read_csv(hp).dropna(subset=["human_score"])
    m = items.merge(hl[["item_id", "human_score"]], on="item_id")
    m["human_score"] = m.human_score.astype(int)
    k_u = cohen_kappa(m.judge_score.tolist(), m.human_score.tolist(), labels=[1, 2, 3, 4, 5])
    k_q = cohen_kappa(m.judge_score.tolist(), m.human_score.tolist(), weights="quadratic", labels=[1, 2, 3, 4, 5])
    k_b = cohen_kappa((m.judge_score >= 4).tolist(), (m.human_score >= 4).tolist(), labels=[False, True])
    print(f"{len(m)} human-labelled items | Cohen's kappa: unweighted {k_u:.3f}, quadratic-weighted {k_q:.3f}, pass/fail {k_b:.3f} | exact agreement {np.mean(m.judge_score == m.human_score):.2f}")
    display(pd.crosstab(m.human_score, m.judge_score, rownames=["human"], colnames=["judge"]))
    RESULTS["task4"]["kappa"] = {"n": len(m), "unweighted": k_u, "quadratic": k_q, "binary": k_b}
else:
    print("data/human_labels.csv not found -> template written to data/human_labels_template.csv. "
          "Fill `human_score` (1-5) for >= 15 items, save as data/human_labels.csv, and re-run this cell.")
# Sanity check of our kappa implementation against scikit-learn on synthetic ratings
from sklearn.metrics import cohen_kappa_score
a_, b_ = [1, 2, 3, 4, 5, 3, 4, 2, 5, 1, 3, 4, 4, 2, 5], [1, 2, 3, 3, 5, 3, 4, 3, 4, 1, 2, 4, 5, 2, 5]
print(f"kappa check: ours {cohen_kappa(a_, b_, labels=[1,2,3,4,5]):.4f} vs sklearn {cohen_kappa_score(a_, b_):.4f}; quadratic ours "
      f"{cohen_kappa(a_, b_, 'quadratic', [1,2,3,4,5]):.4f} vs sklearn {cohen_kappa_score(a_, b_, weights='quadratic'):.4f}")
save_results()
"""),
md(r"""
### 4.7 Inference — Task 4
<!--T4_INFERENCE-->
"""),
]
