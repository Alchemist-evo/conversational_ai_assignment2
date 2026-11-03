from common import code, md, module

CELLS = [
md(r"""
---
# Module 2 · Task 3 — Memory, Query Decomposition, Iterative Refinement, Multi-Source Retrieval & Advanced RAG (2 marks)

**Objective.** A memory system and retrieval pipeline that decomposes complex research questions, refines them iteratively, retrieves from multiple sources, applies advanced RAG variants and consciously engineers the context window.

**Memory architecture**
| Memory | What it stores | Implementation | Lifetime |
|---|---|---|---|
| Working | Last N = 10 research steps (plans, observations, findings) | `collections.deque(maxlen=10)` + LLM rolling summary of evicted steps | One session |
| Episodic | Past research sessions (query, findings, user preference) | MiniLM embeddings in FAISS IndexFlatIP (cosine), top-k recall, appended after each query | Persistent |
| Semantic | Domain knowledge (4 collections) | Chunk → embed → store (BM25 + FAISS) → hybrid retrieve | Persistent |
"""),
module("memory.py"),
md(r"""
### 3.1 Working memory: bounded deque with truncation and a rolling summary
We push 14 research steps into a deque with N = 10. Whenever a step falls off the left end, it is queued for summarisation, and every 3 evictions the LLM folds the queue into a ≤ 120-word rolling summary. Old information is therefore *compressed* rather than lost, and the working-memory token cost stays bounded.
"""),
code(r"""
from ra.memory import WorkingMemory, EpisodicMemory
wm = WorkingMemory(n=10, llm=llm, summarize_every=3)
steps = [("plan", "Decompose: how does C4 photosynthesis differ from C3?"), ("search", "academic: C3 C4 RuBisCO -> D3-A1"),
         ("finding", "C3 fixes CO2 via RuBisCO into 3-PGA [D3-A1]"), ("finding", "RuBisCO oxygenation = photorespiration, worse in heat [D3-A1]"),
         ("search", "wikipedia: C4 carbon fixation"), ("finding", "C4 uses PEP carboxylase in mesophyll -> oxaloacetate [D3-A1]"),
         ("finding", "Kranz anatomy concentrates CO2 in bundle sheath [WIKI-C4]"), ("search", "news: C4 rice"),
         ("finding", "C4 rice project aims +50% yield, years away [D3-N1]"), ("finding", "C4 costs ~2 extra ATP per CO2 [D3-A2]"),
         ("finding", "C4 advantage in hot, dry, high-light conditions [D3-A2]"), ("finding", "3% of species are C4 but 23% of productivity [D3-A2]"),
         ("plan", "Gap: water-use efficiency numbers"), ("finding", "C4 higher water and nitrogen use efficiency [D3-A2]")]
log = []
for i, (t, c) in enumerate(steps, 1):
    ev = wm.add({"type": t, "content": c})
    log.append({"step": i, "buffer_len": len(wm.buffer), "evicted": ev["content"][:50] if ev else "", "summary_calls": wm.summary_calls, "wm_tokens": wm.tokens()})
wm.flush()
display(Markdown("**Table 3.1 – Working-memory truncation log.**")); display(pd.DataFrame(log))
print("Rolling summary of evicted steps:\n", textwrap.fill(wm.summary, 120)); print("\nCurrent buffer (10 most recent):"); print(wm.render()[-900:])
"""),
md(r"""
### 3.2 Episodic memory: 50 synthetic past sessions in a vector store
"""),
code(r"""
epi = EpisodicMemory(D.sessions)
display(Markdown("**Table 3.2 – Sample of the 50 SYNTHETIC research sessions (flag `synthetic=True`).**"))
display(pd.DataFrame(D.sessions)[["session_id", "record_type", "date", "user_query", "key_findings", "user_preference"]].head(6))
for qtext in ["What is the evidence that intermittent fasting helps weight loss?", "why does the economy grow"]:
    print(f"\nQuery: {qtext}")
    for s, sim in epi.retrieve(qtext, k=3): print(f"  cos={sim:.3f}  {s['session_id']}  {s['user_query'][:70]}")
assert all(s["topic_id"] is None or D.topics[s["topic_id"]]["split"] == "dev" for s in D.sessions), "held-out leaked into memory!"
print("\nCheck passed: no held-out topic appears in episodic memory.")
"""),
md(r"""
### 3.3 Semantic memory: chunking strategy × overlap experiment
**Benchmark.** We build a long-document corpus by concatenating NQ passages into documents of about 2,000 tokens. The 100 `retrieval_dev` gold passages are mixed with 300 distractor passages and grouped 8 per document (seeded). Each gold passage therefore sits inside a long document, as real articles do.

**Metric.** *Answer Recall@5* is the share of the 100 questions whose gold short answer string appears in at least one of the top-5 retrieved chunks. The retriever is hybrid BM25 + dense + RRF throughout, so only chunking varies.

**Grid.** Fixed-size (256 and 512 tokens), recursive (256 target) and semantic (embedding-similarity breakpoints at the 80th percentile, max 512), each crossed with overlap ∈ {0.1, 0.2, 0.3}. We also record how many tokens the top-5 chunks put into the context window (cost).
"""),
code(r"""
from ra.chunking import chunk_document
from ra.evaluation import gold_match, recall_at_k, fact_coverage, redundancy
rng = random.Random(SEED)
rd = D.nq_split("retrieval_dev"); pool = rd + D.nq_split("distractor")[:300]; rng.shuffle(pool)
long_docs = [{"doc_id": f"LONG-{i:03d}", "source": "wikipedia", "title": "", "text": "\n\n".join(p["passage"] for p in pool[i:i+8])} for i in range(0, len(pool), 8)]
from ra.tokens import count_tokens
print(f"{len(long_docs)} long documents, mean {np.mean([count_tokens(d['text']) for d in long_docs]):.0f} tokens")
configs = [(s, z, o) for s, z in [("fixed", 256), ("fixed", 512), ("recursive", 256), ("semantic", 512)] for o in (0.1, 0.2, 0.3)]
ch_rows = []
for strat, size, ov in configs:
    t0 = time.time()
    chunks = [c for d in long_docs for c in chunk_document(d, strat, size, ov, emb)]
    idx = HybridIndex(chunks, "bench"); build_t = time.time() - t0
    hits_all, lat = [], []
    for r in rd:
        t1_ = time.perf_counter(); hits = idx.search(r["question"], k=5); lat.append(time.perf_counter() - t1_)
        hits_all.append(hits)
    rec5 = np.mean([any(gold_match(h.chunk.text, r["answers"]) for h in hits) for r, hits in zip(rd, hits_all)])
    rec1 = np.mean([gold_match(hits[0].chunk.text, r["answers"]) if hits else 0 for r, hits in zip(rd, hits_all)])
    ctx_tok = np.mean([sum(count_tokens(h.chunk.text) for h in hits) for hits in hits_all])
    ch_rows.append({"strategy": strat, "size": size, "overlap": ov, "n_chunks": len(chunks), "mean_chunk_tok": np.mean([c.n_tokens for c in chunks]),
                    "answer_recall@1": rec1, "answer_recall@5": rec5, "context_tokens@5": ctx_tok, "query_latency_ms": 1000 * np.mean(lat), "index_build_s": build_t})
emb.save()
chdf = pd.DataFrame(ch_rows)
chdf["label"] = chdf.strategy + "-" + chdf["size"].astype(str)
best = chdf.sort_values(["answer_recall@5", "answer_recall@1", "context_tokens@5"], ascending=[False, False, True]).iloc[0]
display(Markdown("**Table 3.3 – Chunking strategy × overlap (100 NQ questions over long documents).**")); display(chdf.drop(columns="label").round(3))
print(f"BEST: {best.strategy} size={best['size']} overlap={best.overlap} -> Recall@5 {best['answer_recall@5']:.2f}, {best['context_tokens@5']:.0f} context tokens")
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for i, lab in enumerate(chdf.label.unique()):
    s = chdf[chdf.label == lab]; axes[0].plot(s.overlap, s["answer_recall@5"], marker="o", label=lab, color=PALETTE[i])
    axes[1].plot(s.overlap, s["context_tokens@5"], marker="o", label=lab, color=PALETTE[i])
axes[0].set_title("Fig 3.1 – Answer Recall@5 vs overlap"); axes[0].set_xlabel("overlap ratio"); axes[0].legend()
axes[1].set_title("Fig 3.2 – Context tokens used by top-5 chunks"); axes[1].set_xlabel("overlap ratio")
plt.tight_layout(); save(fig, "fig3_chunking"); plt.show()
RESULTS["task3"] = {"chunking": chdf.round(4).to_dict("records"), "best_chunking": {"strategy": best.strategy, "size": int(best["size"]), "overlap": float(best.overlap),
                    "recall5": float(best["answer_recall@5"])}}
"""),
code(r"""
# Rebuild the knowledge collections with the experimentally selected chunking strategy
CH = D.build_chunks(best.strategy, int(best["size"]), float(best.overlap), embedder=emb)
R = MultiSourceRetriever(CH, role="researcher"); emb.save()
print("Collections rebuilt with", best.strategy, int(best["size"]), best.overlap, {k: len(v) for k, v in CH.items()})
"""),
md(r"""
### 3.4 Hybrid retrieval (BM25 + dense + RRF) and cross-encoder: Recall@5 / Recall@10
**How it works.**
* **BM25** scores term overlap. Term frequency saturates through k1 and is normalised by document length through b. It is good for rare entities, numbers and exact terms.
* **Dense** retrieval embeds query and chunk with MiniLM (384-d, L2-normalised) and ranks by cosine similarity using FAISS inner product. It captures paraphrase and semantics.
* **RRF** fuses the two *rank lists*: score(d) = Σ 1/(60 + rank). It needs no score calibration and rewards documents that both retrievers rank well.
* The **cross-encoder** reads (query, passage) jointly and re-scores the top-30 fused candidates. It is the most accurate stage and also the slowest.

**Labelled evaluation.** On the 100 NQ `retrieval_dev` questions over the full Wikipedia collection, the gold passage is the relevance label, giving true Recall@k. On the 8 curated dev questions, the relevant documents are the gold support documents of the key facts, across all three sources.
"""),
code(r"""
ce = CrossEncoderReranker()
def ret_eval(questions, rel_fn, sources, label):
    rows = []
    for method in ["bm25", "dense", "hybrid", "hybrid+CE"]:
        r5, r10, lat = [], [], []
        for q in questions:
            t0 = time.perf_counter()
            if method == "hybrid+CE":
                hits = R.search_flat(q["question"], sources, k=30, method="hybrid"); hits = ce.rerank(q["question"], hits)
            else:
                hits = R.search_flat(q["question"], sources, k=10, method=method)
            lat.append(time.perf_counter() - t0)
            if len(sources) > 1 and method != "hybrid+CE":  # interleave sources by rank for multi-source lists
                hits = sorted(hits, key=lambda h: (h.rank, -h.score))
            ids = [h.doc_id for h in hits]; rel = rel_fn(q)
            r5.append(recall_at_k(ids, rel, 5)); r10.append(recall_at_k(ids, rel, 10))
        rows.append({"benchmark": label, "method": method, "Recall@5": np.nanmean(r5), "Recall@10": np.nanmean(r10), "latency_ms": 1000 * np.mean(lat)})
    return rows
nq_dev_q = [{"question": r["question"], "gold": r["doc_id"]} for r in rd]
rows = ret_eval(nq_dev_q, lambda q: {q["gold"]}, ("wikipedia",), "NQ retrieval_dev (100, wikipedia)")
dev_cq = D.by_use("task3")
rows += ret_eval(dev_cq, D.relevant_docs, ("academic", "wikipedia", "news"), "Curated dev (8, multi-source)")
retdf = pd.DataFrame(rows)
display(Markdown("**Table 3.4 – Retrieval configurations.**")); display(retdf.round(3))
fig, ax = plt.subplots(figsize=(9, 3.8))
retdf.pivot(index="method", columns="benchmark", values="Recall@10").reindex(["bm25", "dense", "hybrid", "hybrid+CE"]).plot(kind="bar", ax=ax, rot=0, color=PALETTE[:2])
bar_labels(ax, "{:.2f}"); ax.set_ylabel("Recall@10"); ax.set_title("Fig 3.3 – Recall@10 vs retrieval configuration"); ax.set_ylim(0, 1.1)
save(fig, "fig3_3_recall10"); plt.show()
RESULTS["task3"]["retrieval"] = retdf.round(4).to_dict("records")
"""),
module("decomposition.py"),
md(r"""
### 3.5 Query decomposition (CoT) and iterative refinement
Decomposition uses a **CoT prompt**: the model first reasons about the question's *facets* (definitions, mechanisms, comparisons, evidence for and against, recent developments) and then writes 2–4 self-contained sub-queries. Each sub-query retrieves the top-5 from each source. In **iterative refinement**, the model then reads compact evidence notes, lists the remaining *gaps* and proposes up to 2 new sub-queries, for at most 3 iterations.

* **Evidence Coverage (%)** is the share of the question's gold key facts for which at least one supporting document was retrieved.
* **Retrieval Redundancy (%)** is the share of retrieved chunks that duplicate an already-retrieved chunk.
"""),
code(r"""
from ra.decomposition import decompose, refine, retrieve_for, iterative_research
traces = {}
for q in dev_cq:
    traces[q["qid"]] = iterative_research(llm, R, q["question"], gold_q=q, max_iters=3)
dec_rows = [{"qid": qid, "question": tr["question"][:55], "static sub-queries": tr["iterations"][0]["new_sub_queries"],
             "refinement sub-queries": [it["new_sub_queries"] for it in tr["iterations"][1:]]} for qid, tr in traces.items()]
display(Markdown("**Table 3.5 – CoT decomposition and refinement sub-queries.**")); display(pd.DataFrame(dec_rows))
q_ex = dev_cq[2]["qid"]; print(f"CoT facet reasoning for {q_ex}:\n", textwrap.fill(traces[q_ex]["iterations"][0]["info"]["reasoning"], 120))
"""),
code(r"""
it_rows = []
for qid, tr in traces.items():
    its = tr["iterations"]
    for k in range(4):
        it = its[min(k, len(its) - 1)]
        it_rows.append({"qid": qid, "iteration": k, "coverage": it["coverage"], "redundancy": it["redundancy"], "cum_tokens": it["cum_tokens"],
                        "cum_latency_s": it["cum_llm_latency_s"] + it["cum_retrieval_latency_s"], "n_hits": it["n_hits"]})
itdf = pd.DataFrame(it_rows)
curve = itdf.groupby("iteration")[["coverage", "redundancy", "cum_tokens", "cum_latency_s"]].mean()
curve["marginal_coverage_gain"] = curve.coverage.diff()
static_vs = pd.DataFrame({"static decomposition": curve.loc[0], "iterative refinement (≤3)": curve.loc[3]}).T[["coverage", "redundancy", "cum_tokens", "cum_latency_s"]]
display(Markdown("**Table 3.6 – Coverage, redundancy and cost per refinement iteration (mean of 8 dev questions).**")); display(curve.round(3))
display(Markdown("**Table 3.7 – Static vs iterative.**")); display(static_vs.round(3))
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, (qid, s) in enumerate(itdf.groupby("qid")): ax.plot(s.iteration, 100 * s.coverage, color="grey", alpha=0.3)
ax.plot(curve.index, 100 * curve.coverage, marker="o", lw=3, color=PALETTE[0], label="mean coverage")
ax.plot(curve.index, 100 * curve.redundancy, marker="s", lw=2, color=PALETTE[3], label="mean redundancy")
ax.set_xticks(range(4)); ax.set_xlabel("Iteration (0 = static decomposition)"); ax.set_ylabel("%"); ax.legend()
ax.set_title("Fig 3.4 – Evidence coverage vs iteration number"); save(fig, "fig3_4_coverage_iteration"); plt.show()
RESULTS["task3"]["refinement"] = curve.round(4).to_dict()
"""),
md(r"""
### 3.6 Multi-source retrieval per sub-query, and the retrieval ablation study (Configs A–D)
* **A**: single query, single source (Wikipedia only, top-10)
* **B**: single query, multi-source (top-5 from academic, Wikipedia and news)
* **C**: query decomposition (first 3 sub-queries), multi-source
* **D**: iterative refinement (static + ≤ 3 refinement rounds), multi-source

**Synthesis quality** (1–5) is scored by the LLM judge on four anchored criteria: completeness, accuracy, balance and citation quality. We report the mean, computed from a synthesis over the top-8 unique documents for each configuration. The ablation runs on all 8 dev questions, and one example question is shown in detail.
"""),
code(r"""
from ra.synthesis import hits_to_evidence, synthesize, full_synthesis
from ra.evaluation import judge_synthesis_quality
def rank_dedupe(hits, n=8):
    return hits_to_evidence(sorted(hits, key=lambda h: (h.rank, -h.score)), max_words=140)[:n]
ex_q = dev_cq[4]
per = R.search(traces[ex_q["qid"]]["iterations"][0]["new_sub_queries"][0], ("academic", "wikipedia", "news"), 5)
display(Markdown(f"**Multi-source top-5 per source for the first sub-query of {ex_q['qid']}:**"))
display(pd.DataFrame([{"source": s, "rank": h.rank, "doc": h.doc_id, "title": h.chunk.meta.get("title", "")[:60], "rrf": round(h.score, 4)} for s, l in per.items() for h in l]))
abl_rows, abl_answers = [], {}
for q in dev_cq:
    tr = traces[q["qid"]]
    cfg_hits = {"A: 1 query, Wikipedia": R.search(q["question"], ("wikipedia",), 10)["wikipedia"],
                "B: 1 query, multi-source": R.search_flat(q["question"], ("academic", "wikipedia", "news"), 5),
                "C: decomposition, multi-source": retrieve_for(R, tr["iterations"][0]["new_sub_queries"][:3]),
                "D: iterative, multi-source": tr["hits"]}
    for cfg, hits in cfg_hits.items():
        ev = rank_dedupe(hits)
        ans, r = synthesize(llm, q["question"], ev, purpose="t3_ablation_synth")
        qual = judge_synthesis_quality(llm, q, ans, ev, purpose="t3_ablation_judge")
        abl_answers[(q["qid"], cfg)] = ans
        abl_rows.append({"qid": q["qid"], "config": cfg, "coverage": fact_coverage([h.doc_id for h in hits], q),
                         "redundancy": redundancy([h.chunk.chunk_id for h in hits]), "quality": qual["overall"], "n_hits": len(hits)})
abl = pd.DataFrame(abl_rows)
display(Markdown(f"**Table 3.8 – Retrieval ablation for one complex question ({ex_q['qid']}: {ex_q['question']}).**"))
display(abl[abl.qid == ex_q["qid"]].round(3))
abl_mean = abl.groupby("config")[["coverage", "redundancy", "quality"]].mean()
display(Markdown("**Table 3.9 – Retrieval ablation averaged over the 8 dev questions.**")); display(abl_mean.round(3))
RESULTS["task3"]["retrieval_ablation"] = abl_mean.round(4).to_dict()
"""),
module("context.py"),
md(r"""
### 3.7 Context engineering: the final prompt, its token breakdown and component classification
The final prompt is assembled from labelled components:
1. system prompt
2. tool definitions
3. top-2 episodic sessions
4. top-3 RAG chunks
5. the last 3 conversation turns
6. the current query
7. reserved output budget

The current query comes **last**, closest to generation. Evidence is ordered with the most relevant chunks at the edges, which mitigates "lost in the middle". First we hold a real 10-turn research conversation, which also gives us the token-growth curve. Its last 3 turns then serve as the conversation history in the breakdown.
"""),
code(r"""
from ra.context import build_context, token_breakdown, classify_components, render_history, arrange, long_context_docs, SYSTEM_PROMPT
from ra.tokens import count_tokens
conv_q = ["What causes inflation?", "What is demand-pull inflation exactly?", "And cost-push inflation?", "How do expectations matter?",
          "What did the monetarists argue?", "Is the money-inflation link strong today?", "What drove inflation in 2021-2023?",
          "How high did US inflation get?", "How do central banks respond?", "Summarise the main causes in two sentences."]
history, summary, growth = [], "", []
wm_conv = WorkingMemory(n=10, llm=llm, summarize_every=3)
for i, uq in enumerate(conv_q, 1):
    ev = hits_to_evidence(R.search_flat(uq, ("academic", "wikipedia", "news"), 1), max_words=110)[:3]
    ctx = build_context(uq, rag_evidence=ev, history=history, tool_schemas=TOOL_SCHEMAS,
                        working_memory=("Summary of earlier turns: " + wm_conv.summary) if wm_conv.summary else "")
    eng_tokens = sum(count_tokens(m["content"]) for m in ctx["messages"])
    naive_hist = "\n\n".join(f"User: {t['user']}\nAssistant: {t['assistant']}" for t in history)
    naive_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(ctx["components"]["Tool Definitions"]) + count_tokens(naive_hist) + count_tokens(ctx["components"]["Retrieved RAG Knowledge"]) + count_tokens(uq)
    r = llm.chat(ctx["messages"], max_tokens=260, purpose="t3_conversation")
    history.append({"user": uq, "assistant": r.text}); wm_conv.add({"type": "turn", "content": f"Q: {uq} A: {r.text[:200]}"})
    growth.append({"turn": i, "naive_full_history": naive_tokens, "engineered (last-3 + summary)": eng_tokens})
growth = pd.DataFrame(growth).set_index("turn")
display(growth.T)
"""),
code(r"""
# Token breakdown for one query (turn 11 of the conversation): all required components
query = "Compare demand-pull and cost-push explanations of inflation with the monetarist view."
epi_items = epi.retrieve(query, k=2)
rag_ev = hits_to_evidence(ce.rerank(query, R.search_flat(query, ("academic", "wikipedia", "news"), 5)), max_words=140)[:3]
rag_ev = [rag_ev[i] for i in ([0, 2, 1] if len(rag_ev) == 3 else range(len(rag_ev)))]  # most relevant at the edges
CTX = build_context(query, EpisodicMemory.render(epi_items), rag_ev, history, TOOL_SCHEMAS)
tb = token_breakdown(CTX)
display(Markdown("**Table 3.10 – Context-engineering token breakdown for one query (actual tokens, o200k_base).**")); display(tb)
print("---- Final prompt (abbreviated) ----\nSYSTEM:", CTX["messages"][0]["content"][:700], "...\n\nUSER:", CTX["messages"][1]["content"][:1500], "...")
cls = classify_components(CTX, query, epi_items, rag_ev, history)
cat = cls.groupby("category").tokens.sum(); cat_pct = (100 * cat / cat.sum()).round(1)
display(Markdown("**Table 3.11 – Context component classification (similarity to the query, threshold 0.45).**")); display(cls.round(3))
display(pd.DataFrame({"tokens": cat, "% of context": cat_pct}))
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
comp = tb[tb.Component != "TOTAL"]
left = 0
for i, rrow in enumerate(comp.itertuples()):
    axes[0].barh(["context window"], [rrow.Tokens], left=left, color=PALETTE[i % 8], label=f"{rrow.Component} ({rrow.Tokens})"); left += rrow.Tokens
axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3); axes[0].set_title("Fig 3.5 – Token breakdown across components (stacked)")
axes[1].plot(growth.index, growth.iloc[:, 0], marker="o", label="naive: full history", color=PALETTE[3])
axes[1].plot(growth.index, growth.iloc[:, 1], marker="s", label="engineered: last 3 turns + summary", color=PALETTE[0])
axes[1].set_xlabel("turn"); axes[1].set_ylabel("prompt tokens"); axes[1].legend(); axes[1].set_title("Fig 3.6 – Token usage growth over 10 turns")
plt.tight_layout(); save(fig, "fig3_context"); plt.show()
RESULTS["task3"]["token_breakdown"] = tb.to_dict("records"); RESULTS["task3"]["context_categories_pct"] = cat_pct.to_dict()
"""),
md(r"""
### 3.8 Long context and "lost in the middle"
**Setup.** The query asks for the clinical-trial evidence on intermittent fasting, so the needed facts sit in 3 short documents: 2 synthetic trials and 1 news report. Those facts are specific numbers that the model cannot know from pre-training. The documents are mixed with **6 long, topically related Wikipedia articles** (Fasting, Calorie restriction, Obesity, Dieting, Insulin resistance, Basal metabolic rate), truncated to 2,000 tokens each, for a total context of **12K+ tokens**.

**Strategies.** We test where the relevant evidence sits:
* **beginning-heavy:** relevant documents first
* **end-heavy:** relevant documents last
* **balanced / reordered:** most relevant at both edges
* **middle:** relevant documents buried in the middle, as a control that exposes the problem

Each strategy runs with 2 different distractor orders (seeds 42 and 43). Quality is the judge's 1–5 synthesis score, plus key-fact coverage.
"""),
code(r"""
lq = next(q for q in D.questions if q["qid"] == "D1-Q3")
rel_docs = [D.docs[i] for i in ["D1-A1", "D1-A2", "D1-N1"]]
scores = ce.model.predict([(lq["question"], d["text"]) for d in rel_docs]); rel_docs = [rel_docs[i] for i in np.argsort(-scores)]
distr = long_context_docs(D, D.long_context_distractors, 2000)
lim_rows = []
for seed_ in (42, 43):
    dd = distr[:]; random.Random(seed_).shuffle(dd)
    for strat in ["beginning", "end", "balanced", "middle"]:
        docs = arrange(rel_docs, dd, strat)
        body = "\n\n".join(f"[D{i+1}] {d['title']} ({d['venue']}, {d['year']})\n{d['text']}" for i, d in enumerate(docs))
        prompt = f"Documents:\n{body}\n\nQuestion: {lq['question']}\nAnswer in 150-200 words using specific findings (numbers, study designs) from the documents, citing [D#]."
        r = llm.chat(prompt, role="long_context", max_tokens=400, purpose="t3_lost_in_middle")
        qual = judge_synthesis_quality(llm, lq, r.text, purpose="t3_lim_judge"); corr = judge_correctness(llm, lq, r.text, purpose="t3_lim_cov")
        pos = [i + 1 for i, d in enumerate(docs) if d["doc_id"] in {x["doc_id"] for x in rel_docs}]
        lim_rows.append({"seed": seed_, "strategy": strat, "relevant_positions": pos, "context_tokens": r.input_tokens, "quality": qual["overall"],
                         "key_fact_coverage": corr["coverage"], "model": r.model})
lim = pd.DataFrame(lim_rows)
display(Markdown("**Table 3.12 – Lost-in-the-middle experiment.**")); display(lim)
limm = lim.groupby("strategy")[["quality", "key_fact_coverage", "context_tokens"]].mean().reindex(["beginning", "end", "balanced", "middle"])
display(limm.round(2))
fig, ax = plt.subplots(figsize=(8, 3.5)); limm.quality.plot(kind="bar", ax=ax, color=PALETTE[:4], rot=0); bar_labels(ax, "{:.2f}")
ax.set_ylabel("Synthesis quality (1-5)"); ax.set_ylim(0, 5.5); ax.set_title("Fig 3.7 – Synthesis quality vs evidence position (12K-token context)"); save(fig, "fig3_7_lim"); plt.show()
RESULTS["task3"]["lost_in_middle"] = limm.round(3).to_dict()
"""),
md(r"""
### 3.9 Memory and context ablation (Configs A–E)
All configurations answer the same 8 dev questions with the same generator; only the context changes.
| Config | Context given to the LLM |
|---|---|
| A | **No memory**: question only (closed-book) |
| B | **Working memory only**: question + the session's last 10 research steps (sub-queries and a one-line Wikipedia observation per sub-query) |
| C | **Working + Episodic + RAG**: B + top-3 episodic sessions + top-3 hybrid chunks per source for the raw question, naively concatenated |
| D | **Full context engineering**: structured prompt (system rules, top-2 episodic, decomposition-based retrieval with credibility-weighted RRF, top-8 evidence with relevant evidence at the edges, query last) |
| E | **D + cross-encoder re-ranking** of the fused candidates before selecting the top-8 |
"""),
code(r"""
from ra.synthesis import weighted_rrf, SOURCE_WEIGHT_PRESETS, SYNTH_SYSTEM
mem_rows = []
for q in dev_cq:
    tr = traces[q["qid"]]; subs = tr["all_queries"]
    wmq = WorkingMemory(n=10)
    for sq in subs:
        wmq.add({"type": "search", "content": sq})
        top = R.search(sq, ("wikipedia",), 1)["wikipedia"]
        if top: wmq.add({"type": "observation", "content": " ".join(top[0].chunk.text.split()[:30])})
    tC = time.perf_counter()
    epi3 = EpisodicMemory.render(epi.retrieve(q["question"], 3)); epi2 = EpisodicMemory.render(epi.retrieve(q["question"], 2))
    naive_ev = hits_to_evidence(R.search_flat(q["question"], ("academic", "wikipedia", "news"), 3), 140)
    ret_C = time.perf_counter() - tC
    per = {s: [] for s in ("academic", "wikipedia", "news")}
    for h in tr["hits"]:
        per[h.source].append(h)
    for s in per:  # best rank per doc within each source
        b = {}
        for h in per[s]:
            if h.doc_id not in b or h.rank < b[h.doc_id].rank: b[h.doc_id] = h
        per[s] = sorted(b.values(), key=lambda h: h.rank)
    fused = [h for h, _ in weighted_rrf(per, SOURCE_WEIGHT_PRESETS["credibility"], doc_level_credibility=True)]
    def edges(ev):  # most relevant at the beginning and end
        f, b = [], []
        for i, e in enumerate(ev): (f if i % 2 == 0 else b).append(e)
        out = f + b[::-1]
        for i, e in enumerate(out): e["sid"] = f"S{i+1}"
        return out
    tD = time.perf_counter(); ev_D = edges(hits_to_evidence(fused[:8], 140)); ret_D = ret_C + tr["iterations"][-1]["cum_retrieval_latency_s"] + time.perf_counter() - tD
    tE = time.perf_counter(); ev_E = edges(hits_to_evidence(ce.rerank(q["question"], fused[:20])[:8], 140)); ret_E = ret_D + time.perf_counter() - tE
    ret = {"A: no memory": 0.0, "B: working memory": tr["iterations"][-1]["cum_retrieval_latency_s"], "C: working+episodic+RAG": ret_C,
           "D: full context engineering": ret_D, "E: D + cross-encoder": ret_E}
    cfgs = {
        "A: no memory": [{"role": "user", "content": f"Research question: {q['question']}\nAnswer in 120-220 words."}],
        "B: working memory": [{"role": "user", "content": f"Working memory (recent research steps):\n{wmq.render()}\n\nResearch question: {q['question']}\nAnswer in 120-220 words, citing what you can."}],
        "C: working+episodic+RAG": [{"role": "user", "content": f"{wmq.render()}\n{epi3}\n" + "\n".join(f"[{e['sid']}] {e['title']}: {e['text']}" for e in naive_ev) + f"\n{q['question']}"}],
        "D: full context engineering": build_context(q["question"], epi2, ev_D, None, None)["messages"],
        "E: D + cross-encoder": build_context(q["question"], epi2, ev_E, None, None)["messages"],
    }
    for name, msgs in cfgs.items():
        t0 = time.perf_counter()
        r = llm.chat(msgs, purpose="t3_memory_ablation", max_tokens=450)
        ev_for_judge = {"C: working+episodic+RAG": naive_ev, "D: full context engineering": ev_D, "E: D + cross-encoder": ev_E}.get(name)
        qual = judge_synthesis_quality(llm, q, r.text, ev_for_judge, purpose="t3_memory_judge")
        mem_rows.append({"qid": q["qid"], "config": name, "quality": qual["overall"], "completeness": qual["completeness"], "accuracy": qual["accuracy"],
                         "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "llm_latency_s": r.latency_s,
                         "retrieval_s": ret[name], "total_latency_s": ret[name] + r.latency_s})
mem = pd.DataFrame(mem_rows)
memm = mem.groupby("config")[["quality", "completeness", "accuracy", "input_tokens", "output_tokens", "retrieval_s", "llm_latency_s", "total_latency_s"]].mean()
display(Markdown("**Table 3.13 – Memory/context ablation (mean over 8 dev questions).**")); display(memm.round(2))
fig, ax = plt.subplots(figsize=(10, 3.8)); memm.quality.plot(kind="bar", ax=ax, color=PALETTE[:5], rot=10); bar_labels(ax, "{:.2f}")
ax.set_ylabel("Synthesis quality (1-5)"); ax.set_ylim(0, 5.5); ax.set_title("Fig 3.8 – Synthesis quality vs memory configuration"); save(fig, "fig3_8_memory"); plt.show()
RESULTS["task3"]["memory_ablation"] = memm.round(4).to_dict()
"""),
module("advanced_rag.py"),
md(r"""
### 3.10 Advanced RAG variants: HyDE and CRAG vs the hybrid baseline
* **HyDE.** Questions and answer passages look different in embedding space. HyDE has the LLM write a *hypothetical answer passage* and embeds that for the dense search; BM25 still uses the real question, and RRF fuses the two. It helps when queries are short or phrased differently from the documents.
* **CRAG.** A retrieval *evaluator* scores each retrieved chunk from 0 to 1, and the result decides the action:
  * **Correct:** keep only useful chunks (knowledge refinement).
  * **Incorrect:** discard everything and fall back to **web search** with a rewritten query.
  * **Ambiguous:** do both.
  It helps when the internal collections may not contain the answer.

All three variants go through the same synthesis prompt and the same judge on the 8 dev questions.
"""),
code(r"""
from ra.advanced_rag import baseline, hyde, crag
var_rows, crag_actions = [], []
for q in dev_cq:
    for name, fn in [("Baseline hybrid", lambda: baseline(R, q["question"])), ("HyDE", lambda: hyde(llm, R, q["question"])), ("CRAG", lambda: crag(llm, R, q["question"]))]:
        out = fn()
        ev = rank_dedupe(out["hits"])
        ans, r = synthesize(llm, q["question"], ev, purpose="t3_variant_synth")
        qual = judge_synthesis_quality(llm, q, ans, ev, purpose="t3_variant_judge")
        var_rows.append({"qid": q["qid"], "variant": name, "coverage": fact_coverage([h.doc_id for h in out["hits"]], q), "quality": qual["overall"],
                         "latency_s": out["retrieval_s"] + out["llm_latency"] + r.latency_s, "retrieval_s": out["retrieval_s"],
                         "tokens": out["tokens"] + r.input_tokens + r.output_tokens})
        if name == "CRAG": crag_actions.append({"qid": q["qid"], **{k: v for k, v in out["info"].items() if k != "scores"}})
var = pd.DataFrame(var_rows)
varm = var.groupby("variant")[["coverage", "quality", "latency_s", "tokens"]].mean().reindex(["Baseline hybrid", "HyDE", "CRAG"])
display(Markdown("**Table 3.14 – Advanced RAG variants (mean over 8 dev questions).**")); display(varm.round(3))
display(Markdown("**CRAG evaluator decisions:**")); display(pd.DataFrame(crag_actions))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
varm.quality.plot(kind="bar", ax=axes[0], color=PALETTE[:3], rot=0); bar_labels(axes[0], "{:.2f}"); axes[0].set_ylim(0, 5.5); axes[0].set_title("Fig 3.9 – Synthesis quality vs RAG variant")
varm.latency_s.plot(kind="bar", ax=axes[1], color=PALETTE[3:6], rot=0); bar_labels(axes[1], "{:.2f}"); axes[1].set_ylabel("s"); axes[1].set_title("Fig 3.10 – Latency vs RAG variant")
plt.tight_layout(); save(fig, "fig3_variants"); plt.show()
RESULTS["task3"]["rag_variants"] = varm.round(4).to_dict()
save_results()
"""),
md(r"""
### 3.11 Inference — Task 3
<!--T3_INFERENCE-->
"""),
]
