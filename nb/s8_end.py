from common import code, md

CELLS = [
md(r"""
---
# Summary of expected outputs
"""),
code(r"""
t6 = RESULTS["task6"]; e10 = t6["eval10"]; nq50 = t6["nq50"]
summary = pd.DataFrame([
    ("Synthesis Quality (1–5)", f"{e10['synthesis_quality']:.2f}", "Task 6 §6.3, 10 complex held-out"),
    ("Conflict Resolution Rate (%)", f"{e10['conflict_resolution_rate_%']:.0f}% (gold agreement {e10['gold_agreement_%']:.0f}%)", "Task 6 §6.3"),
    ("Citation Accuracy (%)", f"{e10['citation_accuracy_%']:.0f}%", "Task 6 §6.3"),
    ("Hallucination Rate (%)", f"{e10['halluc_before_%']:.1f}% → {e10['halluc_after_%']:.1f}% (before → after self-correction)", "Task 6 §6.3"),
    ("Confidence Calibration Error (ECE)", f"{t6['ece']:.3f}", "Task 6 §6.3 (claim level)"),
    ("Task Completion Rate (%)", f"{100*t6['completion']:.0f}% complex / {nq50['task_completion_%']:.0f}% NQ-50", "Task 6 §6.3–6.4"),
    ("Cost per Research Query (GPT-4o pricing)", f"${RESULTS['task6']['cost_per_complex_query']:.4f} complex / ${nq50['cost_per_query_$']:.4f} simple", "Task 6 §6.5"),
    ("Latency P50 / P95", f"{t6['latency']['P50']:.1f}s / {t6['latency']['P95']:.1f}s complex; {nq50['latency_P50_s']:.1f}s / {nq50['latency_P95_s']:.1f}s NQ-50", "Task 6 §6.3–6.4"),
    ("Accuracy on 50 held-out NQ", f"{nq50['accuracy_%']:.0f}%", "Task 6 §6.4"),
], columns=["Metric", "Value", "Where"])
display(Markdown("**Table S.1 – Headline results.**")); display(summary)
display(Markdown("**Table S.2 – LLM usage for this notebook run (model that actually served each role).**")); display(llm.usage_table())
save_results()
"""),
md(r"""
# Final conclusion
<!--CONCLUSION-->

# References
1. Kwiatkowski, T., Palomaki, J., Redfield, O., et al. (2019). Natural Questions: A Benchmark for Question Answering Research. *TACL*, 7, 453–466. https://huggingface.co/datasets/natural_questions (CC-BY-SA-3.0). Mirrors used: `google-research-datasets/nq_open`, `sentence-transformers/natural-questions`.
2. Wikipedia contributors. Articles listed in `data/wikipedia_articles.json`, retrieved Nov 2026 via the MediaWiki API (CC-BY-SA).
3. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.
4. Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS*.
5. Zhou, Y., et al. (2023). Large Language Models Are Human-Level Prompt Engineers (APE). *ICLR*.
6. Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR*.
7. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *FnTIR*. Library: `rank-bm25`.
8. Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP*. Model: `sentence-transformers/all-MiniLM-L6-v2`; cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.
9. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs (FAISS). *IEEE TBD*.
10. Cormack, G., Clarke, C., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. *SIGIR*.
11. Liu, N. F., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. *TACL*.
12. Gao, L., et al. (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). *ACL*.
13. Yan, S.-Q., et al. (2024). Corrective Retrieval Augmented Generation (CRAG). arXiv:2401.15884.
14. Asai, A., et al. (2024). Self-RAG. *ICLR*; Jeong, S., et al. (2024). Adaptive-RAG. *NAACL*; Sarthi, P., et al. (2024). RAPTOR. *ICLR*.
15. Es, S., et al. (2024). RAGAS; TruLens RAG Triad documentation (context relevance, groundedness, answer relevance).
16. Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS*. Cohen, J. (1960). A coefficient of agreement for nominal scales.
17. Guo, C., et al. (2017). On Calibration of Modern Neural Networks (ECE). *ICML*.
18. Wang, Y., et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions. *ACL*.
19. Rafailov, R., et al. (2023). Direct Preference Optimization. *NeurIPS*. Ouyang, L., et al. (2022). InstructGPT / RLHF. *NeurIPS*. Schulman, J., et al. (2017). PPO.
20. Hu, E., et al. (2022). LoRA. *ICLR*. Dettmers, T., et al. (2023). QLoRA. *NeurIPS*.
21. Shao, Z., et al. (2024). DeepSeekMath (GRPO). arXiv:2402.03300. Bai, Y., et al. (2022). Constitutional AI. Chen, Z., et al. (2024). SPIN. *ICML*. Lightman, H., et al. (2023). Let's Verify Step by Step (PRM).
22. Google Gemini API documentation (gemini-3.1-flash-lite), ai.google.dev/gemini-api/docs; Groq API documentation (openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b), console.groq.com/docs; both accessed Nov 2026. GPT-4o pricing as specified in the assignment brief.
23. Course material: AIMLCZG521 Sessions 4, 5, 6, 7, 9, 10, 12 (as referenced in the assignment).

*Synthetic data disclosure.* All academic, news and forum documents in `data/curated/` and all 50 episodic sessions are synthetic and are labelled as such in the data and in every citation the system generates (`[Synthetic source]`).
"""),
]
