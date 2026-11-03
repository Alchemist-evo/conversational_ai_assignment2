"""Build the final PDF report from the executed run (outputs/results.json, outputs/figures, nb/inferences.md).

Usage: python report/make_report.py   -> report/Final_Report.html + report/Final_Report.pdf
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures"
R = json.load(open(ROOT / "outputs" / "results.json", encoding="utf-8"))
sys.path.insert(0, str(ROOT))
from build_notebook import load_inferences  # noqa: E402

INF = load_inferences()


def img(name: str, caption: str, width: int = 100) -> str:
    p = FIG / f"{name}.png"
    if not p.exists():
        return f"<p><i>[missing figure {name}]</i></p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}%"><figcaption>{caption}</figcaption></figure>'


def md_to_html(md: str) -> str:
    out = subprocess.run(["pandoc", "-f", "markdown", "-t", "html"], input=md.encode("utf-8"), capture_output=True, check=True)
    return out.stdout.decode("utf-8")


def table(rows: list[dict], cols: list[str] | None = None, fmt: dict | None = None) -> str:
    if not rows:
        return ""
    cols = cols or list(rows[0].keys())
    fmt = fmt or {}
    h = "<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = fmt.get(c, "{:.2f}").format(v)
            cells.append(f"<td>{v}</td>")
        h += "<tr>" + "".join(cells) + "</tr>"
    return h + "</table>"


def sec(key: str) -> str:
    return md_to_html(INF.get(key, "_(not available)_"))


def main():
    t1, t2, t3, t4, t5, t6, t7 = (R.get(k, {}) for k in ("task1", "task2", "task3", "task4", "task5", "task6", "task7"))
    e10, nq = t6.get("eval10", {}), t6.get("nq50", {})
    s1 = t1.get("summary", {})
    tech_rows = [{"Technique": k, "Correct %": s1["correctness"][k], "Depth": s1["depth"][k], "Sources ref.": s1["sources_ref"][k],
                  "In tok": s1["in_tok"][k], "Out tok": s1["out_tok"][k], "Latency s": s1["latency_s"][k], "$ / 1k queries": s1["cost_per_1k_usd"][k]}
                 for k in s1.get("correctness", {})]
    headline = [
        {"Metric": "Synthesis quality (1–5)", "Value": f"{e10.get('synthesis_quality', float('nan')):.2f}"},
        {"Metric": "Conflict resolution rate", "Value": f"{e10.get('conflict_resolution_rate_%', float('nan')):.0f}% (gold agreement {e10.get('gold_agreement_%', float('nan')):.0f}%)"},
        {"Metric": "Citation accuracy", "Value": f"{e10.get('citation_accuracy_%', float('nan')):.0f}%"},
        {"Metric": "Hallucination rate (before → after)", "Value": f"{e10.get('halluc_before_%', float('nan')):.1f}% → {e10.get('halluc_after_%', float('nan')):.1f}%"},
        {"Metric": "Calibration error (ECE)", "Value": f"{t6.get('ece', float('nan')):.3f}"},
        {"Metric": "Task completion", "Value": f"{100*t6.get('completion', float('nan')):.0f}% complex · {nq.get('task_completion_%', float('nan')):.0f}% NQ-50"},
        {"Metric": "Accuracy, 50 held-out NQ", "Value": f"{nq.get('accuracy_%', float('nan')):.0f}%"},
        {"Metric": "Cost per research query (GPT-4o pricing)", "Value": f"${t6.get('cost_per_complex_query', float('nan')):.4f} complex · ${nq.get('cost_per_query_$', float('nan')):.4f} simple"},
        {"Metric": "Latency P50 / P95 (complex)", "Value": f"{t6.get('latency', {}).get('P50', float('nan')):.1f}s / {t6.get('latency', {}).get('P95', float('nan')):.1f}s"},
        {"Metric": "RAG Triad (CR / G / AR)", "Value": " / ".join(f"{v:.2f}" for v in t4.get("triad", {}).values())},
    ]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Final Report – AIMLCZG521 Assignment 2 PS1</title>
<style>
@page {{ size: A4; margin: 16mm 14mm; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #222; max-width: 190mm; margin: auto; }}
h1 {{ color: #2E5A88; font-size: 20pt; margin-bottom: 0; }} h2 {{ color: #2E5A88; border-bottom: 2px solid #2E5A88; padding-bottom: 2px; margin-top: 22px; page-break-after: avoid; }}
h3 {{ color: #333; page-break-after: avoid; }}
table {{ border-collapse: collapse; margin: 8px 0; font-size: 8.8pt; width: 100%; page-break-inside: avoid; }}
th {{ background: #2E5A88; color: white; padding: 4px 6px; text-align: left; }} td {{ border: 1px solid #ccc; padding: 3px 6px; vertical-align: top; }}
tr:nth-child(even) td {{ background: #f4f7fa; }}
figure {{ margin: 8px 0 14px; text-align: center; page-break-inside: avoid; }} figcaption {{ font-size: 8.5pt; color: #555; font-style: italic; }}
.box {{ background: #eef3f8; border-left: 4px solid #2E5A88; padding: 6px 12px; margin: 8px 0; }}
</style></head><body>
<h1>End-to-End Research Assistant with Multi-Source Synthesis and Alignment Analysis</h1>
<p><b>AIMLCZG521 Conversational AI — Assignment 2, Problem Statement 1 · Final Report</b><br>
Team: <i>(names, IDs and contribution areas as listed in the notebook)</i></p>
<div class="box"><b>Summary.</b> We built a research assistant that decomposes questions, calls academic, encyclopedia and news sources in parallel through a ReAct agent with six safety mechanisms, fuses evidence with credibility-weighted RRF and cross-encoder re-ranking, resolves conflicting claims by source credibility, writes cited syntheses, verifies and self-corrects its claims, attaches calibrated confidence, and records a full audit trail. It was evaluated on 10 complex held-out questions and 50 held-out Natural Questions with an LLM judge and a 1–5 rubric. The generator is Google gemini-3.1-flash-lite and the judge is gpt-oss-120b on Groq (both free tiers). Costs are reported at GPT-4o prices, as the brief requires.</div>
<h2>1. Headline results</h2>{table(headline)}
<h2>2. Architecture & workflow</h2>{img('fig0_1_pipeline', 'Figure 1 – Pipeline: Research Question → Decomposition → Multi-Source Function Calling → Evidence → Conflict Resolution → Synthesis → Citation → Memory Update', 92)}
{img('fig6_6_architecture', 'Figure 2 – Production architecture', 88)}
<h2>3. Data</h2>
<p><b>Natural Questions</b> (Kwiatkowski et al., 2019; CC-BY-SA-3.0). 1,000 domain-filtered examples (seed 42), split into Task-1 simple (5), episodic seeds (30), retrieval-dev (100), held-out (50) and distractors (815). The questions and gold answers come from <code>nq_open</code> and the gold passages from <code>sentence-transformers/natural-questions</code>. <b>Wikipedia</b>: 28 real articles. <b>Synthetic curated corpus</b>: 75 labelled documents (academic / news / forum) across 18 topics (8 dev, 10 held-out), with gold key facts and adjudicated conflicts. <b>50 synthetic episodic sessions</b> are built from dev data only, and held-out data never enters memory.</p>
<h2>4. Task 1 — Prompting strategies</h2>{table(tech_rows, fmt={"$ / 1k queries": "{:.3f}"})}{img('fig1_task1', 'Figure 3 – Correctness, depth vs latency, depth vs sources', 100)}{sec('<!--T1_INFERENCE-->')}
<h2>5. Task 2 — Multi-source function calling, safety & security</h2>
<p>Tool-selection accuracy {100*t2.get('tool_selection_accuracy', float('nan')):.0f}%, task completion {100*t2.get('task_completion', float('nan')):.0f}%, parallel latency reduction {t2.get('parallel_reduction_pct', float('nan')):.0f}%, defence success rate {t2.get('defense_success_rate', float('nan')):.0f}% with a {t2.get('false_positive_rate', float('nan')):.0f}% false-positive rate on benign inputs.</p>
{img('fig2_3_react_flow', 'Figure 4 – ReAct flow (real trace)', 80)}{img('fig2_security', 'Figure 5 – Malicious inputs blocked per defence', 90)}{img('fig2_tools_timing_failures', 'Figure 6 – Tool usage, sequential vs parallel, failure modes', 100)}{sec('<!--T2_INFERENCE-->')}
<h2>6. Task 3 — Memory, decomposition, retrieval, context engineering, advanced RAG</h2>
{img('fig3_chunking', 'Figure 7 – Chunking strategy × overlap', 95)}{img('fig3_3_recall10', 'Figure 8 – Recall@10 by retrieval configuration', 70)}{img('fig3_4_coverage_iteration', 'Figure 9 – Evidence coverage vs refinement iteration', 70)}
{img('fig3_context', 'Figure 10 – Context token breakdown and growth over 10 turns', 100)}{img('fig3_7_lim', 'Figure 11 – Lost in the middle', 65)}{img('fig3_8_memory', 'Figure 12 – Memory ablation', 75)}{img('fig3_variants', 'Figure 13 – HyDE / CRAG vs baseline', 90)}{sec('<!--T3_INFERENCE-->')}
<h2>7. Task 4 — Synthesis, conflicts, hallucination, calibration, RAG Triad</h2>
{img('fig4_1_crr_weight', 'Figure 14 – Conflict resolution vs credibility weight', 70)}{img('fig4_halluc_calibration', 'Figure 15 – Detector confusion matrix, hallucination reduction, calibration', 100)}{img('fig4_sources', 'Figure 16 – Quality vs number of sources', 100)}{img('fig4_7_triad', 'Figure 17 – RAG Triad', 60)}{sec('<!--T4_INFERENCE-->')}
<h2>8. Task 5 — Self-Instruct, DPO, PEFT, alignment, paradigms</h2>
<p>Self-Instruct: {t5.get('si_generated')} generated → {t5.get('si_kept')} kept after filtering (removed: {t5.get('si_removed')}). DPO loss = {t5.get('dpo_loss', float('nan')):.4f}. LoRA trainable parameters = {t5.get('lora_trainable_pct', float('nan')):.3f}% of 7B. 70B memory (P×B×1.2): {t5.get('mem70b')}.</p>
{img('fig5_selfinstruct', 'Figure 18 – Self-Instruct task types and filtering', 100)}{img('fig5_peft', 'Figure 19 – PEFT memory', 90)}{img('fig5_6_paradigms', 'Figure 20 – Prompting vs RAG vs fine-tuning', 100)}{img('fig5_7_decision_tree', 'Figure 21 – Decision tree', 85)}{sec('<!--T5_INFERENCE-->')}
<h2>9. Task 6 — Integration, multi-turn, audit trail, evaluation</h2>
{img('fig6_1_multiturn_flow', 'Figure 22 – Multi-turn flow', 95)}{img('fig6_3_audit', 'Figure 23 – Audit trail', 100)}{img('fig6_cost', 'Figure 24 – Cost and tokens per query', 100)}{sec('<!--T6_INFERENCE-->')}
<h2>10. Task 7 — Architecture decisions & cost engineering</h2>{table(t7.get('decisions', []))}
{table(t7.get('routing', []), fmt={"Cost / query ($)": "{:.5f}", "Monthly cost ($)": "{:,.2f}", "Savings vs A ($/month)": "{:,.2f}", "Reduction (%)": "{:.1f}"})}
{table(t7.get('optimisations', []), fmt={"Baseline A ($)": "{:,.2f}", "Monthly cost ($)": "{:,.2f}", "Savings ($)": "{:,.2f}", "Reduction (%)": "{:.1f}"})}
{img('fig7_dashboard', 'Figure 25 – Model comparison dashboard', 100)}{sec('<!--T7_RECOMMENDATION-->')}{sec('<!--T7_INFERENCE-->')}
<h2>11. Conclusion, limitations and future work</h2>{sec('<!--CONCLUSION-->')}
<h2>12. References</h2>
<p>Kwiatkowski et al. (2019) Natural Questions, TACL · Wei et al. (2022) CoT · Yao et al. (2023) ToT; ReAct · Zhou et al. (2023) APE · Cormack et al. (2009) RRF · Reimers &amp; Gurevych (2019) SBERT · Johnson et al. (2019) FAISS · Liu et al. (2023) Lost in the Middle · Gao et al. (2023) HyDE · Yan et al. (2024) CRAG · Wang et al. (2023) Self-Instruct · Rafailov et al. (2023) DPO · Hu et al. (2022) LoRA · Dettmers et al. (2023) QLoRA · Shao et al. (2024) GRPO · Bai et al. (2022) Constitutional AI · Chen et al. (2024) SPIN · Lightman et al. (2023) PRM · Guo et al. (2017) Calibration · Zheng et al. (2023) LLM-as-a-Judge · Google Gemini API documentation (Nov 2026) · AIMLCZG521 course sessions 4–12. The full list is in the notebook.</p>
</body></html>"""
    out = ROOT / "report" / "Final_Report.html"
    out.write_text(html, encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    from export_pdf import to_pdf

    print("PDF:", to_pdf(out))


if __name__ == "__main__":
    main()
