from common import code, md, module

CELLS = [
md(r"""
# AIMLCZG521 – Conversational AI · Assignment 2 · Problem Statement 1
## End-to-End Research Assistant with Multi-Source Synthesis and Alignment Analysis

**Course:** Conversational AI (AIMLCZG521) — EC-1 Assessment · **Total marks:** 10

### Student details & contribution areas
> ✏️ *Fill in before submission. Every member must own ≥ 2 contribution areas, and each claimed area must point to the notebook sections that evidence it.*

| # | Name | BITS ID | Contribution areas (≥ 2) | Evidence (notebook sections) |
|---|------|---------|--------------------------|------------------------------|
| 1 | _Name_ | _ID_ | Prompt Engineering & Multi-Step Reasoning; Paradigm Differentiation | Task 1; Task 5 Part C |
| 2 | _Name_ | _ID_ | Multi-Source Function Calling & Security; Integration, Multi-Turn, Audit Trail & Evaluation | Task 2; Task 6 |
| 3 | _Name_ | _ID_ | Memory, Query Decomposition, Retrieval & Advanced RAG; RAG Architecture Decisions & Cost Engineering | Task 3; Task 7 |
| 4 | _Name_ | _ID_ | Multi-Source Synthesis, Calibration & RAG Triad; Instruction Tuning, RLHF, PEFT & Alignment | Task 4; Task 5 Parts A–B |
| 5 | _Name_ | _ID_ | Visualization & Reporting; Integration, Multi-Turn, Audit Trail & Evaluation | All figures; Final report |

### Problem statement
Build and evaluate an **end-to-end research assistant**. It synthesises information from multiple sources, reasons about conflicting evidence, remembers past research sessions, and aligns with user preferences:

**Research Question → Query Decomposition → Multi-Source Function Calling → Evidence Gathering → Conflict Resolution → Synthesis → Citation Generation → Memory Update → Alignment & Efficiency Analysis**

### How this notebook is organised
| Section | Content | Marks |
|---|---|---|
| 0 | Setup, data, splits, reproducibility | – |
| Task 1 | Multi-step reasoning strategy evaluation (Zero-shot, Few-shot, CoT, ToT, APE) | 1.0 |
| Task 2 | Multi-source function calling, ReAct, parallel calls, safety & security | 1.0 |
| Task 3 | Memory, decomposition, iterative refinement, hybrid retrieval, context engineering, HyDE / CRAG | 2.0 |
| Task 4 | Synthesis, conflict resolution, citations, hallucination detection, calibration, RAG Triad | 1.5 |
| Task 5 | Self-Instruct, DPO, PEFT, GRPO & alignment survey, paradigm differentiation | 2.0 |
| Task 6 | `ResearchAssistant` integration, multi-turn, audit trail, evaluation, production architecture | 1.5 |
| Task 7 | Architecture decision table, cost engineering, model routing, final recommendation | 1.0 |
| End | Conclusion, limitations, improvements, references | – |

Every task follows the same layout: **objective → design & justification → implementation → results & figures → inference**.
"""),
md(r"""
## 0. Setup

### 0.1 Tools & libraries
| Purpose | Library / model | Role (black-box or from scratch) |
|---|---|---|
| LLM inference | Two free-tier providers through their OpenAI-compatible endpoints (`openai` SDK). **Generator: Google gemini-3.1-flash-lite.** **Judge: openai/gpt-oss-120b on Groq**, a different model family, which reduces self-preference bias. Fallbacks are listed in `ra/config.py` (`MIXED_MODEL_ROLES`); the table at the end shows which model served each call | API only. All prompting, agents, memory, synthesis and evaluation logic is ours |
| Token counting | `tiktoken` (`o200k_base`, the GPT-4o tokenizer) | Counting only |
| Lexical retrieval | `rank-bm25` (BM25Okapi) | Allowed black box |
| Dense retrieval | `sentence-transformers/all-MiniLM-L6-v2` + `faiss-cpu` (IndexFlatIP) | Allowed black box |
| Fusion | Reciprocal Rank Fusion, a few lines of our code | Allowed (formula) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Allowed black box |
| Validation | `pydantic` v2 | Library; schemas and rules are ours |
| Data / plots | `datasets`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `networkx` | Utilities |

**No agent framework is used** (no LangChain or LlamaIndex agents). The ReAct loop, memory, decomposition, refinement, HyDE/CRAG, context engineering, conflict resolution, citation, verification, calibration, RAG Triad, Self-Instruct, DPO/PEFT analysis, security and cost engineering are all implemented in the `ra/` package. Each module is written out below with `%%writefile`, so this notebook is self-contained.

### 0.2 Reproducibility
* **Seed = 42** for Python, NumPy, torch, data sampling, and simulated tool latency/failures. Gemini's OpenAI-compatible endpoint does not accept a `seed` parameter, so LLM reproducibility comes from temperature 0 plus the response cache below.
* **LLM response cache.** Every LLM call is cached in SQLite, keyed on (model, messages, temperature, max_tokens, json-mode, seed). The cache stores the *original* latency and token usage. Re-running the notebook replays identical outputs and reports the latencies that were measured live.
* **Role-based model selection with fallback.** If a free-tier daily limit is exhausted, the next model in the role's list is used. Every call logs which model served it (see the usage table at the end).
* All hyper-parameters live in `ra/config.py` and are printed below.
"""),
code(r"""
import os, sys, json, time, random, warnings, textwrap, re
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.makedirs("ra", exist_ok=True); os.makedirs("outputs/figures", exist_ok=True); os.makedirs("cache", exist_ok=True)
print("Python", sys.version.split()[0], "| working dir:", os.getcwd())
"""),
module("__init__.py"),
module("config.py"),
module("tokens.py"),
module("llm.py"),
code(r"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 200)
from ra import config
from ra.config import set_seed, SEED
from ra.llm import LLM, ScriptedLLM, parse_json
set_seed(SEED)
llm = LLM(verbose=True)
print("Backend:", llm.backend)
print("Resolved model per role:", json.dumps(llm.resolved_roles(), indent=1))
print(json.dumps({k: v for k, v in config.describe().items() if k != "MODEL_ROLES"}, indent=1))
RESULTS = {}          # key numbers collected across tasks (used by Task 7 and the final report)
def save_results():
    json.dump(RESULTS, open("outputs/results.json", "w"), indent=1, default=float)
"""),
md(r"""
### 0.3 Dataset details & source

**Primary dataset: Natural Questions (NQ).** Kwiatkowski et al. (2019), Google Research. Licence **CC-BY-SA-3.0**.
* Questions + gold short answers come from [`google-research-datasets/nq_open`](https://huggingface.co/datasets/google-research-datasets/nq_open).
* Gold Wikipedia passages come from [`sentence-transformers/natural-questions`](https://huggingface.co/datasets/sentence-transformers/natural-questions).
* The two are joined on question text. We keep pairs whose gold answer appears in the passage, filter to the **Science, Health & Economics** domain with a whole-word keyword list, and sample **1,000** examples (seed 42).
* **Modification notice:** the dataset was subsetted, domain-filtered and joined across two HF mirrors, and passages were truncated to 300 words.

**Additional sources (needed for multi-source conflict scenarios):**
1. **Real Wikipedia articles** (CC-BY-SA) for 18 research topics + 10 extra articles: long-context distractors and the inflation conversation. Fetched once through the MediaWiki API.
2. **Curated research corpus (SYNTHETIC, clearly labelled):** 75 hand-written documents across 18 topics. There are 43 academic abstracts, 28 news articles and 4 forum posts, with fictional authors, venues and `.example` URLs. Credible sources follow published scientific consensus. Low-credibility sources (tabloids, blogs, industry PR, forums) deliberately make misleading claims, so the sources genuinely **conflict**. Two news items carry **indirect prompt-injection** payloads for the security tests. Every topic has **gold key facts** with supporting-document labels and **adjudicated conflicts** (gold winning position or "contested").
3. **50 SYNTHETIC past research sessions** for episodic memory. They are built only from dev topics and the NQ `episodic` split, and each is marked `"synthetic": true`.

`data/build_data.py` regenerates every file. The outputs are shipped with the notebook, so execution does not depend on the network.
"""),
module("chunking.py"),
module("retrieval.py"),
module("data.py"),
module("evaluation.py"),
module("credibility.py"),
module("viz.py"),
code(r"""
from pathlib import Path
if not Path("data/nq_subset.json").exists():          # regenerate only if the shipped data is missing
    import subprocess; subprocess.run([sys.executable, "data/build_data.py"], check=True)
from ra.data import ResearchData
from ra.retrieval import Embedder, MultiSourceRetriever, HybridIndex, CrossEncoderReranker
from ra import viz
from ra.viz import save, bar_labels, PALETTE
D = ResearchData.load()
emb = Embedder()
print(f"NQ examples: {len(D.nq)} | documents: {len(D.docs)} | topics: {len(D.topics)} | synthetic episodic sessions: {len(D.sessions)}")
src_counts = pd.Series([d['source'] + (' (synthetic)' if d.get('synthetic') else ' (real)') for d in D.docs.values()]).value_counts()
display(src_counts.rename("documents").to_frame())
display(Markdown("**Table 0.1 – Documented dataset splits (the held-out data is never used as episodic memory or for tuning).**"))
display(D.split_table())
"""),
code(r"""
# Example records: one NQ item, one synthetic academic doc, one synthetic tabloid item, one gold conflict
r = D.nq_split("heldout")[0]
print("NQ example:", r["question"], "->", r["answers"], "| passage:", r["passage"][:150], "...")
for did in ["D1-A1", "D1-N2"]:
    d = D.docs[did]; print(f"\n{did} [{d['source']}/{d['outlet_type']}, synthetic={d['synthetic']}] {d['title']}\n  {d['text'][:220]}...")
print("\nGold conflict:", json.dumps(D.topics["D1"]["conflicts"][0], indent=1)[:600])
"""),
code(r"""
# Default knowledge collections (recursive chunking, 256 tokens, 0.2 overlap; Task 3 re-selects the strategy experimentally)
t0 = time.time()
CH = D.build_chunks("recursive", 256, 0.2, embedder=emb)
R = MultiSourceRetriever(CH, role="researcher")
emb.save()
print({k: len(v) for k, v in CH.items()}, f"chunks indexed in {time.time()-t0:.1f}s")
"""),
md(r"""
### 0.4 System overview
The diagram below is the target architecture that Tasks 1–6 build up piece by piece. Task 6 assembles it into one `ResearchAssistant` class.
"""),
code(r"""
N = {
 "q": (1.2, 7, "User research\nquestion", PALETTE[0]), "sec": (3.6, 7, "Security gate\n(Pydantic, RBAC)", PALETTE[3]),
 "route": (6.0, 7, "Adaptive router\n(new / follow-up / reformulate)", PALETTE[4], 2.6), "epi": (9.0, 7, "Episodic memory\n(FAISS, top-2)", PALETTE[2]),
 "dec": (1.2, 5, "CoT query\ndecomposition", PALETTE[0]), "tools": (3.9, 5, "Parallel tool calls\nacademic | wiki | news", PALETTE[1], 2.6),
 "ref": (6.6, 5, "Iterative\nrefinement (≤3)", PALETTE[0]), "fuse": (9.3, 5, "Credibility-weighted\nRRF + cross-encoder", PALETTE[1], 2.6),
 "ctx": (1.2, 3, "Context\nengineering", PALETTE[4]), "conf": (3.9, 3, "Conflict detection\n& resolution", PALETTE[3], 2.4),
 "syn": (6.6, 3, "Cited synthesis", PALETTE[0]), "ver": (9.3, 3, "Claim verification\n+ self-correction", PALETTE[3], 2.4),
 "cal": (1.2, 1, "Confidence\ncalibration", PALETTE[2]), "cit": (3.9, 1, "Citation\ngeneration (APA)", PALETTE[2]),
 "mem": (6.6, 1, "Memory update\n(working + episodic)", PALETTE[2], 2.4), "aud": (9.3, 1, "Audit trail (JSON)\n+ evaluation", PALETTE[5], 2.4),
}
E = [("q","sec",""),("sec","route",""),("route","epi",""),("epi","dec",""),("dec","tools",""),("tools","ref",""),("ref","tools","gaps","--"),
     ("ref","fuse",""),("fuse","ctx",""),("ctx","conf",""),("conf","syn",""),("syn","ver",""),("ver","cal",""),("cal","cit",""),("cit","mem",""),("mem","aud","")]
fig, ax = viz.diagram(N, E, "Figure 0.1 – End-to-end research assistant pipeline", figsize=(13, 7.5), xlim=(0, 10.6), ylim=(0.2, 7.8))
save(fig, "fig0_1_pipeline"); plt.show()
"""),
]
