"""Global configuration: seeds, paths, model roles, pricing and hyper-parameters.

Every hyper-parameter used anywhere in the project lives here so that the
notebook can print the full configuration (assignment rule: "All
hyperparameters and decisions must be documented").
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # torch optional
        pass


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(os.environ.get("RA_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
CURATED_DIR = DATA_DIR / "curated"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
CACHE_DIR = ROOT / "cache"
for _d in (DATA_DIR, OUT_DIR, FIG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

try:  # load .env BEFORE reading LLM_BACKEND / API keys
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

# Reference "today" used for recency / source-freshness computations.
REFERENCE_DATE = "2026-11-01"

# ----------------------------------------------------------------------------
# LLM backend & model roles
# ----------------------------------------------------------------------------
# Each role has an ordered preference list. The first model that is available
# (and not rate-limit-exhausted) serves the call. The model that actually
# served each call is logged so results are always attributable.
BACKEND = os.environ.get("LLM_BACKEND", "groq")

MODEL_ROLES: dict[str, list[str]] = {
    # Small, fast model that does the research work (generation/synthesis).
    # Llama-4-Scout first: on the Groq free tier it has ~5x the tokens-per-minute of the 8B model.
    "generator": [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
    ],
    # Larger, *different* model used as LLM-as-a-Judge (reduces self-preference bias).
    "judge": [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
    ],
    # High tokens-per-minute model for the 10K+ token "lost in the middle" test.
    "long_context": [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ],
    # Tiers used in the Task 7 model-routing / Pareto experiment.
    "tier_small": ["llama-3.1-8b-instant"],
    "tier_moe": ["meta-llama/llama-4-scout-17b-16e-instruct", "openai/gpt-oss-20b"],
    "tier_dense": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
}

OPENAI_MODEL_ROLES: dict[str, list[str]] = {
    "generator": ["gpt-4o-mini"],
    "judge": ["gpt-4o"],
    "long_context": ["gpt-4o-mini"],
    "tier_small": ["gpt-4o-mini"],
    "tier_moe": ["gpt-4o-mini"],
    "tier_dense": ["gpt-4o"],
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Google Gemini through its OpenAI-compatible endpoint. Candidate lists are filtered at run time against the models
# the key can actually see (client.models.list()), so newer names are tried first and older ones act as fallbacks.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_ROLES: dict[str, list[str]] = {
    # gemini-3.1-flash-lite: fastest model on the free tier (2.5-series models are closed to new keys; Pro = quota 0).
    "generator": ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash"],
    # Judge = a larger, *different* model than the generator.
    "judge": ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"],
    "long_context": ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash"],
    "tier_small": ["gemini-3.1-flash-lite"],
    "tier_moe": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
    "tier_dense": ["gemini-3.5-flash", "gemini-3.8-flash"],
}
# Mixed providers (free tiers): Gemini generates, Groq judges (different model family -> less self-preference bias).
# Each entry is "provider:model"; the first available, non-exhausted entry serves the call.
MIXED_MODEL_ROLES: dict[str, list[str]] = {
    "generator": ["gemini:gemini-3.1-flash-lite", "groq:openai/gpt-oss-20b", "groq:qwen/qwen3.8-27b", "gemini:gemini-3.5-flash-lite"],
    "judge": ["groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b", "gemini:gemini-3.5-flash-lite"],
    "long_context": ["gemini:gemini-3.1-flash-lite", "gemini:gemini-3.5-flash-lite", "groq:openai/gpt-oss-20b"],
    "tier_small": ["groq:openai/gpt-oss-20b"],
    "tier_moe": ["gemini:gemini-3.1-flash-lite"],
    "tier_dense": ["groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b"],
}
# Local model served on this PC (Intel Arc A770, 16 GB) by LM Studio / Ollama / llama.cpp through an
# OpenAI-compatible endpoint. LOCAL_MODEL="auto" picks the first chat model the server reports as available.
#   LM Studio : LOCAL_BASE_URL=http://localhost:1234/v1   (default)
#   Ollama    : LOCAL_BASE_URL=http://localhost:11434/v1
#   llama.cpp : LOCAL_BASE_URL=http://localhost:8080/v1
LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://localhost:1234/v1")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "auto")
LOCAL_TIMEOUT_S = float(os.environ.get("LOCAL_TIMEOUT_S", "600"))  # local generation of long prompts can be slow
LOCAL_MODEL_ROLES: dict[str, list[str]] = {r: [f"local:{LOCAL_MODEL}"] for r in
                                           ("generator", "judge", "long_context", "tier_small", "tier_moe", "tier_dense")}
# Hybrid (recommended with a local GPU): the local model does the bulk generation work with no rate limits; the judge
# stays a large model of a *different* family on Groq (less self-preference bias, credible RAG-Triad scores); the
# 10K+ token lost-in-the-middle test goes to Gemini (1M context window) because local context is usually 8-32K.
HYBRID_MODEL_ROLES: dict[str, list[str]] = {
    "generator": [f"local:{LOCAL_MODEL}", "gemini:gemini-3.1-flash-lite", "groq:openai/gpt-oss-20b"],
    "judge": ["groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b", "gemini:gemini-3.5-flash-lite"],
    "long_context": ["gemini:gemini-3.1-flash-lite", "gemini:gemini-3.5-flash-lite", f"local:{LOCAL_MODEL}"],
    "tier_small": [f"local:{LOCAL_MODEL}", "groq:openai/gpt-oss-20b"],
    "tier_moe": ["gemini:gemini-3.1-flash-lite", "groq:openai/gpt-oss-20b"],
    "tier_dense": ["groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b", "gemini:gemini-3.5-flash"],
}
# Device for the embedding model + cross-encoder: "cpu" (default, bit-identical to the cached run), "xpu" (Intel Arc
# with a PyTorch XPU build), "cuda", or "auto". Both models are tiny, so CPU is fast enough.
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")


def torch_device() -> str:
    if EMBED_DEVICE != "auto":
        return EMBED_DEVICE
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# "none" disables thinking where the model allows it; Pro models cannot disable thinking, so use "low".
GEMINI_REASONING_EFFORT: dict[str, str] = {}  # default "low" (minimum accepted by Gemini 3.x)
GEMINI_THINKING_HEADROOM = 1024
# Free-tier requests-per-minute pacing (seconds between calls per model); adjust if you are on a paid tier.
MIN_SECONDS_BETWEEN_CALLS = {
    "gemini-3.1-flash-lite": 4.1, "gemini-3.5-flash-lite": 4.1, "gemini-3.5-flash": 6.1, "gemini-3.8-flash": 6.1,
}

# ----------------------------------------------------------------------------
# Pricing (assignment-specified; used for ALL cost numbers so they are comparable)
# Source: assignment PDF, Task 6.5 and Task 7 Part B (GPT-4o list price as given).
# ----------------------------------------------------------------------------
PRICING = {  # USD per 1M tokens
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "small_lm": {"input": 0.10, "output": 0.30},
    "moe_mid": {"input": 0.60, "output": 1.10},
}
TIER_TO_PRICE = {"tier_small": "small_lm", "tier_moe": "moe_mid", "tier_dense": "gpt-4o"}


def cost_usd(input_tokens: float, output_tokens: float, price_key: str = "gpt-4o") -> float:
    p = PRICING[price_key]
    return input_tokens / 1e6 * p["input"] + output_tokens / 1e6 * p["output"]


# ----------------------------------------------------------------------------
# Retrieval / memory hyper-parameters
# ----------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOKENIZER_ENCODING = "o200k_base"  # GPT-4o tokenizer, matches the pricing assumption

RRF_K = 60  # standard constant from Cormack et al. (2009)
TOP_K_PER_SOURCE = 5
WORKING_MEMORY_N = 10
EPISODIC_TOP_K = 3
CHUNK_SIZE_DEFAULT = 256
CHUNK_OVERLAP_DEFAULT = 0.2
MAX_REFINEMENT_ITERS = 3

# Agent safety limits (Task 2)
MAX_ITERATIONS = 10
REPETITION_LIMIT = 3
CONSECUTIVE_FAILURE_LIMIT = 3
AGENT_TIMEOUT_S = 60.0

# Generation defaults
GEN_TEMPERATURE = 0.0
TOT_TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 700
OUTPUT_TOKEN_BUDGET = 800  # reserved in the context-window budget


def describe() -> dict:
    """Return all hyper-parameters as a dict (printed in the notebook)."""
    return {
        "SEED": SEED,
        "BACKEND": BACKEND,
        "MODEL_ROLES": {"groq": MODEL_ROLES, "openai": OPENAI_MODEL_ROLES, "gemini": GEMINI_MODEL_ROLES, "mixed": MIXED_MODEL_ROLES,
                        "local": LOCAL_MODEL_ROLES, "hybrid": HYBRID_MODEL_ROLES}.get(BACKEND, MODEL_ROLES),
        "LOCAL_BASE_URL": LOCAL_BASE_URL if BACKEND in ("local", "hybrid") else None,
        "EMBED_DEVICE": EMBED_DEVICE,
        "EMBED_MODEL": EMBED_MODEL,
        "CROSS_ENCODER_MODEL": CROSS_ENCODER_MODEL,
        "TOKENIZER_ENCODING": TOKENIZER_ENCODING,
        "RRF_K": RRF_K,
        "TOP_K_PER_SOURCE": TOP_K_PER_SOURCE,
        "WORKING_MEMORY_N": WORKING_MEMORY_N,
        "EPISODIC_TOP_K": EPISODIC_TOP_K,
        "CHUNK_SIZE_DEFAULT": CHUNK_SIZE_DEFAULT,
        "CHUNK_OVERLAP_DEFAULT": CHUNK_OVERLAP_DEFAULT,
        "MAX_REFINEMENT_ITERS": MAX_REFINEMENT_ITERS,
        "MAX_ITERATIONS": MAX_ITERATIONS,
        "REPETITION_LIMIT": REPETITION_LIMIT,
        "CONSECUTIVE_FAILURE_LIMIT": CONSECUTIVE_FAILURE_LIMIT,
        "AGENT_TIMEOUT_S": AGENT_TIMEOUT_S,
        "GEN_TEMPERATURE": GEN_TEMPERATURE,
        "TOT_TEMPERATURE": TOT_TEMPERATURE,
        "MAX_OUTPUT_TOKENS": MAX_OUTPUT_TOKENS,
        "PRICING_USD_PER_1M": PRICING,
        "REFERENCE_DATE": REFERENCE_DATE,
    }
