"""LLM client: one thin wrapper around any OpenAI-compatible chat endpoint.

Why a custom wrapper instead of a framework?
* We need *exact* token usage and latency for every call (Tasks 1, 3, 6, 7).
* We need a persistent response cache so that the executed notebook is
  reproducible and re-runs cost nothing.
* We need role-based model selection (generator vs judge vs long-context) with
  automatic fallback when a free-tier rate limit is exhausted.

The cache stores the *original* latency and token counts of each call, so a
cached replay reports the same numbers that were measured live.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import config
from .tokens import count_messages, count_tokens

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv(config.ROOT / ".env")
except Exception:
    pass


class LocalOutputCorrupt(Exception):
    """The local model produced garbage (e.g. only '?' characters) - a GPU/runtime fault, not a real answer."""


def _looks_corrupt(text: str) -> bool:
    """Garbage from a faulted GPU backend: mostly '?' / U+FFFD, or one character repeated (e.g. 'GGGGGGGG...')."""
    t = text.strip()
    if not t:
        return False  # an empty answer is handled by the callers' defaults
    if sum(ch in "?�" for ch in t) / len(t) > 0.5:
        return True
    # 40+ identical characters in a row (markdown rules such as ---- or ==== are allowed)
    return re.search(r"([^\s\-=_*#.|~])\1{39,}", t) is not None


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    cached: bool = False
    purpose: str = ""

    @property
    def cost_gpt4o(self) -> float:
        return config.cost_usd(self.input_tokens, self.output_tokens, "gpt-4o")


@dataclass
class CallRecord:
    purpose: str
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def as_list(v: Any) -> list:
    """Judges sometimes return a scalar where a list was asked for (e.g. {"supports": true} for a single pair)."""
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


_VALUE_KEYS = ("correct", "is_correct", "supported", "supports", "support", "label", "verdict", "value", "answer",
               "score", "rating")


def item_value(v: Any) -> Any:
    """Judges sometimes return items as objects, e.g. {"claim": 1, "correct": true}: take the value field."""
    if isinstance(v, dict):
        for k in _VALUE_KEYS:
            if k in v:
                return v[k]
        return next((x for x in v.values() if isinstance(x, (bool, int, float))), None)
    return v


def truthy(v: Any) -> bool:
    """bool() of a JSON value, but "false"/"no"/"0" strings count as False and {"correct": false} items are read."""
    v = item_value(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "y", "1", "supported", "correct")
    return bool(v)


def list_field(data: Any, key: str) -> list:
    """The list the caller asked for under `key`, whether the model returned {"key": [...]}, a bare list (kept by
    chat_json under "_list"), or a single scalar."""
    if isinstance(data, dict):
        v = data.get(key)
        return as_list(v if v is not None else data.get("_list"))
    return as_list(data)


def parse_json(text: str, default: Any = None) -> Any:
    """Robustly extract the first JSON object/array from an LLM response."""
    if text is None:
        return default
    t = _THINK_RE.sub("", text).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.M)
    try:
        return json.loads(t)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = t.find(open_c)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except Exception:
                        break
    return default


def _parse_reset(s: str | None) -> float:
    """Parse Groq reset headers such as '7.66s', '2m59.56s', '120ms'."""
    if not s:
        return 0.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", s):
        v = float(num)
        total += {"ms": v / 1000, "s": v, "m": v * 60, "h": v * 3600}[unit]
    return total


class LLM:
    """Role-aware, cached, rate-limit-aware chat client."""

    def __init__(self, backend: str | None = None, cache_path=None, verbose: bool = False):
        self.backend = backend or config.BACKEND
        self.verbose = verbose
        self.calls: list[CallRecord] = []
        self._lock = threading.Lock()
        self._exhausted: set[str] = set()
        self._limits: dict[str, dict[str, float]] = {}
        cache_path = cache_path or (config.CACHE_DIR / "llm_cache.sqlite")
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, payload TEXT)"
        )
        self._db.commit()
        self.clients: dict = {}
        self.available_models: set[str] | None = None
        self._last_call: dict[str, float] = {}
        if self.backend == "mock":
            self.roles = {r: [f"mock-{r}"] for r in config.MODEL_ROLES}
            return
        roles = {"groq": config.MODEL_ROLES, "openai": config.OPENAI_MODEL_ROLES, "gemini": config.GEMINI_MODEL_ROLES,
                 "mixed": config.MIXED_MODEL_ROLES, "local": config.LOCAL_MODEL_ROLES,
                 "hybrid": config.HYBRID_MODEL_ROLES}.get(self.backend)
        if roles is None:
            raise ValueError(f"Unknown backend {self.backend}")
        self.roles = {r: list(c) for r, c in roles.items()}  # copy: "local:auto" is rewritten below
        providers = sorted({self._split(c)[0] for cands in self.roles.values() for c in cands})
        from openai import OpenAI

        avail = set()
        for prov in providers:
            if not self._has_key(prov):
                continue
            if prov == "groq":
                cl = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=config.GROQ_BASE_URL, max_retries=0, timeout=120)
            elif prov == "gemini":
                key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                cl = OpenAI(api_key=key, base_url=config.GEMINI_BASE_URL, max_retries=0, timeout=180)
            elif prov == "local":  # LM Studio / Ollama / llama.cpp on this PC; any key is accepted
                cl = OpenAI(api_key=os.environ.get("LOCAL_API_KEY", "local"), base_url=config.LOCAL_BASE_URL,
                            max_retries=0, timeout=config.LOCAL_TIMEOUT_S)
            else:
                cl = OpenAI(max_retries=0, timeout=120)
            try:
                ids = [m.id.replace("models/", "") for m in cl.models.list().data]
            except Exception as e:  # offline / bad key: rely on cache only
                if verbose:
                    print(f"[llm] could not list {prov} models:", e)
                if prov == "local":  # local server not running: skip it so calls fall back to the cloud models
                    continue
                ids = []
            if prov == "local":
                self._resolve_local_model(ids)
            self.clients[prov] = cl
            avail |= {prov + ":" + i for i in ids}
        self.available_models = avail or None

    # ------------------------------------------------------------------ utils
    def _resolve_local_model(self, ids: list[str]) -> None:
        """Replace 'local:auto' with the first chat model the local server reports (embedding models are skipped)."""
        chat_ids = [i for i in ids if "embed" not in i.lower()]
        if config.LOCAL_MODEL != "auto" or not chat_ids:
            return
        for r, cands in self.roles.items():
            self.roles[r] = [f"local:{chat_ids[0]}" if c == "local:auto" else c for c in cands]
        if self.verbose:
            print(f"[llm] local server model: {chat_ids[0]}")

    def _split(self, cand: str) -> tuple[str, str]:
        """'groq:llama-3.3-70b' -> ('groq', 'llama-3.3-70b'); bare names belong to the single backend."""
        if ":" in cand and cand.split(":", 1)[0] in ("groq", "gemini", "openai", "local"):
            p, m = cand.split(":", 1)
            return p, m
        return self.backend, cand

    @staticmethod
    def _has_key(provider: str) -> bool:
        env = {"groq": ["GROQ_API_KEY"], "openai": ["OPENAI_API_KEY"], "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"]}
        if provider == "local":
            return True  # no key needed; reachability is checked when the client is created
        return any(os.environ.get(k) for k in env.get(provider, []))

    def models_for(self, role: str) -> list[str]:
        models = self.roles.get(role, self.roles["generator"])
        if self.backend != "mock":
            models = [m if ":" in m.split("/")[0] else f"{self.backend}:{m}" for m in models]
        if self.available_models:
            avail = [m for m in models if m in self.available_models]
            models = avail or models
        return models

    def resolved_roles(self) -> dict[str, str]:
        return {r: self.models_for(r)[0] for r in self.roles}

    @staticmethod
    def _key(model: str, messages, temperature, max_tokens, json_mode, seed) -> str:
        blob = json.dumps(
            {"m": model, "msgs": messages, "t": temperature, "mt": max_tokens, "j": json_mode, "s": seed},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str):
        with self._lock:
            row = self._db.execute("SELECT payload FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, key: str, model: str, payload: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (key, model, payload) VALUES (?,?,?)",
                (key, model, json.dumps(payload, ensure_ascii=False)),
            )
            self._db.commit()

    def _pace(self, model: str, est_tokens: int) -> None:
        """Pro-active pacing using the rate-limit headers of the previous call."""
        gap = config.MIN_SECONDS_BETWEEN_CALLS.get(model, 0.0)
        if gap:
            wait = self._last_call.get(model, 0.0) + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call[model] = time.time()
        lim = self._limits.get(model)
        if not lim:
            return
        elapsed = time.time() - lim["t"]
        if lim["remaining_tokens"] < est_tokens and elapsed < lim["reset_tokens"]:
            time.sleep(min(lim["reset_tokens"] - elapsed + 0.5, 65))
        if lim["remaining_requests"] < 1 and elapsed < lim["reset_requests"]:
            time.sleep(min(lim["reset_requests"] - elapsed + 0.5, 65))

    # ------------------------------------------------------------------- main
    def chat(
        self,
        messages: list[dict] | str,
        role: str = "generator",
        temperature: float = config.GEN_TEMPERATURE,
        max_tokens: int = config.MAX_OUTPUT_TOKENS,
        json_mode: bool = False,
        seed: int = config.SEED,
        purpose: str = "",
        system: str | None = None,
    ) -> LLMResponse:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if system:
            messages = [{"role": "system", "content": system}] + messages
        candidates = self.models_for(role)

        # 1) cache lookup across the role's candidate models (in preference order)
        for cand in candidates:  # cache key uses the bare model name
            hit = self._cache_get(self._key(self._split(cand)[1], messages, temperature, max_tokens, json_mode, seed))
            if hit:
                resp = LLMResponse(**{**hit, "cached": True, "purpose": purpose})
                self._record(resp, role)
                return resp

        if self.backend == "mock":
            return self._mock(messages, candidates[0], json_mode, purpose, role)
        if os.environ.get("LLM_CACHE_ONLY") == "1":  # replay audit: log the miss, answer with the mock model
            import traceback
            with open(os.environ.get("LLM_MISS_LOG", "cache_misses.log"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"role": role, "purpose": purpose, "model": candidates[0],
                                    "stack": traceback.format_stack(limit=6)[:-1], "messages": messages},
                                   ensure_ascii=False) + "\n")
            return self._mock(messages, candidates[0], json_mode, purpose, role)

        if not self.clients:
            raise RuntimeError(
                "No API key / local server found and response not cached. Set GROQ_API_KEY / GEMINI_API_KEY in .env, "
                "start your local LLM server (LLM_BACKEND=local|hybrid), or use LLM_BACKEND=mock for an offline smoke test."
            )

        # 2) live call with fallback across candidate models
        last_err = None
        est = count_messages(messages) + max_tokens
        send_seed, send_reasoning = True, True
        for cand in candidates:
            if cand in self._exhausted:
                continue
            provider, model = self._split(cand)
            client = self.clients.get(provider)
            if client is None:
                continue
            for attempt in range(8):
                self._pace(model, est)
                try:
                    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                    if send_seed and provider != "gemini":  # Gemini's OpenAI endpoint rejects `seed`
                        kwargs["seed"] = seed
                    if provider == "gemini" and send_reasoning:
                        # Gemini 2.5+/3 "thinking" tokens count against max_tokens: keep thinking minimal and
                        # reserve extra room so the visible answer is never truncated.
                        kwargs["reasoning_effort"] = config.GEMINI_REASONING_EFFORT.get(model, "low")
                        kwargs["max_tokens"] = max_tokens + config.GEMINI_THINKING_HEADROOM
                    if json_mode:
                        if provider == "local":  # LM Studio accepts only json_schema; grammar-constrain to an object
                            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                                "name": "response", "schema": {"type": "object"}}}
                        else:
                            kwargs["response_format"] = {"type": "json_object"}
                    if "gpt-oss" in model or "qwen3" in model:
                        kwargs["max_tokens"] = max_tokens + 1024  # room for hidden reasoning
                    t0 = time.perf_counter()
                    raw = client.chat.completions.with_raw_response.create(**kwargs)
                    latency = time.perf_counter() - t0
                    completion = raw.parse()
                    h = raw.headers
                    self._limits[model] = {
                        "t": time.time(),
                        "remaining_tokens": float(h.get("x-ratelimit-remaining-tokens", 1e9) or 1e9),
                        "reset_tokens": _parse_reset(h.get("x-ratelimit-reset-tokens")),
                        "remaining_requests": float(h.get("x-ratelimit-remaining-requests", 1e9) or 1e9),
                        "reset_requests": _parse_reset(h.get("x-ratelimit-reset-requests")),
                    }
                    text = completion.choices[0].message.content or ""
                    text = _THINK_RE.sub("", text).strip()
                    if provider == "local" and _looks_corrupt(text):
                        # GPU backends (e.g. Vulkan on Intel Arc) can get into a corrupted state and emit "?????".
                        # Never cache that; retry, and stop with a clear message if it persists.
                        raise LocalOutputCorrupt(f"local model returned corrupted text: {text[:40]!r}")
                    usage = completion.usage
                    resp = LLMResponse(
                        text=text,
                        input_tokens=getattr(usage, "prompt_tokens", None) or count_messages(messages),
                        output_tokens=getattr(usage, "completion_tokens", None) or count_tokens(text),
                        latency_s=latency,
                        model=model,
                        purpose=purpose,
                    )
                    payload = asdict(resp)
                    payload.pop("cached")
                    payload.pop("purpose")
                    self._cache_put(
                        self._key(model, messages, temperature, max_tokens, json_mode, seed), model, payload
                    )
                    self._record(resp, role)
                    return resp
                except Exception as e:  # noqa: BLE001 - classify provider errors
                    last_err = e
                    if isinstance(e, LocalOutputCorrupt):
                        if attempt < 2:
                            time.sleep(5)
                            continue
                        raise RuntimeError("The local model keeps returning corrupted text. Reload it (LM Studio: "
                                           "eject + load, or `lms unload --all; lms load <model>`) and re-run; "
                                           "everything already answered is cached.") from e
                    status = getattr(e, "status_code", None)
                    msg = (str(e) + " " + json.dumps(getattr(e, "body", None) or "", default=str)).lower()
                    if status == 429:
                        retry_after = None
                        try:
                            retry_after = float(e.response.headers.get("retry-after"))
                        except Exception:
                            pass
                        if retry_after is None:
                            m = re.search(r"(?:try again|retry) in ([\dhms.]+)", msg) or re.search(r"retrydelay[\"']?\s*:\s*[\"']?([\d.]+s)", msg)
                            retry_after = _parse_reset(m.group(1)) if m else 20.0
                        if retry_after > 90 or any(t in msg for t in ("per day", "perday", "tpd", "rpd", "daily")):
                            if self.verbose:
                                print(f"[llm] {model} daily limit hit -> falling back")
                            self._exhausted.add(cand)
                            break
                        time.sleep(retry_after + 0.5)
                        continue
                    if status == 413 or "too large" in msg or "context_length" in msg:
                        break  # try the next (bigger) model
                    if status == 400 and "seed" in msg and send_seed:
                        send_seed = False
                        continue
                    if status == 400 and ("reasoning" in msg or "thinking" in msg) and send_reasoning:
                        send_reasoning = False
                        continue
                    if status == 400 and json_mode and ("json" in msg or "response_format" in msg):
                        json_mode = False  # model failed JSON validation: retry free-form
                        continue
                    if status in (404,) or "model_not_found" in msg or "decommissioned" in msg:
                        break
                    if status is None and provider == "local":
                        if not config.LOCAL_FALLBACK:  # keep the generator consistent: fail loudly instead
                            raise RuntimeError(f"Local LLM server unreachable or timed out ({e}); LOCAL_FALLBACK=0, so "
                                               "no cloud fallback. Start/reload the local model and re-run.") from e
                        break  # local server stopped or timed out: fall back to the next model instead of retrying
                    if status is not None and provider == "local" and not config.LOCAL_FALLBACK:
                        if status >= 500 and attempt < 3:
                            time.sleep(3 * (attempt + 1))
                            continue
                        raise RuntimeError(f"Local LLM error {status}: {e}; LOCAL_FALLBACK=0, so no cloud fallback.") from e
                    if status is None or status >= 500:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break
        raise RuntimeError(f"All models for role '{role}' failed. Last error: {last_err}")

    def chat_json(self, messages, default=None, **kw) -> tuple[Any, LLMResponse]:
        resp = self.chat(messages, json_mode=True, **kw)
        data = parse_json(resp.text, default)
        if isinstance(default, dict) and not isinstance(data, dict):
            # The caller expects an object but the model answered with a list. A one-element list holding an object
            # is that object ([{"labels": [...]}]); any other list is kept under "_list" so list_field() can read a
            # bare answer such as [true, false] or [{"claim": 1, "correct": true}, ...]. Anything else -> default.
            if isinstance(data, list):
                first = next((x for x in data if isinstance(x, dict)), {})
                data = first if len(data) == 1 and first else {**default, **first, "_list": data}
            else:
                data = default
        return data, resp

    # --------------------------------------------------------------- tracking
    def _record(self, resp: LLMResponse, role: str) -> None:
        self.calls.append(
            CallRecord(resp.purpose, role, resp.model, resp.input_tokens, resp.output_tokens, resp.latency_s, resp.cached)
        )
        if os.environ.get("LLM_TRACE"):  # optional debugging trace: one JSON line per call
            with open(os.environ["LLM_TRACE"], "a", encoding="utf-8") as f:
                f.write(json.dumps({"purpose": resp.purpose, "role": role, "model": resp.model, "cached": resp.cached,
                                    "text": resp.text[:400]}, ensure_ascii=False) + "\n")

    def usage_since(self, start_index: int) -> dict:
        calls = self.calls[start_index:]
        return {
            "calls": len(calls),
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "latency_s": sum(c.latency_s for c in calls),
        }

    def usage_table(self):
        import pandas as pd

        if not self.calls:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(c) for c in self.calls])
        return (
            df.groupby(["role", "model"])
            .agg(calls=("model", "size"), input_tokens=("input_tokens", "sum"), output_tokens=("output_tokens", "sum"),
                 cached=("cached", "sum"), total_latency_s=("latency_s", "sum"))
            .reset_index()
        )

    # ------------------------------------------------------------------ mock
    def _mock(self, messages, model, json_mode, purpose, role) -> LLMResponse:
        """Deterministic offline stand-in used ONLY for smoke tests of the plumbing."""
        last = messages[-1]["content"]
        h = int(hashlib.md5(last.encode()).hexdigest(), 16)
        if json_mode:
            text = "{}"
        else:
            snippet = re.sub(r"\s+", " ", last)[:300]
            text = (
                f"Thought: mock reasoning step one because the question asks something. "
                f"Therefore step two follows [S1]. However, sources may disagree [S2].\n"
                f"Final Answer: mock answer ({h % 97}) based on: {snippet}"
            )
        resp = LLMResponse(text, count_messages(messages), count_tokens(text), 0.01 + (h % 50) / 1000, model, False, purpose)
        self._record(resp, role)
        return resp


class ScriptedLLM:
    """Deterministic stand-in that replays scripted responses (used ONLY to demonstrate safety mechanisms such as
    repetition detection or timeouts without spending API calls). Same interface as LLM.chat."""

    def __init__(self, responses: list[str], latency_s: float = 0.0):
        self.responses = responses
        self.i = 0
        self.latency_s = latency_s
        self.calls: list[CallRecord] = []

    def chat(self, messages, role="generator", purpose="", **kw) -> LLMResponse:
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        r = LLMResponse(text, count_messages(msgs), count_tokens(text), self.latency_s, "scripted", False, purpose)
        self.calls.append(CallRecord(purpose, role, "scripted", r.input_tokens, r.output_tokens, r.latency_s, False))
        return r

    def chat_json(self, messages, default=None, **kw):
        r = self.chat(messages, **kw)
        return parse_json(r.text, default), r

    def usage_since(self, start_index: int) -> dict:
        calls = self.calls[start_index:]
        return {"calls": len(calls), "input_tokens": sum(c.input_tokens for c in calls),
                "output_tokens": sum(c.output_tokens for c in calls), "latency_s": sum(c.latency_s for c in calls)}
