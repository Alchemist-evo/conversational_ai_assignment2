"""Task 5 Part B — DPO loss, PEFT memory/parameter analysis and GRPO advantage computation (from first principles)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def dpo_loss(pw_theta: float, pl_theta: float, pw_ref: float, pl_ref: float, beta: float = 1.0) -> dict:
    """L_DPO = -log sigma( beta * [ log(pi_theta(y_w)/pi_ref(y_w)) - log(pi_theta(y_l)/pi_ref(y_l)) ] )"""
    ratio_w, ratio_l = pw_theta / pw_ref, pl_theta / pl_ref
    log_w, log_l = math.log(ratio_w), math.log(ratio_l)
    diff = log_w - log_l
    z = beta * diff
    sig = 1 / (1 + math.exp(-z))
    loss = -math.log(sig)
    return {"ratio_w": ratio_w, "ratio_l": ratio_l, "log_ratio_w": log_w, "log_ratio_l": log_l,
            "difference": diff, "beta_x_diff": z, "sigmoid": sig, "loss": loss,
            "implicit_reward_w": beta * log_w, "implicit_reward_l": beta * log_l}


# ------------------------------------------------------------------ PEFT
LLAMA7B = {"params": 6.74e9, "layers": 32, "d_model": 4096, "d_ff": 11008}


def lora_params(r: int = 16, targets=("q", "k", "v", "o"), arch: dict = LLAMA7B) -> int:
    d = arch["d_model"]
    per_matrix = r * (d + d)  # A: d x r, B: r x d  (attention projections are d x d)
    return int(per_matrix * len(targets) * arch["layers"])


def peft_comparison(r: int = 16) -> pd.DataFrame:
    P = LLAMA7B["params"]
    lp = lora_params(r)
    gb = 1024**3
    # Full FT with AdamW mixed precision: bf16 weights (2) + bf16 grads (2) + fp32 master (4) + Adam m,v (8) = 16 B/param
    full_vram = P * 16 / gb
    lora_vram = (P * 2 + lp * 16) / gb  # frozen bf16 base + full optimiser state only for adapters
    qlora_vram = (P * 0.5 + P * 0.5 * 0.0156 + lp * 16) / gb  # NF4 weights (0.5 B) + double-quantised constants
    act = 4.0  # GB activations for batch 1, seq 2k with gradient checkpointing (rough, same for all)
    rows = [
        {"Method": "Full fine-tuning", "Trainable params": f"{P/1e9:.2f} B", "Trainable %": 100.0,
         "GPU VRAM (GB, approx.)": round(full_vram + act, 1), "Storage per task": f"{P*2/gb:.1f} GB (full bf16 copy)",
         "Catastrophic forgetting risk": "High - all weights move", "Best use case": "Large domain shift with lots of data and multi-GPU budget"},
        {"Method": f"LoRA (r={r}, q/k/v/o)", "Trainable params": f"{lp/1e6:.1f} M", "Trainable %": round(100 * lp / P, 3),
         "GPU VRAM (GB, approx.)": round(lora_vram + act, 1), "Storage per task": f"{lp*2/1024**2:.0f} MB adapter",
         "Catastrophic forgetting risk": "Low - base frozen, low-rank update", "Best use case": "Task/style adaptation on one 24-40 GB GPU; many swappable adapters"},
        {"Method": f"QLoRA (NF4 base, r={r})", "Trainable params": f"{lp/1e6:.1f} M", "Trainable %": round(100 * lp / P, 3),
         "GPU VRAM (GB, approx.)": round(qlora_vram + act, 1), "Storage per task": f"{lp*2/1024**2:.0f} MB adapter",
         "Catastrophic forgetting risk": "Low (plus small quantisation error)", "Best use case": "Fine-tuning on a single consumer/Colab GPU (12-16 GB)"},
    ]
    return pd.DataFrame(rows)


def memory_for_model(P: float = 70e9, overhead: float = 1.2) -> pd.DataFrame:
    """M = P x B x 1.2  (bytes per parameter x 20 % overhead for activations / KV-cache / buffers)."""
    rows = []
    for name, B in [("FP32", 4), ("BF16", 2), ("INT8", 1), ("INT4 / NF4", 0.5)]:
        m = P * B * overhead / 1e9
        rows.append({"Precision": name, "Bytes/param (B)": B, "Memory M = P x B x 1.2 (GB)": round(m, 1),
                     "80 GB GPUs needed": math.ceil(m / 80)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ GRPO
def grpo_advantages(rewards: list[float]) -> np.ndarray:
    """Group-relative advantage: A_i = (r_i - mean(r)) / std(r)  — replaces PPO's learned value baseline."""
    r = np.asarray(rewards, float)
    return (r - r.mean()) / (r.std() + 1e-8)
