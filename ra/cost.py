"""Task 6.5 / Task 7 Part B — cost engineering with the assignment's pricing assumptions."""
from __future__ import annotations

import pandas as pd

from .config import PRICING, cost_usd

TOK_IN, TOK_OUT = 2000, 500           # assignment: average tokens per query
QUERIES_PER_DAY, DAYS_PER_MONTH = 10_000, 30


def per_query(price_key: str, tin: float = TOK_IN, tout: float = TOK_OUT) -> float:
    return cost_usd(tin, tout, price_key)


def monthly(cost_per_query: float, qpd: int = QUERIES_PER_DAY, days: int = DAYS_PER_MONTH) -> float:
    return cost_per_query * qpd * days


def routing_table(mix: dict[str, float] | None = None) -> pd.DataFrame:
    mix = mix or {"small_lm": 0.80, "moe_mid": 0.15, "gpt-4o": 0.05}
    a = per_query("gpt-4o")
    b = sum(share * per_query(k) for k, share in mix.items())
    rows = [
        {"Strategy": "A: Flat GPT-4o", "Cost / query ($)": a, "Monthly cost ($)": monthly(a)},
        {"Strategy": "B: Routing 80% Small / 15% MoE / 5% Dense", "Cost / query ($)": b, "Monthly cost ($)": monthly(b)},
    ]
    df = pd.DataFrame(rows)
    df["Savings vs A ($/month)"] = df["Monthly cost ($)"].iloc[0] - df["Monthly cost ($)"]
    df["Reduction (%)"] = 100 * df["Savings vs A ($/month)"] / df["Monthly cost ($)"].iloc[0]
    return df


def optimisation_table(cacheable_input_share: float, context_reduction_share: float) -> pd.DataFrame:
    """Savings of each strategy applied alone to Strategy A (flat GPT-4o).

    * prompt caching: the cacheable prefix (system prompt + tool schemas) is billed at a 50 % or 90 % discount.
    * model routing : Strategy B above.
    * context reduction: drop the measured 'potentially redundant' share of input tokens (Task 3 classification).
    """
    base = monthly(per_query("gpt-4o"))
    p = PRICING["gpt-4o"]
    rows = []
    for disc in (0.5, 0.9):
        tin_eff = TOK_IN * (1 - cacheable_input_share * disc)
        c = monthly(cost_usd(tin_eff, TOK_OUT, "gpt-4o"))
        rows.append({"Strategy": f"Prompt caching ({int(disc*100)}% discount on cached prefix = {cacheable_input_share:.0%} of input)",
                     "Monthly cost ($)": c})
    rt = routing_table()
    rows.append({"Strategy": "Model routing (80/15/5)", "Monthly cost ($)": rt["Monthly cost ($)"].iloc[1]})
    tin_eff = TOK_IN * (1 - context_reduction_share)
    rows.append({"Strategy": f"Context reduction (-{context_reduction_share:.0%} input tokens)",
                 "Monthly cost ($)": monthly(cost_usd(tin_eff, TOK_OUT, "gpt-4o"))})
    tin_comb = TOK_IN * (1 - context_reduction_share) * (1 - cacheable_input_share * 0.9)
    mix = {"small_lm": 0.80, "moe_mid": 0.15, "gpt-4o": 0.05}
    comb = sum(s * cost_usd(tin_comb, TOK_OUT, k) for k, s in mix.items())
    rows.append({"Strategy": "All three combined", "Monthly cost ($)": monthly(comb)})
    df = pd.DataFrame(rows)
    df.insert(1, "Baseline A ($)", base)
    df["Savings ($)"] = base - df["Monthly cost ($)"]
    df["Reduction (%)"] = 100 * df["Savings ($)"] / base
    return df
