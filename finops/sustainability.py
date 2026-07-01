"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


def reasoning_budget_report(requests, region: str = "us-east-1", reasoning_cap_frac: float = 0.10) -> dict:
    """Your-Turn #4: quantify what `is_reasoning` traffic costs in energy & carbon,
    and propose a routing rule that caps it.

    `requests` is an iterable of dicts with 'total_tokens' (int) and 'is_reasoning' (bool).
    Reasoning queries burn ~80x the energy of a plain query (deck: 74-86x), so a small
    share of traffic can dominate the energy/carbon bill. We measure the reasoning share
    of Wh, then model a cap: keep the top `reasoning_cap_frac` of reasoning requests
    (assume the rest could route to a non-reasoning path) and report the Wh/carbon saved.
    """
    reqs = list(requests)
    n_reason = sum(1 for r in reqs if r.get("is_reasoning"))
    wh_reason = wh_plain = 0.0
    for r in reqs:
        tok = int(r.get("total_tokens", 0))
        if r.get("is_reasoning"):
            wh_reason += wh_per_query(tok, is_reasoning=True)
        else:
            wh_plain += wh_per_query(tok, is_reasoning=False)
    wh_total = wh_reason + wh_plain

    # Cap policy: only the top `reasoning_cap_frac` of reasoning requests keep the
    # reasoning path; the rest fall back to the plain (1x) path for the same tokens.
    keep = int(round(n_reason * max(0.0, min(1.0, reasoning_cap_frac))))
    reason_sorted = sorted(
        (r for r in reqs if r.get("is_reasoning")),
        key=lambda r: int(r.get("total_tokens", 0)), reverse=True,
    )
    wh_capped = wh_plain
    for i, r in enumerate(reason_sorted):
        tok = int(r.get("total_tokens", 0))
        wh_capped += wh_per_query(tok, is_reasoning=(i < keep))

    wh_saved = wh_total - wh_capped
    return {
        "n_requests": len(reqs),
        "n_reasoning": n_reason,
        "reasoning_share_pct": round(100.0 * n_reason / len(reqs), 1) if reqs else 0.0,
        "wh_total": round(wh_total, 2),
        "wh_reasoning": round(wh_reason, 2),
        "wh_reasoning_share_pct": round(100.0 * wh_reason / wh_total, 1) if wh_total else 0.0,
        "carbon_g_total": round(carbon_g(wh_total, region), 2),
        "energy_cost_usd": round(energy_cost_usd(wh_total, region), 4),
        # after the routing cap
        "cap_frac": reasoning_cap_frac,
        "wh_after_cap": round(wh_capped, 2),
        "wh_saved": round(wh_saved, 2),
        "wh_saved_pct": round(100.0 * wh_saved / wh_total, 1) if wh_total else 0.0,
        "carbon_g_saved": round(carbon_g(wh_saved, region), 2),
        "energy_usd_saved": round(energy_cost_usd(wh_saved, region), 4),
    }
