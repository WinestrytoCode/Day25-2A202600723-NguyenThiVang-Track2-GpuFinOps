"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Your-Turn #3: estimate how often a cached prompt is re-read before eviction.
# The data carries no per-prompt read count, so infer it from the cached footprint:
# a small cached slice looks like a one-off prompt (write, then no reuse), while a
# large cached slice looks like a shared system/context prefix that gets reused a lot.
CACHE_REUSE_MIN_TOKENS = 400   # below this, treat as a single-read one-off


def _estimated_reads(cached_in: int) -> int:
    if cached_in < CACHE_REUSE_MIN_TOKENS:
        return 1                      # one-off: cache write never amortizes
    return 5                          # reused prefix: pays off


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    cache_skipped = 0
    energy_rows = []
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        total_tokens += inp + out
        energy_rows.append({"total_tokens": inp + out, "is_reasoning": is_reasoning})
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API.
        # Your-Turn #3: only count cache savings when caching actually pays off.
        eff_cached = cached
        if cached and not pricing.cache_is_worth_it(cached, reads=_estimated_reads(cached)):
            eff_cached = 0
            cache_skipped += 1
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=eff_cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Your-Turn #4: reasoning energy budget + a routing cap.
    reason = sustainability.reasoning_budget_report(energy_rows, reasoning_cap_frac=0.10)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print(f"cache gate: {cache_skipped} req(s) skipped caching (one-off prompt, write surcharge > reads)")
        print(f"reasoning : {reason['n_reasoning']}/{reason['n_requests']} reqs "
              f"({reason['reasoning_share_pct']}%) but {reason['wh_reasoning_share_pct']}% of energy "
              f"({reason['wh_reasoning']:.0f}/{reason['wh_total']:.0f} Wh, "
              f"${reason['energy_cost_usd']:.2f} electricity)")
        print(f"routing cap (keep top {reason['cap_frac']:.0%} reasoning): "
              f"-{reason['wh_saved']:.0f} Wh ({reason['wh_saved_pct']}%), "
              f"-{reason['carbon_g_saved']:.0f} gCO2, -${reason['energy_usd_saved']:.2f}")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_skipped": cache_skipped, "reasoning": reason,
    }


if __name__ == "__main__":
    run()
