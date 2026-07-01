"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30
# NimbusAI is willing to sign a 3-year commitment for its steady always-on fleet,
# so the v2 policy may pick 3yr reserved where the duty cycle amortizes it.
COMMIT_HORIZON_MONTHS = 36


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = simple_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # Your-Turn #1: cost-driven policy that weighs per-GPU interruption rate and
        # the 3yr-vs-1yr discount, picking the cheapest eligible tier.
        v2 = pricing.recommend_tier_v2(
            hpd, DAYS, interruptible, gtype, od, num(c["spot_hr"]),
            num(c["reserved_1yr_hr"]), num(c["reserved_3yr_hr"]), num_gpus=ngpu,
            horizon_months=COMMIT_HORIZON_MONTHS,
        )
        tier = v2["tier"]
        opt_cost = v2["monthly_cost"]

        # Baseline for comparison: the deliberately simple duty-cycle rule.
        simple_tier = pricing.recommend_tier(hpd, interruptible)
        if simple_tier == "spot":
            simple_cost = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
        elif simple_tier == "reserved":
            simple_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            simple_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        simple_monthly += simple_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "reserved_term": v2["reserved_term"],
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    simple_savings_pct = (on_demand_monthly - simple_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':13}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            label = r["tier"] + (f"({r['reserved_term'].split('_')[1]})" if r["reserved_term"] else "")
            print(f"{r['job_id']:18}{r['gpu_type']:7}{label:13}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"vs simple duty-cycle policy: ${simple_monthly:,.0f} ({simple_savings_pct:.1f}% saved) "
              f"-> v2 improves by ${simple_monthly - optimized_monthly:,.0f}/mo")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "simple_monthly": round(simple_monthly), "simple_savings_pct": round(simple_savings_pct, 1)}


if __name__ == "__main__":
    run()
