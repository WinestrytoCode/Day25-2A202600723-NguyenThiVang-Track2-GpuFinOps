"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    # --- Your-Turn extensions: surface the measured results in the deliverable ---
    reason = r2["reasoning"]
    ext_lines = [
        "**#1 Interruption-aware tier policy** — weighs per-GPU eviction rate and the "
        f"3yr-vs-1yr discount. Improves purchasing savings from {r3['simple_savings_pct']}% "
        f"(simple duty-cycle rule) to {r3['savings_pct']}% "
        f"(**+${r3['simple_monthly'] - r3['optimized_monthly']:,.0f}/mo**); it moves the "
        "always-on H100 training fleet from spot to 3yr reserved once rework is priced in.",
        "",
        "**#3 Cache economics gate** — `cache_is_worth_it()` requires the cached prompt to "
        f"be re-read enough to beat the write surcharge; {r2['cache_skipped']} request(s) were "
        "denied cache credit this run (correctly, not free savings).",
        "",
        "**#4 Reasoning energy budget** — `is_reasoning` traffic is "
        f"{reason['reasoning_share_pct']}% of requests but "
        f"**{reason['wh_reasoning_share_pct']}% of energy** "
        f"({reason['wh_reasoning']:.0f}/{reason['wh_total']:.0f} Wh/day). A routing rule "
        f"that keeps only the top {reason['cap_frac']:.0%} of reasoning queries saves "
        f"**{reason['wh_saved']:.0f} Wh/day** ({reason['wh_saved_pct']}%), "
        f"{reason['carbon_g_saved']:.0f} gCO2/day, ${reason['energy_usd_saved']:.2f}/day electricity.",
    ]

    md = report.build_report(
        baseline, optimized, levers, sustainability=sust,
        unit_econ={"baseline_per_m": r2["baseline_per_m"], "optimized_per_m": r2["optimized_per_m"]},
        extensions={"lines": ext_lines},
    )
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
