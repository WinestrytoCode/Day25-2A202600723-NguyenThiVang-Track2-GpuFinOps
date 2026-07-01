"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


# Per-GPU-type spot interruption rate (per-hour eviction chance) — 2026 snapshot.
# Scarcer/pricier accelerators get reclaimed more often, which erodes the spot
# discount through rework. Falls back to a moderate 5% for unknown types.
SPOT_INTERRUPT_RATE = {
    "H100": 0.06, "H200": 0.09, "B200": 0.12,
    "A100": 0.04, "MI300X": 0.05, "A10G": 0.02, "L4": 0.015,
}


def recommend_tier_v2(
    hours_per_day: float,
    days_per_month: float,
    interruptible: bool,
    gpu_type: str,
    on_demand_hr: float,
    spot_hr: float,
    reserved_1yr_hr: float,
    reserved_3yr_hr: float,
    num_gpus: int = 1,
    horizon_months: int = 12,
) -> dict:
    """Your-Turn #1: cost-driven tier choice that weighs interruption rate and 3yr-vs-1yr.

    Unlike the simple duty-cycle rule, this prices out *every* eligible tier for the
    real monthly gpu-hours and picks the cheapest, subject to policy constraints:

      - 'spot' is only eligible for interruptible jobs, and its effective cost is
        inflated by the per-GPU-type interruption rate (rework), so a flaky H200/B200
        may lose to reserved even though the sticker spot rate is lower.
      - reserved 3yr is only allowed when the commitment horizon can amortize it
        (>= 24 months); otherwise 1yr competes. Both must clear break-even utilization.

    Returns the winning tier plus the full priced comparison for the report.
    """
    gpu_hours = max(0.0, hours_per_day) * max(0.0, days_per_month) * max(1, num_gpus)
    duty = max(0.0, hours_per_day) / 24.0

    candidates = {"on_demand": gpu_hours * on_demand_hr}

    if interruptible:
        rate = SPOT_INTERRUPT_RATE.get(gpu_type, 0.05)
        sim = spot_checkpoint_cost(gpu_hours, spot_hr, on_demand_hr, interrupt_rate=rate)
        candidates["spot"] = sim["spot_cost"]

    # Reserved only makes sense above break-even (else you pay for idle capacity).
    if duty >= break_even_utilization(1.0 - reserved_1yr_hr / on_demand_hr):
        candidates["reserved_1yr"] = gpu_hours * reserved_1yr_hr
    if horizon_months >= 24 and duty >= break_even_utilization(1.0 - reserved_3yr_hr / on_demand_hr):
        candidates["reserved_3yr"] = gpu_hours * reserved_3yr_hr

    best = min(candidates, key=candidates.get)
    # Collapse the two reserved variants to a single 'reserved' label for M3 compatibility.
    tier = "reserved" if best.startswith("reserved") else best
    return {
        "tier": tier,
        "reserved_term": best if best.startswith("reserved") else None,
        "monthly_cost": round(candidates[best], 2),
        "on_demand_cost": round(candidates["on_demand"], 2),
        "priced": {k: round(v, 2) for k, v in candidates.items()},
    }


def cache_is_worth_it(
    cached_in: int,
    reads: int = 1,
    write_surcharge: float = 0.25,
    price_in_per_m: float = 3.0,
    storage_cost_per_m_hr: float = 0.0,
    stored_hours: float = 0.0,
    cache_discount: float = 0.10,
) -> bool:
    """Your-Turn #3: does prompt caching actually pay for this prompt?

    Caching is not free: some providers surcharge the cache *write* (Anthropic ~+25%
    on first write) and/or charge *storage* per stored-token-hour (Gemini). It only
    wins once the cached tokens are re-read enough times to beat those costs.

    Modeled per 1 cached token, all relative to full input price = 1.0:
      - naive (no cache) over `reads` reads:            reads * 1.0
      - cached: one write (1 + write_surcharge) + reads * cache_discount + storage
    `storage_cost_per_m_hr` is a $/1M-token-hour storage rate, normalized against the
    input price (`price_in_per_m`) so it lands on the same 1.0-per-input-token scale.
    Returns True when the cached path is cheaper.
    """
    if cached_in <= 0 or reads <= 0:
        return False
    naive = reads * 1.0
    write = 1.0 + write_surcharge
    read = reads * cache_discount
    # Storage per token per hour, expressed as a fraction of the per-token input price.
    storage_per_tok = (storage_cost_per_m_hr * stored_hours) / price_in_per_m if price_in_per_m > 0 else 0.0
    cached_cost = write + read + storage_per_tok
    return cached_cost < naive


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
