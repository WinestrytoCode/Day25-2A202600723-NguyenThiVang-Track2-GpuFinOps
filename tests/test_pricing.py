import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing


def test_discount_stack_is_multiplicative():
    assert abs(pricing.discount_stack(batch=True, cache_hit_frac=1.0) - 0.05) < 1e-9   # ~95% off
    assert pricing.discount_stack() == 1.0
    assert abs(pricing.discount_stack(batch=True) - 0.5) < 1e-9


def test_break_even():
    assert abs(pricing.break_even_utilization(0.45) - 0.55) < 1e-9
    assert pricing.break_even_utilization(0.0) == 1.0


def test_recommend_tier():
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_request_cost_and_cache():
    full = pricing.request_cost(1000, 1000, 3.0, 15.0)
    cached = pricing.request_cost(1000, 1000, 3.0, 15.0, cached_in=1000)
    assert cached < full                       # caching reduces cost
    batched = pricing.request_cost(1000, 1000, 3.0, 15.0, batch=True)
    assert abs(batched - full * 0.5) < 1e-9    # batch = -50%


def test_spot_checkpoint_saves():
    res = pricing.spot_checkpoint_cost(100, 1.5, 2.5)
    assert res["spot_cost"] < res["on_demand_cost"]
    assert res["savings_pct"] > 0


# --- Your-Turn #1: interruption-aware, discount-aware tier policy ---

def test_recommend_tier_v2_picks_cheapest_reserved_for_steady_job():
    # 24/7 A100 for 30 days with a 36-month horizon -> 3yr reserved is cheapest.
    r = pricing.recommend_tier_v2(24, 30, False, "A100", 1.79, 1.1, 1.4, 1.0,
                                  num_gpus=1, horizon_months=36)
    assert r["tier"] == "reserved"
    assert r["reserved_term"] == "reserved_3yr"
    assert r["monthly_cost"] < r["on_demand_cost"]


def test_recommend_tier_v2_short_horizon_blocks_3yr():
    # A 12-month horizon must never commit to a 3yr term.
    r = pricing.recommend_tier_v2(24, 30, False, "A100", 1.79, 1.1, 1.4, 1.0,
                                  num_gpus=1, horizon_months=12)
    assert "reserved_3yr" not in r["priced"]


def test_recommend_tier_v2_high_interrupt_rate_can_beat_spot():
    # An always-on interruptible H100 (6% eviction) is cheaper on 3yr reserved than spot.
    r = pricing.recommend_tier_v2(24, 30, True, "H100", 2.5, 1.5, 2.0, 1.4,
                                  num_gpus=1, horizon_months=36)
    assert r["priced"]["reserved_3yr"] < r["priced"]["spot"]
    assert r["tier"] == "reserved"


# --- Your-Turn #3: cache only pays above a read threshold ---

def test_cache_is_worth_it_thresholds_on_reads():
    assert pricing.cache_is_worth_it(1000, reads=1) is False   # single read: write surcharge loses
    assert pricing.cache_is_worth_it(1000, reads=5) is True     # re-read enough: caching wins
    assert pricing.cache_is_worth_it(0, reads=10) is False      # nothing cached


def test_cache_is_worth_it_storage_can_kill_it():
    # Heavy per-token-hour storage (Gemini-style) can make caching not worth it.
    cheap = pricing.cache_is_worth_it(1000, reads=5, storage_cost_per_m_hr=0.0, stored_hours=0)
    pricey = pricing.cache_is_worth_it(1000, reads=5, storage_cost_per_m_hr=5.0, stored_hours=24)
    assert cheap is True and pricey is False
