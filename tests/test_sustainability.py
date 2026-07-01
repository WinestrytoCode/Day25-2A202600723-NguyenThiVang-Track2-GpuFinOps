import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import sustainability


def test_reasoning_query_costs_far_more_energy():
    plain = sustainability.wh_per_query(1000, is_reasoning=False)
    reasoning = sustainability.wh_per_query(1000, is_reasoning=True)
    assert reasoning / plain == sustainability.REASONING_ENERGY_MULTIPLIER


# --- Your-Turn #4: reasoning energy budget + routing cap ---

def test_reasoning_budget_report_shares_and_cap():
    # 90 plain + 10 reasoning, all same token count -> reasoning dominates energy.
    reqs = [{"total_tokens": 1000, "is_reasoning": False} for _ in range(90)]
    reqs += [{"total_tokens": 1000, "is_reasoning": True} for _ in range(10)]
    rep = sustainability.reasoning_budget_report(reqs, reasoning_cap_frac=0.10)

    assert rep["n_requests"] == 100 and rep["n_reasoning"] == 10
    assert rep["reasoning_share_pct"] == 10.0
    # 10% of requests but the vast majority of energy (80x multiplier).
    assert rep["wh_reasoning_share_pct"] > 80.0
    # Capping to the top 10% of reasoning queries saves a large chunk of energy.
    assert rep["wh_saved"] > 0 and rep["wh_saved_pct"] > 50.0
    assert rep["carbon_g_saved"] > 0 and rep["energy_usd_saved"] > 0


def test_reasoning_budget_no_reasoning_is_noop():
    reqs = [{"total_tokens": 500, "is_reasoning": False} for _ in range(5)]
    rep = sustainability.reasoning_budget_report(reqs)
    assert rep["n_reasoning"] == 0
    assert rep["wh_saved"] == 0.0
