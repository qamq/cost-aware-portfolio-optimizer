import numpy as np
import pandas as pd

from cost_aware_portfolio import (
    CostAwareParams,
    compute_trade_cost_breakdown,
    solve_cost_aware_nonlinear_pg,
    solve_rebalance_to_target_nonlinear_pg,
)


def _inputs():
    idx = pd.Index(["A", "B", "C", "D"])
    alpha = pd.Series([0.03, 0.02, 0.01, 0.00], index=idx)
    prev = pd.Series([0.25, 0.25, 0.25, 0.25], index=idx)
    adv = pd.Series([1e9, 1e9, 1e9, 1e9], index=idx)
    sigma = pd.Series([0.02, 0.02, 0.02, 0.02], index=idx)
    spread = pd.Series([0.0004, 0.0005, 0.0006, 0.0007], index=idx)
    Sigma = np.diag([0.0004, 0.0005, 0.0006, 0.0007])
    return idx, alpha, prev, adv, sigma, spread, Sigma


def test_alpha_solver_is_feasible():
    _, alpha, prev, adv, sigma, spread, Sigma = _inputs()
    params = CostAwareParams(
        lambda_risk=5.0,
        pos_adv_cap=1.0,
        trade_adv_cap=1.0,
        box_abs=0.60,
        max_iters=1000,
        step=1.0,
    )
    out = solve_cost_aware_nonlinear_pg(
        alpha=alpha,
        Sigma=Sigma,
        w_prev=prev,
        adv=adv,
        sigma=sigma,
        aum=10e6,
        spreads_oneway=spread,
        params=params,
    )
    assert np.isclose(out.sum(), 1.0, atol=1e-7)
    assert (out >= -1e-10).all()
    assert (out <= params.box_abs + 1e-8).all()
    assert bool(out.attrs["solver_converged"])


def test_trade_cap_limits_target_tracking():
    idx, _, prev, adv, sigma, spread, _ = _inputs()
    target = pd.Series([1.0, 0.0, 0.0, 0.0], index=idx)
    small_adv = pd.Series([2e6, 2e6, 2e6, 2e6], index=idx)
    params = CostAwareParams(
        pos_adv_cap=10.0,
        trade_adv_cap=0.10,
        box_abs=1.0,
        target_penalty=100.0,
        allow_partial_long_only=True,
    )
    out = solve_rebalance_to_target_nonlinear_pg(
        w_target=target,
        w_prev=prev,
        adv=small_adv,
        sigma=sigma,
        aum=100e6,
        spreads_oneway=spread,
        params=params,
    )
    trade_cap = params.trade_adv_cap * 2e6 / 100e6
    assert np.all(np.abs((out - prev).to_numpy()) <= trade_cap + 1e-8)


def test_same_trade_has_more_impact_at_larger_aum():
    idx, _, prev, adv, sigma, spread, _ = _inputs()
    w = pd.Series([0.35, 0.25, 0.20, 0.20], index=idx)
    low = compute_trade_cost_breakdown(
        w=w, w_prev=prev, spreads_oneway=spread, sigma=sigma, adv=adv, aum=10e6
    )
    high = compute_trade_cost_breakdown(
        w=w, w_prev=prev, spreads_oneway=spread, sigma=sigma, adv=adv, aum=1e9
    )
    assert np.isclose(low.linear_cost, high.linear_cost)
    assert high.impact_cost > low.impact_cost
    assert high.total_cost > low.total_cost
