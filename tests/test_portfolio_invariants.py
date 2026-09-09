"""Cross-module accounting and execution invariants for the portfolio engine.

These tests are intentionally broader than the focused unit tests for individual
modules.  They verify that the state, optimizer, diagnostics, and dynamic-NAV
layers agree on the same economic objects:

* cash + risky weights = 1 after drift and after rebalancing;
* delta_w = w_post - w_pre is the authoritative trade vector;
* dollar_trade = current AUM * delta_w;
* one-way turnover = 0.5 * sum(abs(delta_w));
* one-day ADV trade caps constrain the actual rebalance;
* legacy positions that exceed a static position cap are grandfathered rather
  than liquidated unrealistically in one step;
* partial exits remain as residual holdings for the next scheduled rebalance;
* diagnostics consume the canonical RebalanceResult trade vector;
* canonical DAILY ADV / DAILY sigma drive market-impact diagnostics even when
  stale historical weekly proxies are present;
* dynamic NAV applies a period's return only to the NEXT scheduled rebalance.

The file should live at::

    Scripts/Portfolio/test_portfolio_invariants.py

No production behavior is defined here.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cost_aware_portfolio.optimizer import (
    CostAwareParams,
    compute_trade_cost_breakdown,
    solve_rebalance_to_target_nonlinear_pg,
)
from cost_aware_portfolio.dynamic_nav import _DynamicNAVController
from cost_aware_portfolio.diagnostics import PortfolioDiagnostics
from cost_aware_portfolio.state import (
    PreTradeState,
    build_pretrade_state,
    build_rebalance_result,
    drift_posttrade_portfolio,
)
from cost_aware_portfolio.transaction_costs import (
    TC_ADV_DAILY_COL,
    TC_SIGMA_DAILY_COL,
)


TOL = 1e-9


def _series(values, names):
    return pd.Series(values, index=pd.Index(names, dtype=object), dtype=float)


def _target_params(
    *,
    pos_adv_cap=np.inf,
    trade_adv_cap=np.inf,
    execution_days=1.0,
):
    """Stable target-tracking parameters for small deterministic test problems."""
    return CostAwareParams(
        lambda_risk=0.0,
        eta_tc=0.0,
        eta_linear=0.0,
        eta_impact=0.0,
        impact_kappa=1.0,
        impact_exponent=1.5,
        impact_smooth_eps=1e-5,
        spread_bps_default=0.0,
        dollar_neutral=False,
        long_only=True,
        gross_target=1.0,
        pos_adv_cap=float(pos_adv_cap),
        trade_adv_cap=float(trade_adv_cap),
        execution_days=float(execution_days),
        box_abs=1.0,
        target_penalty=1000.0,
        max_iters=2000,
        step=0.1,
        tol=1e-10,
        allow_partial_long_only=True,
    )


def test_drift_preserves_nav_accounting_for_signed_book():
    post = _series([0.40, -0.20], ["LONG", "SHORT"])
    rets = _series([0.10, -0.05], ["LONG", "SHORT"])
    cash = 0.80

    drift = drift_posttrade_portfolio(
        post,
        rets,
        cash_weight=cash,
        cash_return=0.0,
    )

    expected_growth = 0.80 + 0.40 * 1.10 - 0.20 * 0.95
    assert np.isclose(drift.growth_factor, expected_growth, atol=TOL)
    assert np.isclose(drift.portfolio_return, expected_growth - 1.0, atol=TOL)
    assert np.isclose(drift.cash_weight + drift.weights.sum(), 1.0, atol=TOL)
    assert np.isclose(drift.weights["LONG"], 0.44 / expected_growth, atol=TOL)
    assert np.isclose(drift.weights["SHORT"], -0.19 / expected_growth, atol=TOL)


def test_rebalance_result_closes_cash_trade_and_dollar_identities_on_union():
    pre = PreTradeState(
        date=pd.Timestamp("2024-01-12"),
        aum=100_000_000.0,
        weights=_series([0.20, 0.30, 0.50], ["A", "B", "C"]),
        cash_weight=0.0,
    )
    # A and C are explicit exits; D is a new entry.
    post = _series([0.40, 0.60], ["B", "D"])

    result = build_rebalance_result(
        pretrade_state=pre,
        posttrade_weights=post,
    )

    expected_delta = _series([-0.20, 0.10, -0.50, 0.60], ["A", "B", "C", "D"])
    pd.testing.assert_series_equal(
        result.delta_w.sort_index(),
        expected_delta.sort_index().rename("delta_w"),
        check_names=True,
        check_index_type=False,
    )
    assert np.allclose(
        result.dollar_trades.sort_index().to_numpy(),
        100_000_000.0 * expected_delta.sort_index().to_numpy(),
        atol=1e-6,
    )
    assert np.isclose(result.turnover_oneway, 0.5 * expected_delta.abs().sum(), atol=TOL)
    assert np.isclose(result.pretrade_cash_weight + result.pretrade_weights.sum(), 1.0, atol=TOL)
    assert np.isclose(result.posttrade_cash_weight + result.posttrade_weights.sum(), 1.0, atol=TOL)


def test_rebalance_result_rejects_inconsistent_posttrade_cash():
    pre = PreTradeState(
        date=pd.Timestamp("2024-01-12"),
        aum=1_000_000.0,
        weights=_series([0.50], ["A"]),
        cash_weight=0.50,
    )

    with pytest.raises(ValueError, match=r"cash \+ risky weights must equal 1"):
        build_rebalance_result(
            pretrade_state=pre,
            posttrade_weights=_series([0.60], ["A"]),
            posttrade_cash_weight=0.50,
        )


def test_one_day_trade_cap_limits_actual_optimizer_move():
    names = ["A", "B"]
    aum = 100_000_000.0
    adv = _series([20_000_000.0, 20_000_000.0], names)
    prev = _series([0.50, 0.50], names)
    target = _series([1.00, 0.00], names)
    sigma = _series([0.02, 0.02], names)
    spread = _series([0.0, 0.0], names)
    params = _target_params(trade_adv_cap=0.10, execution_days=1.0)

    solved = solve_rebalance_to_target_nonlinear_pg(
        w_target=target,
        w_prev=prev,
        adv=adv,
        sigma=sigma,
        aum=aum,
        spreads_oneway=spread,
        params=params,
    )

    trade_cap = 0.10 * 20_000_000.0 / aum
    delta = (solved - prev).abs()
    assert np.all(delta.to_numpy() <= trade_cap + 1e-8)
    assert np.isclose(solved.sum(), 1.0, atol=1e-8)


def test_position_cap_limits_new_positions():
    names = ["A", "B", "C", "D"]
    aum = 100_000_000.0
    adv = _series([100_000_000.0] * 4, names)
    prev = _series([0.0] * 4, names)
    target = _series([1.0, 0.0, 0.0, 0.0], names)
    sigma = _series([0.02] * 4, names)
    spread = _series([0.0] * 4, names)
    params = _target_params(pos_adv_cap=0.25, trade_adv_cap=np.inf)

    solved = solve_rebalance_to_target_nonlinear_pg(
        w_target=target,
        w_prev=prev,
        adv=adv,
        sigma=sigma,
        aum=aum,
        spreads_oneway=spread,
        params=params,
    )

    position_cap = 0.25 * 100_000_000.0 / aum
    assert np.all(solved.to_numpy() <= position_cap + 1e-8)
    assert np.all(solved.to_numpy() >= -1e-10)
    assert np.isclose(solved.sum(), 1.0, atol=1e-8)


def test_grandfathered_legacy_position_exits_only_as_fast_as_trade_cap_allows():
    names = ["LEGACY"]
    aum = 100_000_000.0
    adv = _series([100_000_000.0], names)
    sigma = _series([0.02], names)
    spread = _series([0.0], names)
    params = _target_params(
        pos_adv_cap=0.20,
        trade_adv_cap=0.10,
        execution_days=1.0,
    )

    # The position begins above the 20% position cap.  The optimizer must not
    # pretend it can jump from 50% to 20% (or zero) in one day when the one-day
    # trade cap permits only 10% of NAV.
    prev_1 = _series([0.50], names)
    target = _series([0.0], names)
    solved_1 = solve_rebalance_to_target_nonlinear_pg(
        w_target=target,
        w_prev=prev_1,
        adv=adv,
        sigma=sigma,
        aum=aum,
        spreads_oneway=spread,
        params=params,
        desired_target_sum=0.0,
    )
    assert np.isclose(solved_1["LEGACY"], 0.40, atol=1e-8)
    assert solved_1["LEGACY"] > 0.20  # grandfathered above static position cap

    # At the next scheduled weekly rebalance, the residual 40% holding is the
    # actual starting position.  Another one-day trade can reduce it only to 30%.
    pre_2 = build_pretrade_state(
        date=pd.Timestamp("2024-01-19"),
        aum=aum,
        previous_posttrade_weights=solved_1,
        realized_returns=_series([0.0], names),
        previous_cash_weight=1.0 - float(solved_1.sum()),
        previous_date=pd.Timestamp("2024-01-12"),
    )
    assert np.isclose(pre_2.weights["LEGACY"], 0.40, atol=1e-8)

    solved_2 = solve_rebalance_to_target_nonlinear_pg(
        w_target=target,
        w_prev=pre_2.weights,
        adv=adv,
        sigma=sigma,
        aum=aum,
        spreads_oneway=spread,
        params=params,
        desired_target_sum=0.0,
    )
    assert np.isclose(solved_2["LEGACY"], 0.30, atol=1e-8)


def test_diagnostics_turnover_decomposition_uses_canonical_rebalance_result():
    pre = PreTradeState(
        date=pd.Timestamp("2024-01-12"),
        aum=100.0,
        weights=_series([0.20, 0.30, 0.50], ["A", "B", "C"]),
        cash_weight=0.0,
    )
    result = build_rebalance_result(
        pretrade_state=pre,
        posttrade_weights=_series([0.40, 0.60], ["B", "D"]),
    )

    decomp = PortfolioDiagnostics.turnover_decomposition_from_rebalance_result(result)
    assert np.isclose(decomp["entry_turnover"], 0.30, atol=TOL)
    assert np.isclose(decomp["exit_turnover"], 0.35, atol=TOL)
    assert np.isclose(decomp["resize_turnover"], 0.05, atol=TOL)
    assert np.isclose(decomp["total_turnover"], result.turnover_oneway, atol=TOL)
    assert np.isclose(
        decomp["entry_turnover"] + decomp["exit_turnover"] + decomp["resize_turnover"],
        result.turnover_oneway,
        atol=TOL,
    )


def test_diagnostics_costs_match_optimizer_cost_breakdown_on_same_trade_vector():
    names = ["A", "B"]
    pre = PreTradeState(
        date=pd.Timestamp("2024-01-12"),
        aum=100_000_000.0,
        weights=_series([0.40, 0.60], names),
        cash_weight=0.0,
    )
    result = build_rebalance_result(
        pretrade_state=pre,
        posttrade_weights=_series([0.30, 0.70], names),
    )

    frame = pd.DataFrame(
        {
            TC_ADV_DAILY_COL: [20_000_000.0, 50_000_000.0],
            TC_SIGMA_DAILY_COL: [0.02, 0.03],
            "spread_oneway": [0.001, 0.002],
            # Deliberately absurd stale proxies.  They must be ignored when the
            # canonical forecast-input path is enabled.
            "DollarVol_20d": [1.0, 1.0],
            "mw_var": [999.0, 999.0],
        },
        index=names,
    )
    ctx = SimpleNamespace(
        tc_use_forecast_inputs=True,
        tc_aum_dollars=100_000_000.0,
        tc_execution_days=1.0,
        tc_impact_kappa=1.0,
        tc_impact_exponent=1.5,
    )

    linear, impact, total, tradable = PortfolioDiagnostics.trade_cost_series_from_rebalance_result(
        ctx=ctx,
        frame=frame,
        rebalance_result=result,
    )
    assert bool(tradable.all())

    expected = compute_trade_cost_breakdown(
        w=result.posttrade_weights,
        w_prev=result.pretrade_weights,
        spreads_oneway=frame["spread_oneway"],
        sigma=frame[TC_SIGMA_DAILY_COL],
        adv=frame[TC_ADV_DAILY_COL],
        aum=result.aum,
        impact_kappa=ctx.tc_impact_kappa,
        impact_exponent=ctx.tc_impact_exponent,
        execution_days=ctx.tc_execution_days,
    )

    assert np.isclose(float(linear.sum()), expected.linear_cost, atol=TOL)
    assert np.isclose(float(impact.sum()), expected.impact_cost, atol=TOL)
    assert np.isclose(float(total.sum()), expected.total_cost, atol=TOL)
    assert np.isclose(float((linear + impact).sum()), float(total.sum()), atol=TOL)


def test_canonical_daily_inputs_are_invariant_to_stale_weekly_proxy_values():
    names = ["A", "B"]
    pre = PreTradeState(
        date=pd.Timestamp("2024-01-12"),
        aum=100_000_000.0,
        weights=_series([0.50, 0.50], names),
        cash_weight=0.0,
    )
    result = build_rebalance_result(
        pretrade_state=pre,
        posttrade_weights=_series([0.45, 0.55], names),
    )
    base = pd.DataFrame(
        {
            TC_ADV_DAILY_COL: [25_000_000.0, 40_000_000.0],
            TC_SIGMA_DAILY_COL: [0.02, 0.025],
            "spread_oneway": [0.001, 0.0015],
            "DollarVol_20d": [1.0, 1.0],
            "mw_var": [1e6, 1e6],
        },
        index=names,
    )
    altered = base.copy()
    altered["DollarVol_20d"] = [1e15, 1e15]
    altered["mw_var"] = [1e-16, 1e-16]

    ctx = SimpleNamespace(
        tc_use_forecast_inputs=True,
        tc_aum_dollars=100_000_000.0,
        tc_execution_days=1.0,
        tc_impact_kappa=1.0,
        tc_impact_exponent=1.5,
    )

    _, _, cost_1, _ = PortfolioDiagnostics.trade_cost_series_from_rebalance_result(
        ctx=ctx,
        frame=base,
        rebalance_result=result,
    )
    _, _, cost_2, _ = PortfolioDiagnostics.trade_cost_series_from_rebalance_result(
        ctx=ctx,
        frame=altered,
        rebalance_result=result,
    )
    assert np.allclose(cost_1.to_numpy(), cost_2.to_numpy(), atol=TOL, rtol=0.0)


def test_dynamic_nav_defers_nav_update_until_next_scheduled_rebalance():
    controller = _DynamicNAVController(100.0)
    ctx = SimpleNamespace(tc_aum_dollars=np.nan)

    nav_1 = controller.begin_rebalance(ctx=ctx, date=pd.Timestamp("2024-01-05"))
    assert np.isclose(nav_1, 100.0)
    assert np.isclose(ctx.tc_aum_dollars, 100.0)

    controller.finish_rebalance(
        date=pd.Timestamp("2024-01-05"),
        row=[0.00, 0.10],
        cut=2,
    )
    # The current date's diagnostics still see beginning NAV.
    assert np.isclose(ctx.tc_aum_dollars, 100.0)
    assert np.isclose(controller.final_nav, 110.0)

    nav_2 = controller.begin_rebalance(ctx=ctx, date=pd.Timestamp("2024-01-12"))
    assert np.isclose(nav_2, 110.0)
    assert np.isclose(ctx.tc_aum_dollars, 110.0)

    controller.finish_rebalance(
        date=pd.Timestamp("2024-01-12"),
        row=[0.00, -0.10],
        cut=2,
    )
    assert np.isclose(controller.final_nav, 99.0)

    accounting = controller.accounting_frame()
    assert np.allclose(accounting["nav_start"].to_numpy(), [100.0, 110.0])
    assert np.allclose(accounting["nav_end"].to_numpy(), [110.0, 99.0])
    assert np.allclose(
        accounting["nav_end"].to_numpy(),
        accounting["nav_start"].to_numpy() * (1.0 + accounting["net_return"].to_numpy()),
        atol=TOL,
    )

