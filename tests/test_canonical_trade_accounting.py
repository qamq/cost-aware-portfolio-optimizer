"""Tests that turnover and costs use the canonical RebalanceResult.delta_w."""

import numpy as np
import pandas as pd

from cost_aware_portfolio.diagnostics import PortfolioDiagnostics
from cost_aware_portfolio.state import (
    PreTradeState,
    build_rebalance_result,
    combine_signed_rebalance_results,
)


class _CostContext:
    tc_use_forecast_inputs = True
    tc_aum_dollars = 100_000_000.0
    tc_impact_kappa = 1.0
    tc_impact_exponent = 1.5
    tc_pos_adv_cap = 0.50
    tc_trade_adv_cap = 0.25


def _result(pre, post, date="2024-01-12"):
    pre = pd.Series(pre, dtype=float)
    state = PreTradeState(
        date=pd.Timestamp(date),
        aum=100_000_000.0,
        weights=pre,
        cash_weight=1.0 - float(pre.sum()),
    )
    post = pd.Series(post, dtype=float)
    return build_rebalance_result(pretrade_state=state, posttrade_weights=post)


def test_combine_signed_leg_results_nets_low_and_high_before_delta():
    low = _result({"A": 0.00, "B": 0.02}, {"A": 0.02, "B": 0.00})
    high = _result({"A": 0.02, "B": 0.00}, {"A": 0.00, "B": 0.02})

    signed = combine_signed_rebalance_results(low_abs_result=low, high_result=high)

    # A moves from +2% high to -2% low; B makes the opposite migration.
    assert np.isclose(signed.pretrade_weights["A"], 0.02)
    assert np.isclose(signed.posttrade_weights["A"], -0.02)
    assert np.isclose(signed.delta_w["A"], -0.04)
    assert np.isclose(signed.delta_w["B"], 0.04)
    assert np.isclose(signed.turnover_oneway, 0.04)


def test_cost_series_uses_rebalance_result_delta_w_exactly():
    ctx = _CostContext()
    result = _result({"A": 0.10}, {"A": 0.20})
    frame = pd.DataFrame(
        {
            "tc_adv_daily": [100_000_000.0],
            "tc_sigma_daily": [0.02],
            "spread_oneway": [0.001],
        },
        index=["A"],
    )

    linear, impact, total, tradable = PortfolioDiagnostics.trade_cost_series_from_rebalance_result(
        ctx=ctx,
        frame=frame,
        rebalance_result=result,
    )

    delta = 0.10
    expected_linear = 0.001 * delta
    expected_impact = 0.02 * np.sqrt(1.0) * (delta ** 1.5)
    assert bool(tradable["A"])
    assert np.isclose(linear["A"], expected_linear)
    assert np.isclose(impact["A"], expected_impact)
    assert np.isclose(total["A"], expected_linear + expected_impact)


def test_debug_row_ignores_bogus_prev_to_df_when_canonical_result_is_supplied():
    ctx = _CostContext()
    result = _result({"A": 0.10, "B": -0.10}, {"A": 0.20, "B": 0.00})
    frame = pd.DataFrame(
        {
            "adv_hat": [100_000_000.0, 100_000_000.0],
            "sigma_hat": [0.02, 0.02],
            "spread_oneway": [0.001, 0.001],
            "MarketCap": [1e9, 1e9],
        },
        index=["A", "B"],
    )
    to_df = pd.DataFrame(
        {"weight": result.posttrade_weights, "next_week_ret": [0.0, 0.0]},
        index=["A", "B"],
    )
    # Deliberately wrong legacy state.  Canonical diagnostics must not use it.
    prev_to_df = pd.DataFrame(
        {"weight": [0.90, -0.90], "next_week_ret": [0.50, -0.50]},
        index=["A", "B"],
    )
    state = {"portfolio_debug_rows": []}

    PortfolioDiagnostics.append_portfolio_debug_row(
        ctx=ctx,
        prev_state=state,
        date=pd.Timestamp("2024-01-12"),
        reb_frame=frame,
        to_df=to_df,
        prev_to_df=prev_to_df,
        ret_name="next_week_ret",
        rebalance_result=result,
    )

    row = state["portfolio_debug_rows"][-1]
    assert np.isclose(row["total_turnover"], result.turnover_oneway)

    _, _, canonical_total, _ = PortfolioDiagnostics.trade_cost_series_from_rebalance_result(
        ctx=ctx, frame=frame, rebalance_result=result
    )
    assert np.isclose(row["total_cost"], float(canonical_total.sum()))

    bogus_turnover = 0.5 * (abs(0.20 - 0.90) + abs(0.00 - (-0.90)))
    assert not np.isclose(row["total_turnover"], bogus_turnover)


def test_turnover_decomposition_sums_to_authoritative_turnover():
    result = _result({"A": 0.10, "B": 0.10, "C": 0.00}, {"A": 0.20, "B": 0.00, "C": 0.10})
    decomp = PortfolioDiagnostics.turnover_decomposition_from_rebalance_result(result)
    assert np.isclose(decomp["total_turnover"], result.turnover_oneway)
    assert np.isclose(
        decomp["entry_turnover"] + decomp["exit_turnover"] + decomp["resize_turnover"],
        result.turnover_oneway,
    )
