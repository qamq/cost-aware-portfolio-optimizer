"""Focused regression tests for the public rebalance-state implementation."""

import numpy as np
import pandas as pd

from cost_aware_portfolio.state import (
    build_pretrade_state,
    build_rebalance_result,
    build_rebalance_universe,
    compute_trade_vector,
    drift_posttrade_portfolio,
)


def test_long_only_drift_preserves_accounting_identity():
    w = pd.Series({"A": 0.60, "B": 0.40})
    r = pd.Series({"A": 0.10, "B": -0.05})

    out = drift_posttrade_portfolio(w, r)

    assert np.isclose(out.growth_factor, 1.04)
    assert np.isclose(out.portfolio_return, 0.04)
    assert np.isclose(out.weights.sum() + out.cash_weight, 1.0)
    assert np.isclose(out.weights["A"], 0.66 / 1.04)
    assert np.isclose(out.weights["B"], 0.38 / 1.04)


def test_long_short_drift_does_not_force_target_gross():
    w = pd.Series({"LONG": 0.50, "SHORT": -0.50})
    r = pd.Series({"LONG": 0.10, "SHORT": -0.10})

    out = drift_posttrade_portfolio(w, r)

    # Portfolio return is +10%: +5% from the long and +5% from the short.
    assert np.isclose(out.portfolio_return, 0.10)
    assert np.isclose(out.growth_factor, 1.10)
    assert np.isclose(out.weights.sum() + out.cash_weight, 1.0)

    # Gross exposure drifts with NAV instead of being silently reset to 1.0.
    assert not np.isclose(out.weights.abs().sum(), w.abs().sum())


def test_legacy_holding_creates_explicit_exit_trade():
    pre = pd.Series({"A": 0.03, "B": 0.02})
    post = pd.Series({"B": 0.02, "C": 0.03})

    delta = compute_trade_vector(pre, post)

    assert set(delta.index) == {"A", "B", "C"}
    assert np.isclose(delta["A"], -0.03)
    assert np.isclose(delta["B"], 0.00)
    assert np.isclose(delta["C"], 0.03)


def test_rebalance_result_uses_one_authoritative_trade_vector():
    state = build_pretrade_state(
        date=pd.Timestamp("2024-01-12"),
        aum=100_000_000.0,
        previous_posttrade_weights=pd.Series({"A": 0.03, "B": 0.02}),
        realized_returns=pd.Series({"A": 0.20, "B": 0.00}),
        previous_cash_weight=0.95,
        previous_date=pd.Timestamp("2024-01-05"),
    )

    result = build_rebalance_result(
        pretrade_state=state,
        posttrade_weights=pd.Series({"B": 0.02, "C": 0.03}),
    )

    assert set(result.delta_w.index) == {"A", "B", "C"}
    assert np.allclose(
        result.dollar_trades.to_numpy(),
        result.aum * result.delta_w.to_numpy(),
    )
    assert np.isclose(result.turnover_oneway, 0.5 * result.delta_w.abs().sum())
    assert np.isclose(result.posttrade_weights.sum() + result.posttrade_cash_weight, 1.0)


def test_first_rebalance_starts_all_cash():
    state = build_pretrade_state(
        date=pd.Timestamp("2024-01-05"),
        aum=50_000_000.0,
    )

    assert state.weights.empty
    assert np.isclose(state.cash_weight, 1.0)
    assert np.isclose(state.growth_factor, 1.0)


def test_rebalance_universe_keeps_legacy_names():
    pre = pd.Series({"A": 0.03, "B": 0.02, "ZERO": 0.0})
    universe = build_rebalance_universe(pre, pd.Index(["B", "C"]))

    assert set(universe) == {"A", "B", "C"}
