import numpy as np
import pandas as pd

from cost_aware_portfolio.state import (
    build_pretrade_state,
    build_rebalance_result,
    compute_trade_vector,
    drift_posttrade_portfolio,
)


def test_union_preserves_explicit_exit_trade():
    pre = pd.Series({"A": 0.6, "B": 0.4})
    post = pd.Series({"A": 1.0})
    delta = compute_trade_vector(pre, post)
    assert set(delta.index) == {"A", "B"}
    assert np.isclose(delta["B"], -0.4)


def test_drift_satisfies_accounting_identity():
    post = pd.Series({"A": 0.5, "B": 0.5})
    rets = pd.Series({"A": 0.10, "B": -0.05})
    drift = drift_posttrade_portfolio(post, rets)
    assert np.isclose(drift.weights.sum() + drift.cash_weight, 1.0)
    assert np.isclose(drift.growth_factor, 1.025)


def test_rebalance_result_turnover_uses_single_trade_vector():
    pre = build_pretrade_state(date=pd.Timestamp("2026-01-01"), aum=100e6)
    post = pd.Series({"A": 0.6, "B": 0.4})
    result = build_rebalance_result(pretrade_state=pre, posttrade_weights=post)
    assert np.isclose(result.turnover_oneway, 0.5)
    assert np.isclose(result.total_absolute_dollar_trades, 100e6)
