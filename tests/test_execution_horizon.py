"""Tests for the explicit execution-horizon convention in transaction costs."""

import numpy as np
import pandas as pd
import pytest

from cost_aware_portfolio.optimizer import (
    CostAwareParams,
    compute_long_only_feasible_interval,
    compute_trade_cost_breakdown,
)


def _interval(execution_days):
    params = CostAwareParams(
        long_only=True,
        dollar_neutral=False,
        pos_adv_cap=np.inf,
        trade_adv_cap=0.10,
        execution_days=execution_days,
        box_abs=np.inf,
    )
    return compute_long_only_feasible_interval(
        w_prev=pd.Series({"A": 0.0, "B": 0.0}),
        adv=pd.Series({"A": 20_000_000.0, "B": 20_000_000.0}),
        aum=100_000_000.0,
        params=params,
    )


def test_one_day_trade_cap_is_fraction_of_one_day_adv():
    lo, hi = _interval(1.0)
    assert np.isclose(lo, 0.0)
    # Each stock: 10% participation * $20m daily ADV / $100m NAV = 2%.
    assert np.isclose(hi, 0.04)


def test_five_day_execution_expands_trade_capacity_exactly_fivefold():
    _, hi_1d = _interval(1.0)
    _, hi_5d = _interval(5.0)
    assert np.isclose(hi_5d, 5.0 * hi_1d)


def test_position_adv_cap_does_not_scale_with_execution_days():
    def maximum(days):
        params = CostAwareParams(
            long_only=True,
            dollar_neutral=False,
            pos_adv_cap=0.50,
            trade_adv_cap=np.inf,
            execution_days=days,
            box_abs=np.inf,
        )
        return compute_long_only_feasible_interval(
            w_prev=pd.Series({"A": 0.0, "B": 0.0}),
            adv=pd.Series({"A": 20_000_000.0, "B": 20_000_000.0}),
            aum=100_000_000.0,
            params=params,
        )[1]

    assert np.isclose(maximum(1.0), maximum(5.0))
    assert np.isclose(maximum(1.0), 0.20)


def test_impact_uses_execution_window_volume():
    common = dict(
        w=pd.Series({"A": 0.10}),
        w_prev=pd.Series({"A": 0.0}),
        spreads_oneway=pd.Series({"A": 0.0}),
        sigma=pd.Series({"A": 0.02}),
        adv=pd.Series({"A": 20_000_000.0}),
        aum=100_000_000.0,
        impact_kappa=1.0,
        impact_exponent=1.5,
    )
    one = compute_trade_cost_breakdown(execution_days=1.0, **common)
    four = compute_trade_cost_breakdown(execution_days=4.0, **common)
    assert np.isclose(four.impact_cost, 0.5 * one.impact_cost)


def test_default_execution_days_preserves_one_day_formula():
    out = compute_trade_cost_breakdown(
        w=pd.Series({"A": 0.10}),
        w_prev=pd.Series({"A": 0.0}),
        spreads_oneway=pd.Series({"A": 0.0}),
        sigma=pd.Series({"A": 0.02}),
        adv=pd.Series({"A": 20_000_000.0}),
        aum=100_000_000.0,
    )
    expected = 0.02 * np.sqrt(100_000_000.0 / 20_000_000.0) * (0.10 ** 1.5)
    assert np.isclose(out.impact_cost, expected)


def test_cost_params_reject_nonpositive_execution_days():
    params = CostAwareParams(execution_days=0.0)
    with pytest.raises(ValueError, match="execution_days must be a finite positive"):
        params.resolved_execution_days()

