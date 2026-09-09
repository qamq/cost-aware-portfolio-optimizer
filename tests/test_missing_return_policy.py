"""Regression tests for bounded missing-return handling in stateful TC portfolios."""

import numpy as np
import pandas as pd

from cost_aware_portfolio.strategies import PortfolioStrategy, TCOptimizerMixin


class _MissingReturnContext:
    tc_aum_dollars = 100_000_000.0
    tc_missing_return_max_carry_weeks = 4
    tc_missing_return_warn_weight = 0.005
    verbose = False


class _DummyStrategy(TCOptimizerMixin, PortfolioStrategy):
    pass


def _store(strategy, prev_state, date, returns, weights):
    ret_name = "next_week_ret"
    pretrade = strategy._tc_pretrade_state(
        ctx=strategy.ctx,
        prev_state=prev_state,
        state_key="leg",
        date=pd.Timestamp(date),
    )
    frame = pd.DataFrame(
        {ret_name: pd.Series(returns, dtype=float)},
        index=pd.Index(list(returns.keys()), dtype=object),
    )
    # Constructing from a dict inside DataFrame can align unexpectedly; assign
    # explicitly to preserve the requested stock order and values.
    frame[ret_name] = [returns[sid] for sid in frame.index]
    return strategy._tc_store_leg_state(
        prev_state=prev_state,
        state_key="leg",
        pretrade_state=pretrade,
        posttrade_weights=pd.Series(weights, dtype=float),
        frame=frame,
        ret_name=ret_name,
    )


def test_isolated_missing_return_is_marked_flat_and_carried():
    strategy = _DummyStrategy(_MissingReturnContext())
    state = {}

    result = _store(
        strategy,
        state,
        "2024-01-05",
        {"A": np.nan},
        {"A": 0.001},
    )

    saved = state["tc_leg_states"]["leg"]
    event = state["tc_missing_return_events"][-1]

    assert np.isclose(result.posttrade_weights["A"], 0.001)
    assert np.isclose(saved["posttrade_weights"]["A"], 0.001)
    assert np.isclose(saved["realized_returns"]["A"], 0.0)
    assert event["action"] == "mark_flat"
    assert event["consecutive_missing_periods"] == 1


def test_fifth_consecutive_missing_return_moves_residual_state_to_cash():
    strategy = _DummyStrategy(_MissingReturnContext())
    state = {}

    for date in pd.date_range("2024-01-05", periods=5, freq="7D"):
        result = _store(
            strategy,
            state,
            date,
            {"A": np.nan},
            {"A": 0.001},
        )

    saved = state["tc_leg_states"]["leg"]
    event = state["tc_missing_return_events"][-1]

    # The canonical rebalance result remains untouched: no artificial trade or
    # transaction cost is created by the data-resolution convention.
    assert np.isclose(result.posttrade_weights["A"], 0.001)

    # Only the state carried into the next rebalance is closed at the last mark.
    assert np.isclose(saved["posttrade_weights"]["A"], 0.0)
    assert np.isclose(saved["cash_weight"], 1.0)
    assert event["action"] == "terminal_mark_to_cash"
    assert event["consecutive_missing_periods"] == 5


def test_observed_return_resets_streak_and_aggregate_weight_drives_warning():
    strategy = _DummyStrategy(_MissingReturnContext())
    state = {}

    _store(strategy, state, "2024-02-02", {"A": np.nan}, {"A": 0.001})
    _store(strategy, state, "2024-02-09", {"A": 0.10}, {"A": 0.001})
    _store(strategy, state, "2024-02-16", {"A": np.nan}, {"A": 0.001})
    assert state["tc_missing_return_events"][-1]["consecutive_missing_periods"] == 1

    state = {}
    _store(
        strategy,
        state,
        "2024-03-01",
        {"A": np.nan, "B": np.nan},
        {"A": 0.003, "B": 0.003},
    )
    events = state["tc_missing_return_events"]
    assert np.isclose(events[0]["missing_abs_sleeve_weight_total"], 0.006)
    assert all(bool(event["materiality_warning"]) for event in events)
