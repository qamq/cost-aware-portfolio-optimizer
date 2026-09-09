"""Final fail-closed and cross-module edge cases for the portfolio research stack."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cost_aware_portfolio.optimizer import (
    CostAwareParams,
    compute_trade_cost_breakdown,
    solve_cost_aware_nonlinear_pg,
    solve_rebalance_to_target_nonlinear_pg,
)
from cost_aware_portfolio.covariance import CovarianceEngine
from cost_aware_portfolio.state import PreTradeState, build_rebalance_result
from cost_aware_portfolio.fingerprint import build_run_signature
from cost_aware_portfolio.validation import performance_stats
from cost_aware_portfolio.strategies import PortfolioStrategy, TCOptimizerMixin, VWStrategy


class _DummyTCStrategy(TCOptimizerMixin, PortfolioStrategy):
    pass


def _cov_engine():
    return CovarianceEngine(
        mw_window=8,
        mw_min_periods=3,
        mw_ridge=1e-6,
        cov_mode="rolling",
        ewma_alpha=0.1,
        ewma_center=False,
    )





def test_degenerate_tail_signal_fails_closed():
    reb = pd.DataFrame({"up_prob": [0.5] * 20})
    with pytest.raises(RuntimeError, match="degenerate signal tails"):
        PortfolioStrategy._require_tail_thresholds(reb, date=pd.Timestamp("2024-01-05"))


def test_value_weighting_never_substitutes_equal_weights_for_bad_market_cap():
    frame = pd.DataFrame({"MarketCap": [0.0, 0.0]}, index=["A", "B"])
    with pytest.raises(RuntimeError, match="MarketCap sum is non-positive"):
        PortfolioStrategy._strict_value_weights(
            frame, date=pd.Timestamp("2024-01-05"), label="VW"
        )

    missing = pd.DataFrame(index=["A", "B"])
    with pytest.raises(RuntimeError, match="MarketCap is unavailable"):
        PortfolioStrategy._strict_value_weights(
            missing, date=pd.Timestamp("2024-01-05"), label="VW"
        )


def test_tc_execution_support_has_no_screened_panel_fallback():
    strategy = _DummyTCStrategy(SimpleNamespace())
    selection = pd.DataFrame({"up_prob": [0.2, 0.8]}, index=["A", "B"])
    with pytest.raises(RuntimeError, match="requires ctx._tc_execution_frame"):
        strategy._tc_execution_support_frame(
            ctx=SimpleNamespace(),
            date=pd.Timestamp("2024-01-05"),
            selection_frame=selection,
        )


def test_covariance_all_invalid_diagonal_fails_closed():
    engine = _cov_engine()
    bad = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=float)
    with pytest.raises(RuntimeError, match="no positive finite variances"):
        engine._stabilize_covariance(bad)


def test_covariance_partial_corruption_is_repaired_and_psd():
    engine = _cov_engine()
    raw = np.array([[0.04, np.nan], [np.nan, 0.09]], dtype=float)
    out = engine._stabilize_covariance(raw)
    assert np.isfinite(out).all()
    assert np.allclose(out, out.T)
    assert np.min(np.linalg.eigvalsh(out.astype(float))) > 0.0


def test_optimizer_exposes_solver_status_and_warnings_are_not_suppressed():
    idx = pd.Index(["A", "B"])
    alpha = pd.Series([1.0, 0.0], index=idx)
    prev = pd.Series([0.5, 0.5], index=idx)
    adv = pd.Series([1e9, 1e9], index=idx)
    sigma = pd.Series([0.02, 0.02], index=idx)
    spreads = pd.Series([0.0, 0.0], index=idx)
    params = CostAwareParams(
        lambda_risk=0.0,
        eta_tc=0.0,
        eta_linear=0.0,
        eta_impact=0.0,
        long_only=True,
        gross_target=1.0,
        pos_adv_cap=np.inf,
        trade_adv_cap=np.inf,
        box_abs=1.0,
        max_iters=50,
    )
    with pytest.warns(RuntimeWarning, match="outside the intended strongly-convex regime"):
        w = solve_cost_aware_nonlinear_pg(
            alpha=alpha,
            Sigma=np.eye(2),
            w_prev=prev,
            adv=adv,
            sigma=sigma,
            aum=100e6,
            spreads_oneway=spreads,
            params=params,
        )
    assert "solver_method" in w.attrs
    assert "solver_converged" in w.attrs
    assert "solver_fallback_used" in w.attrs


def test_transaction_cost_is_reflected_in_carried_state_weights():
    ctx = SimpleNamespace(tc_aum_dollars=100.0)
    strategy = _DummyTCStrategy(ctx)
    prev_state = {}
    ret_name = "next_week_ret"

    pre0 = strategy._tc_pretrade_state(
        ctx=ctx,
        prev_state=prev_state,
        state_key="leg",
        date=pd.Timestamp("2024-01-05"),
    )
    frame = pd.DataFrame({ret_name: [0.10]}, index=["A"])
    strategy._tc_store_leg_state(
        prev_state=prev_state,
        state_key="leg",
        pretrade_state=pre0,
        posttrade_weights=pd.Series({"A": 1.0}),
        frame=frame,
        ret_name=ret_name,
        transaction_cost=0.01,
    )

    # Net NAV is 100 * (1 + 10% gross return - 1% cost) = 109.
    ctx.tc_aum_dollars = 109.0
    pre1 = strategy._tc_pretrade_state(
        ctx=ctx,
        prev_state=prev_state,
        state_key="leg",
        date=pd.Timestamp("2024-01-12"),
    )
    # The risky dollar holding is still 110, so its weight against net NAV is 110/109.
    assert pre1.weights["A"] == pytest.approx(110.0 / 109.0)
    assert pre1.cash_weight == pytest.approx(-1.0 / 109.0)
    assert pre1.weights["A"] * 109.0 == pytest.approx(110.0)


def test_run_signature_changes_with_signal_and_economic_kwargs():
    signal = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
         "StockID": ["A", "B"], "up_prob": [0.2, 0.8]}
    )
    base = build_run_signature(
        run_kind="test",
        model="HMM",
        signal_df=signal,
        config_fields={"aum": 100e6},
        portfolio_kwargs={"tc_trade_adv_cap": 0.25},
        manager_identity="manager",
    )
    same = build_run_signature(
        run_kind="test",
        model="HMM",
        signal_df=signal.copy(),
        config_fields={"aum": 100e6},
        portfolio_kwargs={"tc_trade_adv_cap": 0.25},
        manager_identity="manager",
    )
    changed_signal = signal.copy()
    changed_signal.loc[0, "up_prob"] = 0.3
    changed = build_run_signature(
        run_kind="test",
        model="HMM",
        signal_df=changed_signal,
        config_fields={"aum": 100e6},
        portfolio_kwargs={"tc_trade_adv_cap": 0.25},
        manager_identity="manager",
    )
    changed_kw = build_run_signature(
        run_kind="test",
        model="HMM",
        signal_df=signal,
        config_fields={"aum": 100e6},
        portfolio_kwargs={"tc_trade_adv_cap": 0.10},
        manager_identity="manager",
    )
    assert base == same
    assert base != changed
    assert base != changed_kw


def test_randomized_rebalance_accounting_identities():
    rng = np.random.RandomState(123)
    for k in range(100):
        n = int(rng.randint(2, 15))
        idx = pd.Index(["S%d" % i for i in range(n)])
        pre_raw = rng.uniform(0.0, 1.0, n)
        pre = pd.Series(0.8 * pre_raw / pre_raw.sum(), index=idx)
        post_raw = rng.uniform(0.0, 1.0, n)
        post = pd.Series(0.8 * post_raw / post_raw.sum(), index=idx)
        aum = float(rng.uniform(1e6, 1e9))
        state = PreTradeState(
            date=pd.Timestamp("2024-01-05") + pd.Timedelta(days=k),
            aum=aum,
            weights=pre,
            cash_weight=1.0 - float(pre.sum()),
        )
        result = build_rebalance_result(pretrade_state=state, posttrade_weights=post)
        expected_delta = result.posttrade_weights - result.pretrade_weights
        assert np.allclose(result.delta_w.to_numpy(), expected_delta.to_numpy())
        assert np.allclose(result.dollar_trades.to_numpy(), aum * expected_delta.to_numpy())
        assert result.turnover_oneway == pytest.approx(0.5 * float(expected_delta.abs().sum()))
        assert result.posttrade_cash_weight + float(result.posttrade_weights.sum()) == pytest.approx(1.0)


def test_small_cross_module_execution_accounting_pipeline():
    idx = pd.Index(["A", "B"])
    aum = 100e6
    pre = pd.Series([0.0, 0.0], index=idx)
    target = pd.Series([0.6, 0.4], index=idx)
    adv = pd.Series([500e6, 400e6], index=idx)
    sigma = pd.Series([0.02, 0.025], index=idx)
    spread = pd.Series([0.0005, 0.0007], index=idx)
    params = CostAwareParams(
        lambda_risk=0.0,
        eta_tc=0.0,
        eta_linear=1.0,
        eta_impact=1.0,
        impact_kappa=1.0,
        long_only=True,
        gross_target=1.0,
        pos_adv_cap=np.inf,
        trade_adv_cap=np.inf,
        box_abs=1.0,
        target_penalty=1000.0,
        max_iters=3000,
    )
    w = solve_rebalance_to_target_nonlinear_pg(
        w_target=target,
        w_prev=pre,
        adv=adv,
        sigma=sigma,
        aum=aum,
        spreads_oneway=spread,
        params=params,
    )
    state = PreTradeState(
        date=pd.Timestamp("2024-01-05"),
        aum=aum,
        weights=pre,
        cash_weight=1.0,
    )
    result = build_rebalance_result(pretrade_state=state, posttrade_weights=w)
    cost = compute_trade_cost_breakdown(
        w=result.posttrade_weights,
        w_prev=result.pretrade_weights,
        spreads_oneway=spread,
        sigma=sigma,
        adv=adv,
        aum=aum,
    )
    asset_ret = pd.Series([0.01, -0.005], index=idx)
    gross = float((result.posttrade_weights * asset_ret).sum())
    net = gross - float(cost.total_cost)
    stats = performance_stats(pd.Series([net, net * 0.5, -net * 0.25, net * 0.75]))

    assert np.isfinite(gross)
    assert np.isfinite(cost.total_cost)
    assert net == pytest.approx(gross - cost.total_cost)
    assert stats["n_periods"] == 4
    assert np.isfinite(stats["sharpe"])
