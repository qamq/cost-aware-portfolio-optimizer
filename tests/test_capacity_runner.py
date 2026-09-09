"""Unit tests for the fixed-AUM capacity runner."""

import numpy as np
import pandas as pd
import pytest

import cost_aware_portfolio.capacity as cap




class FakePortfolioManager:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakePortfolioManager.calls.append(kwargs)
        self._last_tc_debug_df = None
        self._last_tc_debug_summary = None
        self._last_portfolio_debug_df = None
        self._last_portfolio_debug_summary = None

    def calculate_portfolio_rets(self, weight_type, cut, delay):
        dates = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"])
        net = pd.Series([0.010, 0.020, -0.010, 0.000], index=dates)
        total_cost = pd.Series([0.001, 0.001, 0.001, 0.001], index=dates)
        linear = pd.Series([0.0002] * 4, index=dates)
        impact = total_cost - linear

        self._last_tc_debug_df = pd.DataFrame(
            {
                "date": dates,
                "linear_cost": linear.to_numpy(),
                "impact_cost": impact.to_numpy(),
                "total_cost": total_cost.to_numpy(),
            }
        )
        self._last_tc_debug_summary = {
            "common_target_sum_mean": 0.90,
            "low_realized_gross_mean": 0.90,
            "high_realized_gross_mean": 0.90,
            "mean_unfilled_target_share": 0.10,
            "low_unfilled_target_share_mean": 0.10,
            "high_unfilled_target_share_mean": 0.10,
            "low_posttrade_cash_weight_mean": 0.10,
            "high_posttrade_cash_weight_mean": 0.10,
            "low_tradable_share_mean": 1.0,
            "high_tradable_share_mean": 1.0,
            "total_cost_bps_median": 10.0,
            "total_cost_bps_p90": 10.0,
            "total_cost_bps_p95": 10.0,
        }
        self._last_portfolio_debug_summary = {
            "any_pos_cap_bind_pct": 0.25,
            "any_trade_cap_bind_pct": 0.75,
            "pos_cap_bind_weight_share_mean": 0.05,
            "trade_cap_bind_weight_share_mean": 0.20,
        }
        self._last_portfolio_debug_df = pd.DataFrame({"date": dates})

        pf = pd.DataFrame({0: 0.0, 9: net.to_numpy(), "H-L": net.to_numpy()}, index=dates)
        return pf, 0.40


def _signal():
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
            "StockID": ["A", "B"],
            "up_prob": [0.2, 0.8],
        }
    )


def _kwargs():
    return {
        "tc_forecast_df": pd.DataFrame({"dummy": [1]}),
        "tc_trade_adv_cap": 0.25,
        "tc_pos_adv_cap": 0.50,
    }


def test_default_grid_and_execution_convention(tmp_path):
    cfg = cap.FixedAUMCapacityConfig(output_dir=str(tmp_path))
    assert cfg.aum_grid == cap.DEFAULT_AUM_GRID
    assert cfg.execution_days == 1.0
    assert cfg.weight_type == "mw_h_plus_lowneg_tc_nl"


def test_run_reoptimizes_at_each_aum_and_reconstructs_executed_gross(tmp_path):
    FakePortfolioManager.calls = []
    cfg = cap.FixedAUMCapacityConfig(
        output_dir=str(tmp_path),
        aum_grid=[10e6, 100e6],
        start_year=2024,
        end_year=2024,
        eval_start_year=2024,
        resume=False,
        verbose=False,
    )
    result = cap.run_fixed_aum_capacity(
        {"HMM": _signal()},
        config=cfg,
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )

    assert len(FakePortfolioManager.calls) == 2
    assert [x["tc_aum_dollars"] for x in FakePortfolioManager.calls] == [10e6, 100e6]
    assert all(x["tc_execution_days"] == 1.0 for x in FakePortfolioManager.calls)
    assert all(x["tc_use_nonlinear"] is True for x in FakePortfolioManager.calls)
    assert all(x["tc_use_forecast_inputs"] is True for x in FakePortfolioManager.calls)

    row = result.summary[result.summary["aum_dollars"] == 10e6].iloc[0]
    assert np.isclose(row["total_cost_bps_per_rebalance"], 10.0)
    assert np.isclose(row["linear_cost_bps_per_rebalance"], 2.0)
    assert np.isclose(row["impact_cost_bps_per_rebalance"], 8.0)
    assert np.isclose(row["mean_unfilled_target_share"], 0.10)
    assert np.isclose(row["trade_cap_bind_date_pct"], 0.75)
    assert np.isclose(row["avg_turnover_oneway"], 0.40)

    run = result.runs[("HMM", 10e6)]
    assert np.allclose(
        run.executed_gross_returns.to_numpy(),
        run.net_returns.to_numpy() + run.total_cost.to_numpy(),
    )


def test_runner_writes_summary_pivots_and_manifest(tmp_path):
    cfg = cap.FixedAUMCapacityConfig(
        output_dir=str(tmp_path),
        aum_grid=[25e6],
        start_year=2024,
        end_year=2024,
        eval_start_year=2024,
        resume=False,
        verbose=False,
    )
    cap.run_fixed_aum_capacity(
        {"RF": _signal()},
        config=cfg,
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )
    assert (tmp_path / "fixed_aum_capacity_summary.csv").exists()
    assert (tmp_path / "fixed_aum_net_sharpe.csv").exists()
    assert (tmp_path / "fixed_aum_capacity_manifest.json").exists()
    assert (tmp_path / "runs" / "RF" / "aum_25000000" / "weekly_accounting.csv").exists()


def test_resume_reuses_completed_run(tmp_path):
    FakePortfolioManager.calls = []
    cfg_first = cap.FixedAUMCapacityConfig(
        output_dir=str(tmp_path),
        aum_grid=[50e6],
        start_year=2024,
        end_year=2024,
        eval_start_year=2024,
        resume=True,
        verbose=False,
    )
    cap.run_fixed_aum_capacity(
        {"HMM": _signal()},
        config=cfg_first,
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )
    assert len(FakePortfolioManager.calls) == 1

    cap.run_fixed_aum_capacity(
        {"HMM": _signal()},
        config=cfg_first,
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )
    assert len(FakePortfolioManager.calls) == 1


def test_missing_frozen_forecast_source_fails_closed(tmp_path):
    cfg = cap.FixedAUMCapacityConfig(
        output_dir=str(tmp_path),
        aum_grid=[10e6],
        start_year=2024,
        end_year=2024,
        eval_start_year=2024,
        resume=False,
        verbose=False,
    )
    with pytest.raises(ValueError, match="frozen daily transaction-cost forecasts"):
        cap.run_fixed_aum_capacity(
            {"RF": _signal()},
            config=cfg,
            portfolio_kwargs={},
            manager_cls=FakePortfolioManager,
        )


def test_duplicate_or_missing_cost_dates_fail_closed(tmp_path):
    idx = pd.to_datetime(["2024-01-05", "2024-01-12"])
    dup = pd.DataFrame(
        {
            "date": [idx[0], idx[0]],
            "total_cost": [0.001, 0.001],
        }
    )
    with pytest.raises(RuntimeError, match="duplicate dates"):
        cap._debug_series(dup, idx, "total_cost", required=True)

    missing = pd.DataFrame({"date": [idx[0]], "total_cost": [0.001]})
    with pytest.raises(RuntimeError, match="incomplete"):
        cap._debug_series(missing, idx, "total_cost", required=True)


def test_resume_invalidates_when_signal_or_portfolio_kwargs_change(tmp_path):
    FakePortfolioManager.calls = []
    cfg = cap.FixedAUMCapacityConfig(
        output_dir=str(tmp_path),
        aum_grid=[50e6],
        start_year=2024,
        end_year=2024,
        eval_start_year=2024,
        resume=True,
        verbose=False,
    )
    sig = _signal()
    cap.run_fixed_aum_capacity(
        {"HMM": sig}, config=cfg, portfolio_kwargs=_kwargs(), manager_cls=FakePortfolioManager
    )
    assert len(FakePortfolioManager.calls) == 1

    changed_signal = sig.copy()
    changed_signal.loc[changed_signal.index[0], "up_prob"] = 0.25
    cap.run_fixed_aum_capacity(
        {"HMM": changed_signal}, config=cfg, portfolio_kwargs=_kwargs(), manager_cls=FakePortfolioManager
    )
    assert len(FakePortfolioManager.calls) == 2

    changed_kwargs = _kwargs()
    changed_kwargs["tc_trade_adv_cap"] = 0.10
    cap.run_fixed_aum_capacity(
        {"HMM": changed_signal}, config=cfg, portfolio_kwargs=changed_kwargs, manager_cls=FakePortfolioManager
    )
    assert len(FakePortfolioManager.calls) == 3
