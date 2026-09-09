"""Unit tests for the dynamic-NAV transaction-cost runner."""

import numpy as np
import pandas as pd
import pytest

import cost_aware_portfolio.dynamic_nav as dyn


class FakeStrategy:
    def __init__(self, ctx):
        self.ctx = ctx
        self.aum_seen = []
        self.dollar_trade_seen = []

    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        nav = float(self.ctx.tc_aum_dollars)
        self.aum_seen.append(nav)
        # A simple 10% weight trade proxy proves dollar scaling uses current NAV.
        self.dollar_trade_seen.append(0.10 * nav)

        net_return = float(self.ctx.net_returns_by_date[pd.Timestamp(date)])
        row = np.zeros(cut, dtype=float)
        row[cut - 1] = net_return
        to_df = pd.DataFrame({"weight": [1.0]}, index=["A"])
        return row, to_df, None


class FakePortfolioManager:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tc_aum_dollars = float(kwargs["tc_aum_dollars"])
        self.net_returns_by_date = {
            pd.Timestamp("2024-01-05"): 0.10,
            pd.Timestamp("2024-01-12"): -0.10,
            pd.Timestamp("2024-01-19"): 0.00,
        }
        self.cost_by_date = {
            pd.Timestamp("2024-01-05"): 0.01,
            pd.Timestamp("2024-01-12"): 0.02,
            pd.Timestamp("2024-01-19"): 0.00,
        }
        self.strategy = FakeStrategy(self)
        self.aum_seen_after_strategy_return = []
        self._last_tc_debug_df = None
        self._last_tc_debug_summary = None
        self._last_portfolio_debug_df = None
        self._last_portfolio_debug_summary = None
        FakePortfolioManager.instances.append(self)

    def _get_strategy(self, weight_type):
        return self.strategy

    def calculate_portfolio_rets(self, weight_type, cut, delay):
        dates = pd.DatetimeIndex(sorted(self.net_returns_by_date.keys()))
        strategy = self._get_strategy(weight_type)
        prev_state = {}
        rows = []
        tc_rows = []

        for date in dates:
            row, _, _ = strategy.compute_for_date(None, date, cut, "ret", prev_state)
            rows.append(row)

            # The proxy must leave this date's BEGINNING NAV in the manager until
            # after normal manager diagnostics have been collected.
            self.aum_seen_after_strategy_return.append(float(self.tc_aum_dollars))

            cost = float(self.cost_by_date[date])
            tc_rows.append(
                {
                    "date": date,
                    "linear_cost": 0.25 * cost,
                    "impact_cost": 0.75 * cost,
                    "total_cost": cost,
                }
            )

        arr = np.asarray(rows, dtype=float)
        pf = pd.DataFrame(arr, index=dates, columns=list(range(cut)))
        pf["H-L"] = pf[cut - 1] - pf[0]

        self._last_tc_debug_df = pd.DataFrame(tc_rows)
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
        }
        self._last_portfolio_debug_df = pd.DataFrame({"date": dates})
        self._last_portfolio_debug_summary = {
            "total_turnover_mean": 0.33,
            "any_pos_cap_bind_pct": 0.25,
            "any_trade_cap_bind_pct": 0.75,
            "pos_cap_bind_weight_share_mean": 0.05,
            "trade_cap_bind_weight_share_mean": 0.20,
        }
        return pf, 0.33  # manager return and canonical diagnostic turnover must agree


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


def _config(tmp_path, **kwargs):
    base = dict(
        output_dir=str(tmp_path),
        initial_nav=100.0,
        start_year=2024,
        end_year=2024,
        eval_start_year=2024,
        resume=False,
        verbose=False,
    )
    base.update(kwargs)
    return dyn.DynamicNAVConfig(**base)


def test_default_execution_convention_and_validation(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.execution_days == 1.0
    assert cfg.weight_type == "mw_h_plus_lowneg_tc_nl"

    with pytest.raises(ValueError, match="initial_nav must be finite and strictly positive"):
        _config(tmp_path, initial_nav=0.0)


def test_nav_compounds_and_current_nav_is_passed_to_each_rebalance(tmp_path):
    FakePortfolioManager.instances = []
    result = dyn.run_one_dynamic_nav(
        _signal(),
        model="HMM",
        config=_config(tmp_path),
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )

    run = result.nav_accounting
    assert np.allclose(run["nav_start"].to_numpy(), [100.0, 110.0, 99.0])
    assert np.allclose(run["nav_end"].to_numpy(), [110.0, 99.0, 99.0])
    assert np.isclose(result.final_nav, 99.0)

    pm = FakePortfolioManager.instances[-1]
    assert np.allclose(pm.strategy.aum_seen, [100.0, 110.0, 99.0])
    assert np.allclose(pm.strategy.dollar_trade_seen, [10.0, 11.0, 9.9])


def test_manager_diagnostics_see_current_not_next_nav(tmp_path):
    FakePortfolioManager.instances = []
    dyn.run_one_dynamic_nav(
        _signal(),
        model="RF",
        config=_config(tmp_path),
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )
    pm = FakePortfolioManager.instances[-1]
    assert np.allclose(pm.aum_seen_after_strategy_return, [100.0, 110.0, 99.0])


def test_cost_is_not_subtracted_twice_and_dollar_accounting_closes(tmp_path):
    result = dyn.run_one_dynamic_nav(
        _signal(),
        model="HMM",
        config=_config(tmp_path),
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )
    d = result.nav_accounting

    # First period: strategy returned +10% NET and diagnostics report 1% cost.
    # Executed gross is therefore 11%; NAV compounds with +10%, not +9%.
    first = d.iloc[0]
    assert np.isclose(first["executed_gross_return"], 0.11)
    assert np.isclose(first["gross_pnl_dollars"], 11.0)
    assert np.isclose(first["cost_dollars"], 1.0)
    assert np.isclose(first["net_pnl_dollars"], 10.0)
    assert np.isclose(first["nav_end"], 110.0)
    assert np.isclose(first["gross_pnl_dollars"] - first["cost_dollars"], first["net_pnl_dollars"])


def test_summary_prefers_canonical_turnover_and_reports_unfilled_share(tmp_path):
    result = dyn.run_one_dynamic_nav(
        _signal(),
        model="HMM",
        config=_config(tmp_path),
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )
    assert np.isclose(result.summary["avg_turnover_oneway"], 0.33)
    assert np.isclose(result.summary["mean_unfilled_target_share"], 0.10)
    assert np.isclose(result.summary["trade_cap_bind_date_pct"], 0.75)
    assert np.isclose(result.summary["eval_final_nav"], 99.0)


def test_multi_model_runner_writes_outputs_and_resume_reuses_model(tmp_path):
    FakePortfolioManager.instances = []
    cfg = dyn.DynamicNAVConfig(
        output_dir=str(tmp_path),
        initial_nav=100.0,
        start_year=2024,
        end_year=2024,
        eval_start_year=2024,
        resume=True,
        verbose=False,
    )
    result = dyn.run_dynamic_nav(
        {"HMM": _signal(), "RF": _signal()},
        config=cfg,
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )
    assert len(FakePortfolioManager.instances) == 2
    assert set(result.runs.keys()) == {"HMM", "RF"}
    assert (tmp_path / "dynamic_nav_summary.csv").exists()
    assert (tmp_path / "dynamic_nav_paths.csv").exists()
    assert (tmp_path / "dynamic_nav_manifest.json").exists()
    assert (tmp_path / "runs" / "HMM" / "dynamic_nav_weekly_accounting.csv").exists()

    dyn.run_dynamic_nav(
        {"HMM": _signal(), "RF": _signal()},
        config=cfg,
        portfolio_kwargs=_kwargs(),
        manager_cls=FakePortfolioManager,
    )
    assert len(FakePortfolioManager.instances) == 2


def test_missing_forecast_source_and_nonpositive_nav_fail_closed(tmp_path):
    cfg = _config(tmp_path)
    with pytest.raises(ValueError, match="frozen daily transaction-cost forecasts"):
        dyn.run_one_dynamic_nav(
            _signal(),
            model="RF",
            config=cfg,
            portfolio_kwargs={},
            manager_cls=FakePortfolioManager,
        )

    controller = dyn._DynamicNAVController(100.0)
    controller.begin_rebalance(ctx=type("Ctx", (), {})(), date=pd.Timestamp("2024-01-05"))
    with pytest.raises(RuntimeError, match="would become non-positive"):
        controller.finish_rebalance(
            date=pd.Timestamp("2024-01-05"),
            row=[0.0, -1.0],
            cut=2,
        )


def test_dynamic_resume_invalidates_when_signal_or_portfolio_kwargs_change(tmp_path):
    FakePortfolioManager.instances = []
    cfg = dyn.DynamicNAVConfig(
        output_dir=str(tmp_path),
        initial_nav=100.0,
        start_year=2024,
        end_year=2024,
        eval_start_year=2024,
        resume=True,
        verbose=False,
    )
    sig = _signal()
    dyn.run_dynamic_nav(
        {"HMM": sig}, config=cfg, portfolio_kwargs=_kwargs(), manager_cls=FakePortfolioManager
    )
    assert len(FakePortfolioManager.instances) == 1

    changed_signal = sig.copy()
    changed_signal.loc[changed_signal.index[0], "up_prob"] = 0.25
    dyn.run_dynamic_nav(
        {"HMM": changed_signal}, config=cfg, portfolio_kwargs=_kwargs(), manager_cls=FakePortfolioManager
    )
    assert len(FakePortfolioManager.instances) == 2

    changed_kwargs = _kwargs()
    changed_kwargs["tc_trade_adv_cap"] = 0.10
    dyn.run_dynamic_nav(
        {"HMM": changed_signal}, config=cfg, portfolio_kwargs=changed_kwargs, manager_cls=FakePortfolioManager
    )
    assert len(FakePortfolioManager.instances) == 3
