from types import SimpleNamespace

import numpy as np
import pandas as pd

from cost_aware_portfolio.capacity import FixedAUMCapacityConfig, _build_run_summary
from cost_aware_portfolio.dynamic_nav import DynamicNAVConfig, _build_summary as _build_dynamic_summary
from cost_aware_portfolio.diagnostics import PortfolioDiagnostics
from cost_aware_portfolio.state import PreTradeState, build_rebalance_result
from cost_aware_portfolio.transaction_costs import TC_ADV_DAILY_COL, TC_SIGMA_DAILY_COL


def _frame(ids):
    idx = pd.Index(ids, dtype=object)
    return pd.DataFrame(
        {
            TC_ADV_DAILY_COL: [50_000_000.0] * len(idx),
            TC_SIGMA_DAILY_COL: [0.02] * len(idx),
            "spread_oneway": [0.001] * len(idx),
        },
        index=idx,
    )


def _result(date, weights, *, aum=100_000_000.0):
    state = PreTradeState(
        date=pd.Timestamp(date),
        aum=float(aum),
        weights=pd.Series(dtype=float),
        cash_weight=1.0,
    )
    return build_rebalance_result(
        pretrade_state=state,
        posttrade_weights=pd.Series(weights, dtype=float),
    )


def test_paired_tc_debug_reports_realized_cash_and_unfilled_exposure():
    ctx = SimpleNamespace(tc_execution_days=1.0, tc_use_forecast_inputs=True)
    prev_state = {}
    low_frame = _frame(["L1", "L2"])
    high_frame = _frame(["H1", "H2"])

    PortfolioDiagnostics.append_paired_tc_debug_row(
        ctx=ctx,
        prev_state=prev_state,
        date=pd.Timestamp("2024-01-05"),
        common_target_sum=0.80,
        low_frame=low_frame,
        high_frame=high_frame,
        low_weights_abs=pd.Series({"L1": 0.40, "L2": 0.35}),
        high_weights=pd.Series({"H1": 0.40, "H2": 0.40}),
        low_cost=0.001,
        high_cost=0.002,
        low_trad_mask=pd.Series(True, index=low_frame.index),
        high_trad_mask=pd.Series(True, index=high_frame.index),
    )

    row = prev_state["tc_debug_rows"][0]
    assert np.isclose(row["low_posttrade_cash_weight"], 0.25)
    assert np.isclose(row["high_posttrade_cash_weight"], 0.20)
    assert np.isclose(row["low_unfilled_target_share"], 0.25)
    assert np.isclose(row["high_unfilled_target_share"], 0.20)
    assert np.isclose(row["mean_unfilled_target_share"], 0.225)
    assert np.isclose(row["capacity_target_shortfall"], 0.20)
    assert np.isclose(row["low_execution_shortfall_vs_common_target"], 0.05)
    assert np.isclose(row["high_execution_shortfall_vs_common_target"], 0.0)

    _, summary = PortfolioDiagnostics.summarize_tc_debug("ew_tc", prev_state["tc_debug_rows"])
    assert np.isclose(summary["mean_unfilled_target_share"], 0.225)
    assert np.isclose(summary["low_posttrade_cash_weight_mean"], 0.25)
    assert np.isclose(summary["high_posttrade_cash_weight_mean"], 0.20)


def test_tc_portfolio_summary_does_not_subtract_cost_twice():
    dates = pd.to_datetime(["2024-01-05", "2024-01-12"])
    # Active TC strategy return stream is already net of realized costs.
    pf = pd.DataFrame({0: [0.0, 0.0], 9: [0.01, 0.02], "H-L": [0.01, 0.02]}, index=dates)
    debug_rows = [
        {"date": dates[0], "total_cost": 0.002},
        {"date": dates[1], "total_cost": 0.003},
    ]

    _, summary = PortfolioDiagnostics.summarize_portfolio_debug(
        "ew_tc",
        debug_rows,
        portfolio_ret_eval=pf,
        freq="week",
    )

    expected_net_ann = np.mean([0.01, 0.02]) * 52.0
    expected_gross_ann = np.mean([0.012, 0.023]) * 52.0
    assert np.isclose(summary["shadow_net_hl_ann_ret"], expected_net_ann)
    assert np.isclose(summary["gross_hl_ann_ret"], expected_gross_ann)



def test_capacity_summary_prefers_actual_unfilled_metric_over_target_sum_proxy(tmp_path):
    cfg = FixedAUMCapacityConfig(output_dir=str(tmp_path), aum_grid=(100_000_000.0,), resume=False)
    idx = pd.to_datetime(["2024-01-05", "2024-01-12"])
    net = pd.Series([0.01, 0.02], index=idx)
    cost = pd.Series([0.001, 0.001], index=idx)
    row = _build_run_summary(
        model="x",
        aum=100_000_000.0,
        config=cfg,
        net=net,
        total_cost=cost,
        linear_cost=0.5 * cost,
        impact_cost=0.5 * cost,
        avg_turnover=0.1,
        tc_summary={
            "common_target_sum_mean": 0.80,
            "mean_unfilled_target_share": 0.25,
            "low_unfilled_target_share_mean": 0.30,
            "high_unfilled_target_share_mean": 0.20,
        },
        pf_summary={},
        run_signature="test",
    )
    assert np.isclose(row["mean_unfilled_target_share"], 0.25)
    assert np.isclose(row["low_unfilled_target_share_mean"], 0.30)
    assert np.isclose(row["high_unfilled_target_share_mean"], 0.20)


def test_dynamic_summary_prefers_actual_unfilled_metric_over_target_sum_proxy(tmp_path):
    cfg = DynamicNAVConfig(output_dir=str(tmp_path), resume=False)
    accounting = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            "nav_start": [100.0, 101.0],
            "nav_end": [101.0, 102.01],
            "net_return": [0.01, 0.01],
            "executed_gross_return": [0.011, 0.011],
            "total_cost": [0.001, 0.001],
            "linear_cost": [0.0005, 0.0005],
            "impact_cost": [0.0005, 0.0005],
            "cost_dollars": [0.1, 0.101],
            "is_evaluation": [True, True],
        }
    )
    row = _build_dynamic_summary(
        model="x",
        config=cfg,
        accounting=accounting,
        avg_turnover=0.1,
        tc_summary={
            "common_target_sum_mean": 0.80,
            "mean_unfilled_target_share": 0.25,
            "low_unfilled_target_share_mean": 0.30,
            "high_unfilled_target_share_mean": 0.20,
        },
        pf_summary={"total_turnover_mean": 0.1},
        run_signature="test",
    )
    assert np.isclose(row["mean_unfilled_target_share"], 0.25)
    assert np.isclose(row["low_unfilled_target_share_mean"], 0.30)
    assert np.isclose(row["high_unfilled_target_share_mean"], 0.20)
