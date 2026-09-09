"""Dynamic-NAV portfolio simulation for the transaction-cost-aware strategies.

This module answers a different question from ``capacity_runner.py``.

The fixed-AUM capacity grid holds fund size constant and asks how the same
strategy behaves at different permanent AUM levels.  This module instead lets
fund NAV evolve through time:

    NAV_{t+1} = NAV_t * (1 + R_net,t)

The current pre-trade NAV is passed into ``PortfolioManager`` before every
scheduled rebalance.  Therefore all AUM-dependent mechanics use the fund size
that actually exists on that date:

* nonlinear market impact;
* ADV-based position constraints;
* one-day ADV trade-participation constraints;
* canonical dollar trade sizes stored in rebalance state.

Execution convention
--------------------
The default specification remains the project's canonical convention:

* portfolio decisions occur once per week;
* each scheduled rebalance is executed over one trading day;
* there is no mid-week discretionary liquidation rule;
* ``tc_adv_daily`` is daily dollar ADV;
* ``tc_sigma_daily`` is daily return volatility;
* positions that cannot be fully adjusted inside the trade cap remain in the
  stateful portfolio and are carried to the next scheduled rebalance.

Accounting convention
---------------------
The active transaction-cost strategies already return H-L returns AFTER the
realized spread and market-impact cost for that rebalance.  Consequently this
runner compounds NAV directly with the returned net H-L return.  It does NOT
subtract transaction cost a second time.

For reporting only, the executed pre-cost return is reconstructed as:

    executed_gross_return_t = net_return_t + realized_cost_t

and dollar P&L is measured against beginning-of-period NAV:

    gross_pnl_dollars_t = NAV_t * executed_gross_return_t
    cost_dollars_t      = NAV_t * realized_cost_t
    net_pnl_dollars_t   = NAV_t * net_return_t

so that:

    gross_pnl_dollars_t - cost_dollars_t = net_pnl_dollars_t.

Implementation design
---------------------
No alternate optimizer is implemented here.  ``PortfolioManager`` and the
existing strategy objects remain authoritative.  A small strategy proxy sets
``ctx.tc_aum_dollars`` to the current NAV immediately before each rebalance,
then observes the realized net H-L return and schedules the resulting NAV for
the next rebalance.  The manager's diagnostics still see the same current NAV
for the date being reported; the NAV update is applied only when the next
rebalance begins.

This keeps the dynamic-NAV layer isolated from the fixed-AUM path and avoids
changing signal generation, covariance estimation, portfolio state, or cost
formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Type
import json
import math
import re

import numpy as np
import pandas as pd

from cost_aware_portfolio.fingerprint import build_run_signature, manager_identity


@dataclass(frozen=True)
class DynamicNAVConfig:
    """Configuration for one dynamic-NAV simulation."""

    output_dir: str
    initial_nav: float = 100_000_000.0
    weight_type: str = "mw_h_plus_lowneg_tc_nl"
    freq: str = "week"
    cut: int = 10
    delay: int = 0
    start_year: int = 2001
    end_year: int = 2024
    eval_start_year: int = 2001
    country: str = "USA"
    execution_days: float = 1.0
    require_tradability_screens: bool = True
    require_forecast_inputs: bool = True
    save_run_details: bool = True
    resume: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.freq not in ("week", "month", "quarter"):
            raise ValueError("freq must be one of: week, month, quarter.")
        if int(self.cut) < 2:
            raise ValueError("cut must be at least 2.")
        if int(self.start_year) > int(self.end_year):
            raise ValueError("start_year cannot exceed end_year.")
        if int(self.eval_start_year) < int(self.start_year):
            raise ValueError("eval_start_year cannot precede start_year.")
        if int(self.eval_start_year) > int(self.end_year):
            raise ValueError("eval_start_year cannot exceed end_year.")
        if (not np.isfinite(float(self.initial_nav))) or float(self.initial_nav) <= 0.0:
            raise ValueError("initial_nav must be finite and strictly positive.")
        if (not np.isfinite(float(self.execution_days))) or float(self.execution_days) <= 0.0:
            raise ValueError("execution_days must be finite and positive.")


@dataclass
class DynamicNAVRunResult:
    """Outputs for one model's dynamic-NAV simulation."""

    model: str
    initial_nav: float
    final_nav: float
    nav_accounting: pd.DataFrame
    net_returns: pd.Series
    executed_gross_returns: pd.Series
    total_cost: pd.Series
    summary: Dict[str, Any]
    output_dir: Path
    manager: Optional[Any] = None
    portfolio_returns: Optional[pd.DataFrame] = None
    tc_debug_df: Optional[pd.DataFrame] = None
    portfolio_debug_df: Optional[pd.DataFrame] = None


@dataclass
class DynamicNAVResult:
    """Collection of dynamic-NAV simulations for several signals/models."""

    summary: pd.DataFrame
    runs: Dict[str, DynamicNAVRunResult]
    output_dir: Path


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def periods_per_year(freq: str) -> int:
    """Return the annualization factor used throughout the portfolio project."""
    if freq == "week":
        return 52
    if freq == "month":
        return 12
    if freq == "quarter":
        return 4
    raise ValueError("Unsupported frequency: %s" % freq)


def annualized_stats(returns: pd.Series, freq: str) -> Dict[str, float]:
    """Compute arithmetic annualized mean, volatility, and Sharpe ratio."""
    x = pd.to_numeric(pd.Series(returns), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return {"ann_mean": float("nan"), "ann_vol": float("nan"), "sharpe": float("nan")}

    ppy = float(periods_per_year(freq))
    mean_period = float(x.mean())
    vol_period = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    ann_mean = mean_period * ppy
    ann_vol = vol_period * math.sqrt(ppy)
    sharpe = ann_mean / ann_vol if ann_vol > 0.0 and np.isfinite(ann_vol) else float("nan")
    return {"ann_mean": ann_mean, "ann_vol": ann_vol, "sharpe": sharpe}


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return text or "model"


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _normalize_dates(index: Sequence[Any]) -> pd.DatetimeIndex:
    idx = pd.to_datetime(pd.Index(index), errors="coerce")
    if pd.isna(idx).any():
        raise ValueError("Return series contains invalid dates.")
    return pd.DatetimeIndex(idx).normalize()


def _extract_hl_series(pf_ret: pd.DataFrame, model: str) -> pd.Series:
    if "H-L" not in pf_ret.columns:
        raise KeyError("%s: PortfolioManager output is missing the H-L column." % model)
    out = pd.Series(
        pd.to_numeric(pf_ret["H-L"], errors="coerce").to_numpy(dtype=float),
        index=_normalize_dates(pf_ret.index),
        name="net_return",
        dtype=float,
    )
    if out.index.has_duplicates:
        raise RuntimeError("%s: portfolio return dates are not unique." % model)
    if out.isna().any() or (~np.isfinite(out.to_numpy(dtype=float))).any():
        raise RuntimeError("%s: H-L return series contains missing or non-finite values." % model)
    return out.sort_index()


def _debug_series(
    tc_debug_df: pd.DataFrame,
    index: pd.DatetimeIndex,
    column: str,
    *,
    required: bool,
) -> pd.Series:
    if tc_debug_df is None or len(tc_debug_df) == 0:
        if required:
            raise RuntimeError("Transaction-cost diagnostics are missing.")
        return pd.Series(np.nan, index=index, name=column, dtype=float)
    if "date" not in tc_debug_df.columns:
        raise KeyError("Transaction-cost diagnostics are missing the 'date' column.")
    if column not in tc_debug_df.columns:
        if required:
            raise KeyError("Transaction-cost diagnostics are missing '%s'." % column)
        return pd.Series(np.nan, index=index, name=column, dtype=float)

    d = tc_debug_df[["date", column]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d[column] = pd.to_numeric(d[column], errors="coerce")
    if d["date"].isna().any():
        raise RuntimeError("Transaction-cost diagnostics contain invalid dates.")
    if d["date"].duplicated().any():
        dup = d.loc[d["date"].duplicated(keep=False), "date"].head(5).astype(str).tolist()
        raise RuntimeError("Transaction-cost diagnostics contain duplicate dates: %s" % dup)

    series = d.set_index("date")[column].reindex(index)
    if required and series.isna().any():
        missing = series.index[series.isna()][:5].astype(str).tolist()
        raise RuntimeError(
            "Transaction-cost diagnostics are incomplete for '%s'; missing dates include %s"
            % (column, missing)
        )
    return series.astype(float).rename(column)


def _validate_signal_map(signals: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if not signals:
        raise ValueError("signals cannot be empty.")
    out = {}
    for name, frame in signals.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Signal '%s' must be a pandas DataFrame." % name)
        if len(frame) == 0:
            raise ValueError("Signal '%s' is empty." % name)
        out[str(name)] = frame
    return out


def _prepare_manager_kwargs(
    base_kwargs: Optional[Mapping[str, Any]],
    *,
    config: DynamicNAVConfig,
) -> Dict[str, Any]:
    kwargs = dict(base_kwargs or {})

    explicit = (
        "signal_df",
        "freq",
        "portfolio_dir",
        "start_year",
        "end_year",
        "eval_start_year",
        "country",
        "delay_list",
        "load_signal",
        "tc_aum_dollars",
        "tc_execution_days",
    )
    for key in explicit:
        kwargs.pop(key, None)

    kwargs["tc_enable"] = True
    kwargs["tc_use_nonlinear"] = True
    kwargs["tc_aum_dollars"] = float(config.initial_nav)
    kwargs["tc_execution_days"] = float(config.execution_days)

    if config.require_tradability_screens:
        kwargs["tradability_screens"] = True
        kwargs["include_price_adv"] = True

    if config.require_forecast_inputs:
        kwargs["tc_use_forecast_inputs"] = True
        has_path = bool(kwargs.get("tc_forecast_path"))
        has_frame = isinstance(kwargs.get("tc_forecast_df"), pd.DataFrame)
        if not (has_path or has_frame):
            raise ValueError(
                "Dynamic NAV requires frozen daily transaction-cost forecasts. "
                "Provide tc_forecast_path or tc_forecast_df in portfolio_kwargs."
            )

    return kwargs


# -----------------------------------------------------------------------------
# Dynamic AUM controller and strategy proxy
# -----------------------------------------------------------------------------


class _DynamicNAVController:
    """Maintain pre-trade NAV and defer each NAV update until the next rebalance."""

    def __init__(self, initial_nav: float) -> None:
        self.initial_nav = float(initial_nav)
        self.current_nav = float(initial_nav)
        self.pending_nav = None  # type: Optional[float]
        self.records = []  # type: list

    def begin_rebalance(self, *, ctx: Any, date: pd.Timestamp) -> float:
        """Apply the previous period's completed NAV before the next rebalance begins."""
        if self.pending_nav is not None:
            self.current_nav = float(self.pending_nav)
            self.pending_nav = None

        nav = float(self.current_nav)
        if (not np.isfinite(nav)) or nav <= 0.0:
            raise RuntimeError("Dynamic NAV became non-positive or non-finite before rebalance.")

        ctx.tc_aum_dollars = nav
        return nav

    def finish_rebalance(
        self,
        *,
        date: pd.Timestamp,
        row: Sequence[float],
        cut: int,
    ) -> None:
        """Observe this date's net H-L return and schedule next period's NAV."""
        values = np.asarray(row, dtype=float)
        if values.ndim != 1 or len(values) < int(cut):
            raise RuntimeError("Strategy returned an invalid decile-return row for dynamic NAV.")

        net_return = float(values[int(cut) - 1] - values[0])
        if not np.isfinite(net_return):
            raise RuntimeError("Dynamic NAV received a non-finite net H-L return.")

        growth = 1.0 + net_return
        if (not np.isfinite(growth)) or growth <= 0.0:
            raise RuntimeError(
                "Dynamic NAV would become non-positive: date=%s net_return=%.6f"
                % (pd.Timestamp(date).date(), net_return)
            )

        nav_start = float(self.current_nav)
        nav_end = nav_start * growth
        self.records.append(
            {
                "date": pd.Timestamp(date).normalize(),
                "nav_start": nav_start,
                "net_return": net_return,
                "nav_end": nav_end,
            }
        )
        self.pending_nav = nav_end

    @property
    def final_nav(self) -> float:
        if self.pending_nav is not None:
            return float(self.pending_nav)
        return float(self.current_nav)

    def accounting_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["date", "nav_start", "net_return", "nav_end"])
        out = pd.DataFrame(self.records).copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        if out["date"].isna().any() or out["date"].duplicated().any():
            raise RuntimeError("Dynamic NAV controller produced invalid or duplicate rebalance dates.")
        return out.sort_values("date").reset_index(drop=True)


class _DynamicAUMStrategyProxy:
    """Set current NAV immediately before delegating one scheduled rebalance."""

    def __init__(self, delegate: Any, ctx: Any, controller: _DynamicNAVController) -> None:
        self._delegate = delegate
        self._ctx = ctx
        self._controller = controller

    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        self._controller.begin_rebalance(ctx=self._ctx, date=pd.Timestamp(date))
        row, to_df, diag = self._delegate.compute_for_date(df, date, cut, ret_name, prev_state)
        self._controller.finish_rebalance(date=pd.Timestamp(date), row=row, cut=int(cut))

        # Intentionally leave ctx.tc_aum_dollars at this date's BEGINNING NAV.
        # PortfolioManager appends diagnostics after this method returns.  The
        # pending NAV is applied only at the beginning of the next rebalance.
        return row, to_df, diag

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


# -----------------------------------------------------------------------------
# Reporting and persistence
# -----------------------------------------------------------------------------


def _max_drawdown_from_nav(nav_start: float, nav_end: pd.Series) -> float:
    values = np.concatenate(
        [np.asarray([float(nav_start)], dtype=float), pd.to_numeric(nav_end, errors="coerce").to_numpy(dtype=float)]
    )
    if len(values) == 0 or (not np.isfinite(values).all()) or np.any(values <= 0.0):
        return float("nan")
    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1.0
    return float(np.min(drawdowns))


def _build_summary(
    *,
    model: str,
    config: DynamicNAVConfig,
    accounting: pd.DataFrame,
    avg_turnover: float,
    tc_summary: Mapping[str, Any],
    pf_summary: Mapping[str, Any],
    run_signature: str,
) -> Dict[str, Any]:
    eval_df = accounting.loc[accounting["is_evaluation"]].copy()
    if len(eval_df) == 0:
        raise RuntimeError("Dynamic NAV produced no evaluation periods.")

    net_stats = annualized_stats(eval_df["net_return"], config.freq)
    gross_stats = annualized_stats(eval_df["executed_gross_return"], config.freq)
    ppy = float(periods_per_year(config.freq))

    eval_start_nav = float(eval_df.iloc[0]["nav_start"])
    eval_final_nav = float(eval_df.iloc[-1]["nav_end"])
    eval_total_return = eval_final_nav / eval_start_nav - 1.0
    years = float(len(eval_df)) / ppy
    if years > 0.0 and eval_start_nav > 0.0 and eval_final_nav > 0.0:
        eval_cagr = (eval_final_nav / eval_start_nav) ** (1.0 / years) - 1.0
    else:
        eval_cagr = float("nan")

    simulation_final_nav = float(accounting.iloc[-1]["nav_end"])
    simulation_total_return = simulation_final_nav / float(config.initial_nav) - 1.0

    # PortfolioManager's returned avg_turnover follows the project's reporting
    # convention: it averages scheduled rebalance transitions after the initial
    # funding rebalance.  Portfolio diagnostics, by contrast, also record the
    # first all-cash -> invested transition.  Both are canonical for their own
    # purpose, so they are not expected to be numerically identical.
    reported_turnover = float(avg_turnover)
    if not np.isfinite(reported_turnover):
        raise RuntimeError("Dynamic NAV received a non-finite PortfolioManager turnover value.")

    diagnostic_turnover_including_initial = _float_or_nan(
        pf_summary.get(
            "canonical_turnover_oneway_mean",
            pf_summary.get("total_turnover_mean"),
        )
    )

    common_target_sum = _float_or_nan(tc_summary.get("common_target_sum_mean"))
    mean_unfilled = _float_or_nan(tc_summary.get("mean_unfilled_target_share"))
    if not np.isfinite(mean_unfilled):
        raise RuntimeError(
            "Dynamic NAV requires explicit realized mean_unfilled_target_share diagnostics; "
            "refusing to reconstruct it from older summary fields."
        )

    return {
        "run_signature": str(run_signature),
        "model": str(model),
        "initial_nav": float(config.initial_nav),
        "simulation_final_nav": simulation_final_nav,
        "simulation_total_return": simulation_total_return,
        "eval_start_nav": eval_start_nav,
        "eval_final_nav": eval_final_nav,
        "eval_total_return": eval_total_return,
        "eval_cagr": eval_cagr,
        "nav_multiple": eval_final_nav / eval_start_nav,
        "min_nav": float(pd.to_numeric(accounting["nav_end"], errors="coerce").min()),
        "max_nav": float(pd.to_numeric(accounting["nav_end"], errors="coerce").max()),
        "max_drawdown": _max_drawdown_from_nav(eval_start_nav, eval_df["nav_end"]),
        "execution_days": float(config.execution_days),
        "weight_type": str(config.weight_type),
        "freq": str(config.freq),
        "cut": int(config.cut),
        "delay": int(config.delay),
        "start_year": int(config.start_year),
        "end_year": int(config.end_year),
        "eval_start_year": int(config.eval_start_year),
        "country": str(config.country),
        "n_simulation_periods": int(len(accounting)),
        "n_evaluation_periods": int(len(eval_df)),
        "executed_gross_ann_mean": gross_stats["ann_mean"],
        "executed_gross_ann_vol": gross_stats["ann_vol"],
        "executed_gross_sharpe": gross_stats["sharpe"],
        "net_ann_mean": net_stats["ann_mean"],
        "net_ann_vol": net_stats["ann_vol"],
        "net_sharpe": net_stats["sharpe"],
        "annualized_cost_drag": float(eval_df["total_cost"].mean() * ppy),
        "linear_cost_bps_per_rebalance": float(eval_df["linear_cost"].mean() * 1e4),
        "impact_cost_bps_per_rebalance": float(eval_df["impact_cost"].mean() * 1e4),
        "total_cost_bps_per_rebalance": float(eval_df["total_cost"].mean() * 1e4),
        "cumulative_cost_dollars": float(eval_df["cost_dollars"].sum()),
        "avg_turnover_oneway": reported_turnover,
        "diagnostic_turnover_oneway_mean_including_initial": diagnostic_turnover_including_initial,
        "position_cap_bind_date_pct": _float_or_nan(pf_summary.get("any_pos_cap_bind_pct")),
        "trade_cap_bind_date_pct": _float_or_nan(pf_summary.get("any_trade_cap_bind_pct")),
        "position_cap_bind_weight_share_mean": _float_or_nan(pf_summary.get("pos_cap_bind_weight_share_mean")),
        "trade_cap_bind_weight_share_mean": _float_or_nan(pf_summary.get("trade_cap_bind_weight_share_mean")),
        "common_target_sum_mean": common_target_sum,
        "capacity_target_shortfall_mean": _float_or_nan(tc_summary.get("capacity_target_shortfall_mean")),
        "mean_unfilled_target_share": mean_unfilled,
        "low_unfilled_target_share_mean": _float_or_nan(tc_summary.get("low_unfilled_target_share_mean")),
        "high_unfilled_target_share_mean": _float_or_nan(tc_summary.get("high_unfilled_target_share_mean")),
        "low_posttrade_cash_weight_mean": _float_or_nan(tc_summary.get("low_posttrade_cash_weight_mean")),
        "high_posttrade_cash_weight_mean": _float_or_nan(tc_summary.get("high_posttrade_cash_weight_mean")),
        "low_realized_gross_mean": _float_or_nan(tc_summary.get("low_realized_gross_mean")),
        "high_realized_gross_mean": _float_or_nan(tc_summary.get("high_realized_gross_mean")),
        "low_tradable_share_mean": _float_or_nan(tc_summary.get("low_tradable_share_mean")),
        "high_tradable_share_mean": _float_or_nan(tc_summary.get("high_tradable_share_mean")),
        "frozen_nontradable_name_instances": int(tc_summary.get("frozen_nontradable_name_instances", 0) or 0),
        "covariance_excluded_name_instances": int(tc_summary.get("covariance_excluded_name_instances", 0) or 0),
        "insufficient_covariance_no_trade_events": int(tc_summary.get("insufficient_covariance_no_trade_events", 0) or 0),
        "zero_alpha_no_trade_events": int(tc_summary.get("zero_alpha_no_trade_events", 0) or 0),
        "optimizer_solve_calls": int(tc_summary.get("optimizer_solve_calls", 0) or 0),
        "optimizer_fallback_used_events": int(tc_summary.get("optimizer_fallback_used_events", 0) or 0),
        "optimizer_nonconverged_events": int(tc_summary.get("optimizer_nonconverged_events", 0) or 0),
    }


def _run_output_dir(root: Path, model: str) -> Path:
    return root / "runs" / _safe_name(model)


def _save_one_run(run: DynamicNAVRunResult) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.nav_accounting.to_csv(run.output_dir / "dynamic_nav_weekly_accounting.csv", index=False)
    if run.portfolio_returns is not None:
        run.portfolio_returns.to_csv(run.output_dir / "portfolio_returns.csv")
    if run.tc_debug_df is not None:
        run.tc_debug_df.to_csv(run.output_dir / "tc_diagnostics.csv", index=False)
    if run.portfolio_debug_df is not None:
        run.portfolio_debug_df.to_csv(run.output_dir / "portfolio_diagnostics.csv", index=False)
    (run.output_dir / "summary.json").write_text(
        json.dumps(run.summary, indent=2, default=str),
        encoding="utf-8",
    )


def _load_saved_run(
    model: str,
    run_dir: Path,
    *,
    config: DynamicNAVConfig,
    expected_signature: str,
) -> Optional[DynamicNAVRunResult]:
    summary_path = run_dir / "summary.json"
    accounting_path = run_dir / "dynamic_nav_weekly_accounting.csv"
    if not (summary_path.exists() and accounting_path.exists()):
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary.get("model")) != str(model):
            return None
        if str(summary.get("run_signature", "")) != str(expected_signature):
            return None
        expected = {
            "initial_nav": float(config.initial_nav),
            "execution_days": float(config.execution_days),
            "weight_type": str(config.weight_type),
            "freq": str(config.freq),
            "cut": int(config.cut),
            "delay": int(config.delay),
            "start_year": int(config.start_year),
            "end_year": int(config.end_year),
            "eval_start_year": int(config.eval_start_year),
            "country": str(config.country),
        }
        for key, expected_value in expected.items():
            saved_value = summary.get(key)
            if isinstance(expected_value, float):
                try:
                    if not np.isclose(float(saved_value), expected_value):
                        return None
                except (TypeError, ValueError):
                    return None
            elif saved_value != expected_value:
                return None
        accounting = pd.read_csv(accounting_path)
        required = {
            "date",
            "nav_start",
            "executed_gross_return",
            "total_cost",
            "net_return",
            "nav_end",
            "is_evaluation",
        }
        if not required.issubset(accounting.columns):
            return None
        accounting["date"] = pd.to_datetime(accounting["date"], errors="coerce").dt.normalize()
        if accounting["date"].isna().any():
            return None
        for col in ("nav_start", "executed_gross_return", "total_cost", "net_return", "nav_end"):
            accounting[col] = pd.to_numeric(accounting[col], errors="coerce")
            if accounting[col].isna().any():
                return None
        accounting["is_evaluation"] = accounting["is_evaluation"].astype(bool)
        eval_df = accounting.loc[accounting["is_evaluation"]].copy()
        if len(eval_df) == 0:
            return None
        idx = pd.DatetimeIndex(eval_df["date"])
        net = pd.Series(eval_df["net_return"].to_numpy(dtype=float), index=idx, name="net_return")
        gross = pd.Series(eval_df["executed_gross_return"].to_numpy(dtype=float), index=idx, name="executed_gross_return")
        cost = pd.Series(eval_df["total_cost"].to_numpy(dtype=float), index=idx, name="total_cost")
        return DynamicNAVRunResult(
            model=str(model),
            initial_nav=float(summary["initial_nav"]),
            final_nav=float(summary["simulation_final_nav"]),
            nav_accounting=accounting,
            net_returns=net,
            executed_gross_returns=gross,
            total_cost=cost,
            summary=summary,
            output_dir=run_dir,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


# -----------------------------------------------------------------------------
# One model and multi-model runners
# -----------------------------------------------------------------------------


def run_one_dynamic_nav(
    signal_df: pd.DataFrame,
    *,
    model: str,
    config: DynamicNAVConfig,
    portfolio_kwargs: Optional[Mapping[str, Any]] = None,
    manager_cls: Optional[Type[Any]] = None,
) -> DynamicNAVRunResult:
    """Run one signal through the active cost-aware portfolio with evolving NAV."""
    root = Path(config.output_dir).expanduser().resolve()
    run_dir = _run_output_dir(root, model)
    kwargs = _prepare_manager_kwargs(portfolio_kwargs, config=config)
    signature = build_run_signature(
        run_kind="dynamic_nav",
        model=str(model),
        signal_df=signal_df,
        config_fields={
            "initial_nav": float(config.initial_nav),
            "weight_type": str(config.weight_type),
            "freq": str(config.freq),
            "cut": int(config.cut),
            "delay": int(config.delay),
            "start_year": int(config.start_year),
            "end_year": int(config.end_year),
            "eval_start_year": int(config.eval_start_year),
            "country": str(config.country),
            "execution_days": float(config.execution_days),
            "require_tradability_screens": bool(config.require_tradability_screens),
            "require_forecast_inputs": bool(config.require_forecast_inputs),
        },
        portfolio_kwargs=kwargs,
        manager_identity=manager_identity(manager_cls),
    )

    if config.resume:
        saved = _load_saved_run(
            model, run_dir, config=config, expected_signature=signature
        )
        if saved is not None:
            if config.verbose:
                print("[dynamic-nav] reuse %s from $%.1fm" % (model, float(config.initial_nav) / 1e6))
            return saved

    if manager_cls is None:
        raise TypeError(
            "manager_cls is required in the standalone public package. "
            "Pass a PortfolioManager-compatible class from your research integration layer."
        )

    if config.verbose:
        print("[dynamic-nav] run   %s from $%.1fm" % (model, float(config.initial_nav) / 1e6))

    pm = manager_cls(
        signal_df=signal_df.copy(),
        freq=config.freq,
        portfolio_dir=str(run_dir),
        start_year=int(config.start_year),
        end_year=int(config.end_year),
        eval_start_year=int(config.eval_start_year),
        country=config.country,
        delay_list=[int(config.delay)],
        load_signal=True,
        **kwargs,
    )

    if not hasattr(pm, "_get_strategy") or not callable(getattr(pm, "_get_strategy")):
        raise TypeError("Dynamic NAV requires a PortfolioManager-compatible _get_strategy method.")

    controller = _DynamicNAVController(float(config.initial_nav))
    original_get_strategy = pm._get_strategy

    def _dynamic_get_strategy(weight_type):
        delegate = original_get_strategy(weight_type)
        return _DynamicAUMStrategyProxy(delegate, pm, controller)

    pm._get_strategy = _dynamic_get_strategy
    try:
        pf_ret, avg_turnover = pm.calculate_portfolio_rets(
            weight_type=config.weight_type,
            cut=int(config.cut),
            delay=int(config.delay),
        )
    finally:
        pm._get_strategy = original_get_strategy

    controller_df = controller.accounting_frame()
    if len(controller_df) == 0:
        raise RuntimeError("Dynamic NAV controller recorded no rebalances.")

    returned_net = _extract_hl_series(pf_ret, model)
    controller_net = pd.Series(
        controller_df["net_return"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(controller_df["date"]),
        name="net_return",
    )
    overlap = returned_net.index.intersection(controller_net.index)
    if len(overlap) != len(returned_net.index):
        raise RuntimeError("Dynamic NAV accounting is missing evaluation dates returned by PortfolioManager.")
    if not np.allclose(
        returned_net.reindex(overlap).to_numpy(dtype=float),
        controller_net.reindex(overlap).to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise RuntimeError("Dynamic NAV controller return path does not match PortfolioManager H-L returns.")

    tc_debug_df = getattr(pm, "_last_tc_debug_df", None)
    pf_debug_df = getattr(pm, "_last_portfolio_debug_df", None)
    tc_summary = dict(getattr(pm, "_last_tc_debug_summary", None) or {})
    pf_summary = dict(getattr(pm, "_last_portfolio_debug_summary", None) or {})

    all_dates = pd.DatetimeIndex(controller_df["date"])

    # ``tc_debug_df`` is the strategy-level execution diagnostic table and is the
    # authoritative source for the total cost actually deducted from H-L returns.
    # The canonical signed-trade diagnostics in ``pf_debug_df`` carry the linear /
    # impact decomposition.  Prefer that decomposition when available, but retain
    # compatibility with older diagnostic tables that stored the components in the
    # strategy-level table instead.
    total_cost = _debug_series(tc_debug_df, all_dates, "total_cost", required=True)

    linear_source = (
        pf_debug_df
        if isinstance(pf_debug_df, pd.DataFrame) and "linear_cost" in pf_debug_df.columns
        else tc_debug_df
    )
    impact_source = (
        pf_debug_df
        if isinstance(pf_debug_df, pd.DataFrame) and "impact_cost" in pf_debug_df.columns
        else tc_debug_df
    )

    # Component costs are reporting diagnostics rather than NAV inputs.  They may
    # be unavailable for warm-up dates because portfolio diagnostics are filtered
    # to the evaluation sample, so missing component observations are retained as
    # NaN rather than causing an otherwise valid dynamic-NAV simulation to fail.
    linear_cost = _debug_series(linear_source, all_dates, "linear_cost", required=False)
    impact_cost = _debug_series(impact_source, all_dates, "impact_cost", required=False)

    accounting = controller_df.copy()
    accounting["linear_cost"] = linear_cost.to_numpy(dtype=float)
    accounting["impact_cost"] = impact_cost.to_numpy(dtype=float)
    accounting["total_cost"] = total_cost.to_numpy(dtype=float)

    # Carry the implementation state into the weekly NAV ledger.  These are
    # reporting fields only; NAV compounding still depends solely on net_return.
    for diag_col in (
        "common_target_sum",
        "capacity_target_shortfall",
        "low_posttrade_cash_weight",
        "high_posttrade_cash_weight",
        "low_unfilled_target_share",
        "high_unfilled_target_share",
        "mean_unfilled_target_share",
    ):
        accounting[diag_col] = _debug_series(
            tc_debug_df,
            all_dates,
            diag_col,
            required=False,
        ).to_numpy(dtype=float)

    accounting["executed_gross_return"] = accounting["net_return"] + accounting["total_cost"]
    accounting["gross_pnl_dollars"] = accounting["nav_start"] * accounting["executed_gross_return"]
    accounting["cost_dollars"] = accounting["nav_start"] * accounting["total_cost"]
    accounting["net_pnl_dollars"] = accounting["nav_start"] * accounting["net_return"]
    accounting["is_evaluation"] = accounting["date"].dt.year >= int(config.eval_start_year)

    expected_nav_end = accounting["nav_start"] + accounting["net_pnl_dollars"]
    if not np.allclose(
        expected_nav_end.to_numpy(dtype=float),
        accounting["nav_end"].to_numpy(dtype=float),
        atol=1e-6,
        rtol=1e-12,
    ):
        raise RuntimeError("Dynamic NAV violates NAV_t+1 = NAV_t * (1 + net_return_t).")

    pnl_identity = accounting["gross_pnl_dollars"] - accounting["cost_dollars"]
    if not np.allclose(
        pnl_identity.to_numpy(dtype=float),
        accounting["net_pnl_dollars"].to_numpy(dtype=float),
        atol=1e-6,
        rtol=1e-12,
    ):
        raise RuntimeError("Dynamic NAV violates gross P&L - trading cost = net P&L.")

    eval_df = accounting.loc[accounting["is_evaluation"]].copy()
    if len(eval_df) == 0:
        raise RuntimeError("No dynamic-NAV evaluation dates remain after eval_start_year.")
    eval_idx = pd.DatetimeIndex(eval_df["date"])
    eval_net = pd.Series(eval_df["net_return"].to_numpy(dtype=float), index=eval_idx, name="net_return")
    eval_gross = pd.Series(
        eval_df["executed_gross_return"].to_numpy(dtype=float),
        index=eval_idx,
        name="executed_gross_return",
    )
    eval_cost = pd.Series(eval_df["total_cost"].to_numpy(dtype=float), index=eval_idx, name="total_cost")

    summary = _build_summary(
        model=model,
        config=config,
        accounting=accounting,
        avg_turnover=float(avg_turnover),
        tc_summary=tc_summary,
        pf_summary=pf_summary,
        run_signature=signature,
    )

    result = DynamicNAVRunResult(
        model=str(model),
        initial_nav=float(config.initial_nav),
        final_nav=float(controller.final_nav),
        nav_accounting=accounting,
        net_returns=eval_net,
        executed_gross_returns=eval_gross,
        total_cost=eval_cost,
        summary=summary,
        output_dir=run_dir,
        manager=pm,
        portfolio_returns=pf_ret.copy(),
        tc_debug_df=tc_debug_df.copy() if isinstance(tc_debug_df, pd.DataFrame) else None,
        portfolio_debug_df=pf_debug_df.copy() if isinstance(pf_debug_df, pd.DataFrame) else None,
    )

    if config.save_run_details:
        _save_one_run(result)
    return result


def run_dynamic_nav(
    signals: Mapping[str, pd.DataFrame],
    *,
    config: DynamicNAVConfig,
    portfolio_kwargs: Optional[Mapping[str, Any]] = None,
    manager_cls: Optional[Type[Any]] = None,
) -> DynamicNAVResult:
    """Run several signals from the same starting NAV and save comparison tables."""
    signal_map = _validate_signal_map(signals)
    root = Path(config.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    runs = {}  # type: Dict[str, DynamicNAVRunResult]
    rows = []
    nav_paths = []

    for model, signal_df in signal_map.items():
        run = run_one_dynamic_nav(
            signal_df,
            model=model,
            config=config,
            portfolio_kwargs=portfolio_kwargs,
            manager_cls=manager_cls,
        )
        runs[str(model)] = run
        rows.append(dict(run.summary))

        path = run.nav_accounting[["date", "nav_end"]].copy()
        path = path.rename(columns={"nav_end": str(model)})
        nav_paths.append(path)

    summary = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    summary.to_csv(root / "dynamic_nav_summary.csv", index=False)

    if nav_paths:
        wide = nav_paths[0]
        for part in nav_paths[1:]:
            wide = wide.merge(part, on="date", how="outer", validate="one_to_one")
        wide = wide.sort_values("date")
        wide.to_csv(root / "dynamic_nav_paths.csv", index=False)

    manifest = {
        "initial_nav": float(config.initial_nav),
        "weight_type": str(config.weight_type),
        "freq": str(config.freq),
        "cut": int(config.cut),
        "delay": int(config.delay),
        "start_year": int(config.start_year),
        "end_year": int(config.end_year),
        "eval_start_year": int(config.eval_start_year),
        "country": str(config.country),
        "execution_days": float(config.execution_days),
        "require_tradability_screens": bool(config.require_tradability_screens),
        "require_forecast_inputs": bool(config.require_forecast_inputs),
        "models": list(signal_map.keys()),
        "accounting_identity": "NAV_{t+1}=NAV_t*(1+R_net,t)",
        "execution_convention": "scheduled portfolio decision at portfolio frequency; one-day execution by default",
    }
    (root / "dynamic_nav_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    return DynamicNAVResult(summary=summary, runs=runs, output_dir=root)


__all__ = [
    "DynamicNAVConfig",
    "DynamicNAVRunResult",
    "DynamicNAVResult",
    "annualized_stats",
    "periods_per_year",
    "run_one_dynamic_nav",
    "run_dynamic_nav",
]
