"""Fixed-AUM capacity analysis for active transaction-cost-aware portfolios.

This module reruns the same portfolio strategy at a grid of fixed fund sizes while
holding the predictive signals and economic model specification fixed.  It is
intended to answer a single implementation question:

    How does realized portfolio performance deteriorate as AUM increases and
    liquidity constraints become more binding?

The runner deliberately RE-OPTIMIZES at every AUM.  It does not extrapolate a
$100m cost series with a square-root scaling rule.  Changing AUM changes both
market-impact penalties and ADV-based feasibility constraints, so the resulting
portfolio weights, turnover, costs, and returns are allowed to change.

Canonical execution convention
------------------------------
The default specification is:

* portfolio decisions are made once per week;
* each scheduled rebalance is executed over one trading day;
* ``tc_adv_daily`` is daily dollar ADV;
* ``tc_sigma_daily`` is daily return volatility;
* the trade cap is participation in the one-day execution volume;
* positions that cannot be fully adjusted within the trade cap remain in the
  stateful portfolio and are carried into the next weekly rebalance.

For an active transaction-cost strategy, ``PortfolioManager`` returns the H-L
return AFTER realized trading costs have been deducted.  The corresponding
executed pre-cost return is reconstructed only for reporting as:

    executed_gross_return_t = net_return_t + realized_cost_t

This is not the gross return of a separate zero-cost optimizer.  It is the gross
return of the weights actually selected by the cost-aware optimizer at that AUM.

The runner fails closed on missing transaction-cost forecasts or incomplete
per-date cost diagnostics.  Historical ADV / weekly-volatility fallbacks are not
used by this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type
import json
import math
import re

import numpy as np
import pandas as pd

from cost_aware_portfolio.fingerprint import build_run_signature, manager_identity


DEFAULT_AUM_GRID = (
    10_000_000.0,
    25_000_000.0,
    50_000_000.0,
    100_000_000.0,
    250_000_000.0,
    500_000_000.0,
    1_000_000_000.0,
)


@dataclass(frozen=True)
class FixedAUMCapacityConfig:
    """Configuration for a fixed-AUM re-optimization grid."""

    output_dir: str
    aum_grid: Sequence[float] = DEFAULT_AUM_GRID
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
        if (not np.isfinite(float(self.execution_days))) or float(self.execution_days) <= 0.0:
            raise ValueError("execution_days must be finite and positive.")

        cleaned = []
        for value in self.aum_grid:
            aum = float(value)
            if (not np.isfinite(aum)) or aum <= 0.0:
                raise ValueError("Every AUM value must be finite and strictly positive.")
            cleaned.append(aum)
        if not cleaned:
            raise ValueError("aum_grid cannot be empty.")
        object.__setattr__(self, "aum_grid", tuple(sorted(set(cleaned))))


@dataclass
class CapacityRunResult:
    """One model/AUM run and its aligned weekly accounting series."""

    model: str
    aum_dollars: float
    net_returns: pd.Series
    executed_gross_returns: pd.Series
    total_cost: pd.Series
    summary: Dict[str, Any]
    output_dir: Path
    manager: Optional[Any] = None
    portfolio_returns: Optional[pd.DataFrame] = None
    tc_debug_df: Optional[pd.DataFrame] = None
    portfolio_debug_df: Optional[pd.DataFrame] = None
    missing_return_df: Optional[pd.DataFrame] = None


@dataclass
class FixedAUMCapacityResult:
    """Complete capacity grid output."""

    summary: pd.DataFrame
    runs: Dict[Tuple[str, float], CapacityRunResult]
    output_dir: Path

    def pivot(self, metric: str) -> pd.DataFrame:
        """Return model x AUM table for one summary metric."""
        if metric not in self.summary.columns:
            raise KeyError("Unknown capacity metric: %s" % metric)
        return self.summary.pivot(index="model", columns="aum_millions", values=metric)


# -----------------------------------------------------------------------------
# Validation and accounting helpers
# -----------------------------------------------------------------------------


def periods_per_year(freq: str) -> int:
    """Return the annualization factor used by the portfolio project."""
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


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


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
    aum: float,
    config: FixedAUMCapacityConfig,
) -> Dict[str, Any]:
    kwargs = dict(base_kwargs or {})

    # These are controlled centrally by the capacity runner.
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

    # Canonical active transaction-cost path.
    kwargs["tc_enable"] = True
    kwargs["tc_use_nonlinear"] = True
    kwargs["tc_aum_dollars"] = float(aum)
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
                "Fixed-AUM capacity requires frozen daily transaction-cost forecasts. "
                "Provide tc_forecast_path or tc_forecast_df in portfolio_kwargs."
            )

    return kwargs


# -----------------------------------------------------------------------------
# One run
# -----------------------------------------------------------------------------


def _build_run_summary(
    *,
    model: str,
    aum: float,
    config: FixedAUMCapacityConfig,
    net: pd.Series,
    total_cost: pd.Series,
    linear_cost: pd.Series,
    impact_cost: pd.Series,
    avg_turnover: float,
    tc_summary: Mapping[str, Any],
    pf_summary: Mapping[str, Any],
    missing_return_summary: Optional[Mapping[str, Any]] = None,
    run_signature: str = "",
) -> Dict[str, Any]:
    missing_return_summary = dict(missing_return_summary or {})
    executed_gross = (net + total_cost).rename("executed_gross_return")
    gross_stats = annualized_stats(executed_gross, config.freq)
    net_stats = annualized_stats(net, config.freq)
    ppy = float(periods_per_year(config.freq))
    annualized_cost_drag = float(total_cost.mean() * ppy)

    gross_ann_mean = gross_stats["ann_mean"]
    if np.isfinite(gross_ann_mean) and abs(gross_ann_mean) > 1e-15:
        cost_share = annualized_cost_drag / gross_ann_mean
    else:
        cost_share = float("nan")

    common_target_sum = _float_or_nan(tc_summary.get("common_target_sum_mean"))
    mean_unfilled = _float_or_nan(tc_summary.get("mean_unfilled_target_share"))
    if not np.isfinite(mean_unfilled) and np.isfinite(common_target_sum):
        # Backward-compatible fallback for diagnostics generated before explicit
        # realized cash/unfilled reporting existed.
        mean_unfilled = max(0.0, 1.0 - common_target_sum)

    return {
        "model": str(model),
        "run_signature": str(run_signature),
        "aum_dollars": float(aum),
        "aum_millions": float(aum) / 1e6,
        "execution_days": float(config.execution_days),
        "weight_type": str(config.weight_type),
        "n_periods": int(len(net)),
        "executed_gross_ann_mean": gross_stats["ann_mean"],
        "executed_gross_ann_vol": gross_stats["ann_vol"],
        "executed_gross_sharpe": gross_stats["sharpe"],
        "net_ann_mean": net_stats["ann_mean"],
        "net_ann_vol": net_stats["ann_vol"],
        "net_sharpe": net_stats["sharpe"],
        "annualized_cost_drag": annualized_cost_drag,
        "cost_share_of_executed_gross_mean": cost_share,
        "avg_turnover_oneway": float(avg_turnover),
        "linear_cost_bps_per_rebalance": float(linear_cost.mean() * 1e4),
        "impact_cost_bps_per_rebalance": float(impact_cost.mean() * 1e4),
        "total_cost_bps_per_rebalance": float(total_cost.mean() * 1e4),
        "total_cost_bps_median": _float_or_nan(tc_summary.get("total_cost_bps_median")),
        "total_cost_bps_p90": _float_or_nan(tc_summary.get("total_cost_bps_p90")),
        "total_cost_bps_p95": _float_or_nan(tc_summary.get("total_cost_bps_p95")),
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
        "missing_return_event_count": int(missing_return_summary.get("event_count", 0) or 0),
        "missing_return_unique_stock_count": int(missing_return_summary.get("unique_stock_count", 0) or 0),
        "missing_return_terminal_event_count": int(missing_return_summary.get("terminal_event_count", 0) or 0),
        "missing_return_materiality_warning_count": int(
            missing_return_summary.get("materiality_warning_count", 0) or 0
        ),
        "missing_return_max_abs_name_weight": _float_or_nan(
            missing_return_summary.get("max_abs_name_weight", 0.0)
        ),
        "missing_return_mean_abs_name_weight": _float_or_nan(
            missing_return_summary.get("mean_abs_name_weight", 0.0)
        ),
        "missing_return_max_abs_sleeve_weight": _float_or_nan(
            missing_return_summary.get("max_missing_abs_sleeve_weight", 0.0)
        ),
        "missing_return_mean_abs_sleeve_weight": _float_or_nan(
            missing_return_summary.get("mean_missing_abs_sleeve_weight", 0.0)
        ),
        "missing_return_max_consecutive_periods": int(
            missing_return_summary.get("max_consecutive_missing_periods", 0) or 0
        ),
    }


def _run_output_dir(root: Path, model: str, aum: float) -> Path:
    return root / "runs" / _safe_name(model) / ("aum_%d" % int(round(float(aum))))


def _save_one_run(run: CapacityRunResult) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    accounting = pd.DataFrame(
        {
            "date": run.net_returns.index,
            "executed_gross_return": run.executed_gross_returns.to_numpy(dtype=float),
            "total_cost": run.total_cost.to_numpy(dtype=float),
            "net_return": run.net_returns.to_numpy(dtype=float),
        }
    )
    accounting.to_csv(run.output_dir / "weekly_accounting.csv", index=False)
    if run.portfolio_returns is not None:
        run.portfolio_returns.to_csv(run.output_dir / "portfolio_returns.csv")
    if run.tc_debug_df is not None:
        run.tc_debug_df.to_csv(run.output_dir / "tc_diagnostics.csv", index=False)
    if run.portfolio_debug_df is not None:
        run.portfolio_debug_df.to_csv(run.output_dir / "portfolio_diagnostics.csv", index=False)
    if run.missing_return_df is not None and len(run.missing_return_df) > 0:
        run.missing_return_df.to_csv(run.output_dir / "missing_return_events.csv", index=False)
    (run.output_dir / "summary.json").write_text(
        json.dumps(run.summary, indent=2, default=str),
        encoding="utf-8",
    )


def _load_saved_run(
    model: str,
    aum: float,
    run_dir: Path,
    *,
    expected_signature: str,
) -> Optional[CapacityRunResult]:
    summary_path = run_dir / "summary.json"
    accounting_path = run_dir / "weekly_accounting.csv"
    if not (summary_path.exists() and accounting_path.exists()):
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary.get("model")) != str(model):
            return None
        if str(summary.get("run_signature", "")) != str(expected_signature):
            return None
        if not np.isclose(float(summary.get("aum_dollars")), float(aum)):
            return None
        accounting = pd.read_csv(accounting_path)
        required = {"date", "executed_gross_return", "total_cost", "net_return"}
        if not required.issubset(accounting.columns):
            return None
        dates = pd.to_datetime(accounting["date"], errors="coerce").dt.normalize()
        if dates.isna().any():
            return None
        net = pd.Series(pd.to_numeric(accounting["net_return"], errors="coerce").to_numpy(), index=dates, name="net_return")
        gross = pd.Series(pd.to_numeric(accounting["executed_gross_return"], errors="coerce").to_numpy(), index=dates, name="executed_gross_return")
        cost = pd.Series(pd.to_numeric(accounting["total_cost"], errors="coerce").to_numpy(), index=dates, name="total_cost")
        if net.isna().any() or gross.isna().any() or cost.isna().any():
            return None

        # Older checkpoints can have NaN linear/impact summary fields because the
        # runner previously looked only in strategy-level TC diagnostics.  The
        # saved canonical portfolio diagnostics already contain the decomposition,
        # so repair reporting metadata in-place without rerunning the optimizer.
        pf_debug_path = run_dir / "portfolio_diagnostics.csv"
        if pf_debug_path.exists():
            try:
                pf_debug = pd.read_csv(pf_debug_path)
                for col, summary_key in (
                    ("linear_cost", "linear_cost_bps_per_rebalance"),
                    ("impact_cost", "impact_cost_bps_per_rebalance"),
                ):
                    current = pd.to_numeric(
                        pd.Series([summary.get(summary_key)]), errors="coerce"
                    ).iloc[0]
                    if (not np.isfinite(current)) and col in pf_debug.columns:
                        values = pd.to_numeric(pf_debug[col], errors="coerce")
                        if np.isfinite(values.to_numpy(dtype=float)).any():
                            summary[summary_key] = float(values.mean() * 1e4)
            except (OSError, ValueError, TypeError, KeyError):
                pass

        return CapacityRunResult(
            model=str(model),
            aum_dollars=float(aum),
            net_returns=net,
            executed_gross_returns=gross,
            total_cost=cost,
            summary=summary,
            output_dir=run_dir,
        )
    except Exception:
        return None


def run_one_fixed_aum(
    signal_df: pd.DataFrame,
    *,
    model: str,
    aum: float,
    config: FixedAUMCapacityConfig,
    portfolio_kwargs: Optional[Mapping[str, Any]] = None,
    manager_cls: Optional[Type[Any]] = None,
) -> CapacityRunResult:
    """Run one model at one fixed AUM and return canonical capacity accounting."""
    root = Path(config.output_dir).expanduser().resolve()
    run_dir = _run_output_dir(root, model, aum)
    kwargs = _prepare_manager_kwargs(portfolio_kwargs, aum=float(aum), config=config)
    signature = build_run_signature(
        run_kind="fixed_aum_capacity",
        model=str(model),
        signal_df=signal_df,
        config_fields={
            "aum_dollars": float(aum),
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
            model,
            aum,
            run_dir,
            expected_signature=signature,
        )
        if saved is not None:
            if config.verbose:
                print("[capacity] reuse %s @ $%.0fm" % (model, float(aum) / 1e6))
            return saved

    if manager_cls is None:
        raise TypeError(
            "manager_cls is required in the standalone public package. "
            "Pass a PortfolioManager-compatible class from your research integration layer."
        )

    if config.verbose:
        print("[capacity] run   %s @ $%.0fm" % (model, float(aum) / 1e6))

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

    pf_ret, avg_turnover = pm.calculate_portfolio_rets(
        weight_type=config.weight_type,
        cut=int(config.cut),
        delay=int(config.delay),
    )

    net = _extract_hl_series(pf_ret, model)
    tc_debug_df = getattr(pm, "_last_tc_debug_df", None)
    pf_debug_df = getattr(pm, "_last_portfolio_debug_df", None)
    tc_summary = dict(getattr(pm, "_last_tc_debug_summary", None) or {})
    pf_summary = dict(getattr(pm, "_last_portfolio_debug_summary", None) or {})
    missing_return_df = getattr(pm, "_last_missing_return_df", None)
    missing_return_summary = dict(getattr(pm, "_last_missing_return_summary", None) or {})

    # Strategy diagnostics carry the total cost actually deducted from the H-L
    # return.  Canonical signed-trade portfolio diagnostics carry the linear /
    # impact decomposition.  Prefer the canonical component columns when present,
    # while retaining compatibility with older test/debug frames that stored them
    # directly in ``tc_debug_df``.
    total_cost = _debug_series(tc_debug_df, net.index, "total_cost", required=True)
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
    linear_cost = _debug_series(linear_source, net.index, "linear_cost", required=False)
    impact_cost = _debug_series(impact_source, net.index, "impact_cost", required=False)

    # If detailed components are unavailable, use NaN for their individual means;
    # total_cost remains mandatory and authoritative.
    executed_gross = (net + total_cost).rename("executed_gross_return")
    summary = _build_run_summary(
        model=model,
        aum=float(aum),
        config=config,
        net=net,
        total_cost=total_cost,
        linear_cost=linear_cost,
        impact_cost=impact_cost,
        avg_turnover=float(avg_turnover),
        tc_summary=tc_summary,
        pf_summary=pf_summary,
        missing_return_summary=missing_return_summary,
        run_signature=signature,
    )

    run = CapacityRunResult(
        model=str(model),
        aum_dollars=float(aum),
        net_returns=net,
        executed_gross_returns=executed_gross,
        total_cost=total_cost,
        summary=summary,
        output_dir=run_dir,
        manager=pm,
        portfolio_returns=pf_ret.copy(),
        tc_debug_df=tc_debug_df.copy() if isinstance(tc_debug_df, pd.DataFrame) else None,
        portfolio_debug_df=pf_debug_df.copy() if isinstance(pf_debug_df, pd.DataFrame) else None,
        missing_return_df=(
            missing_return_df.copy() if isinstance(missing_return_df, pd.DataFrame) else None
        ),
    )

    if config.save_run_details or config.resume:
        _save_one_run(run)
    return run


# -----------------------------------------------------------------------------
# Full grid and saved tables
# -----------------------------------------------------------------------------


def _save_grid_outputs(result: FixedAUMCapacityResult, config: FixedAUMCapacityConfig) -> None:
    root = result.output_dir
    root.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(root / "fixed_aum_capacity_summary.csv", index=False)

    metrics = (
        "executed_gross_sharpe",
        "net_sharpe",
        "avg_turnover_oneway",
        "total_cost_bps_per_rebalance",
        "annualized_cost_drag",
        "position_cap_bind_date_pct",
        "trade_cap_bind_date_pct",
        "mean_unfilled_target_share",
    )
    for metric in metrics:
        if metric in result.summary.columns:
            result.pivot(metric).to_csv(root / ("fixed_aum_%s.csv" % metric))

    manifest = {
        "analysis": "fixed_aum_reoptimized_capacity",
        "weight_type": config.weight_type,
        "freq": config.freq,
        "cut": int(config.cut),
        "delay": int(config.delay),
        "start_year": int(config.start_year),
        "end_year": int(config.end_year),
        "eval_start_year": int(config.eval_start_year),
        "country": config.country,
        "execution_days": float(config.execution_days),
        "aum_grid": [float(x) for x in config.aum_grid],
        "models": result.summary["model"].drop_duplicates().tolist(),
        "n_model_aum_runs": int(len(result.summary)),
        "note": (
            "Each AUM point is independently re-optimized. Executed gross return equals the "
            "net H-L return from the active TC strategy plus that rebalance's realized cost."
        ),
    }
    (root / "fixed_aum_capacity_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def run_fixed_aum_capacity(
    signals: Mapping[str, pd.DataFrame],
    *,
    config: FixedAUMCapacityConfig,
    portfolio_kwargs: Optional[Mapping[str, Any]] = None,
    manager_cls: Optional[Type[Any]] = None,
) -> FixedAUMCapacityResult:
    """Re-optimize every signal at every fixed AUM and save checkpointed outputs."""
    signal_map = _validate_signal_map(signals)
    root = Path(config.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    runs = {}  # type: Dict[Tuple[str, float], CapacityRunResult]

    for aum in config.aum_grid:
        if config.verbose:
            print("\n" + "=" * 80)
            print("FIXED-AUM CAPACITY: $%.0fm" % (float(aum) / 1e6))
        for model, signal_df in signal_map.items():
            run = run_one_fixed_aum(
                signal_df,
                model=model,
                aum=float(aum),
                config=config,
                portfolio_kwargs=portfolio_kwargs,
                manager_cls=manager_cls,
            )
            runs[(str(model), float(aum))] = run
            rows.append(dict(run.summary))

        # Checkpoint the long-form table after every completed AUM block.
        checkpoint = pd.DataFrame(rows)
        if len(checkpoint):
            checkpoint.to_csv(root / "fixed_aum_capacity_summary_partial.csv", index=False)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["aum_dollars", "net_sharpe", "model"], ascending=[True, False, True]).reset_index(drop=True)
    result = FixedAUMCapacityResult(summary=summary, runs=runs, output_dir=root)
    _save_grid_outputs(result, config)

    partial = root / "fixed_aum_capacity_summary_partial.csv"
    if partial.exists():
        try:
            partial.unlink()
        except OSError:
            pass

    if config.verbose:
        print("\nSaved fixed-AUM capacity outputs to: %s" % root)
    return result


__all__ = [
    "DEFAULT_AUM_GRID",
    "CapacityRunResult",
    "FixedAUMCapacityConfig",
    "FixedAUMCapacityResult",
    "annualized_stats",
    "periods_per_year",
    "run_fixed_aum_capacity",
    "run_one_fixed_aum",
]
