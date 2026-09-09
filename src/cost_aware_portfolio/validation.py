"""Statistical validation utilities for portfolio and signal research.

The functions in this module are intentionally independent of notebook globals.
They cover the inference used in the original research analysis and the
after-cost tests used by the stateful transaction-cost engine.

Main conventions
----------------
* Weekly project results use ``periods_per_year=52``.
* HAC inference is based on an intercept-only OLS or ordinary linear regression
  with Newey-West / HAC covariance estimates.
* Sharpe inference uses paired circular moving-block bootstrap resampling so
  serial dependence and same-week cross-strategy dependence are preserved.
* Hansen SPA is implemented as a studentized, one-sided superior-predictive-
  ability test with Hansen's consistent recentering rule and circular block
  bootstrap draws.
* Deflated Sharpe Ratio is provided as a supplementary diagnostic.  Its trial
  count should be interpreted as the effective number of strategy searches,
  which can be smaller than the raw number when candidate strategies are highly
  correlated.

No function in this module retrains a predictive model or changes portfolio
construction.  It consumes already-computed return streams and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


DEFAULT_PERIODS_PER_YEAR = 52
DEFAULT_HAC_LAGS = 8
DEFAULT_BOOTSTRAP_REPS = 5000
DEFAULT_BLOCK_LENGTH = 8
DEFAULT_SEED = 1729
_NORMAL_975 = 1.959963984540054
_EULER_GAMMA = 0.5772156649015329


def _numeric_series(x: Any, name: Optional[str] = None) -> pd.Series:
    """Return a finite numeric Series while preserving the original index."""
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if name is not None:
        s.name = name
    return s


def _paired_frame(
    a: Any,
    b: Any,
    *,
    require_identical_index: bool = True,
) -> pd.DataFrame:
    """Return paired finite observations with strict sample alignment by default.

    The final research analysis compares strategies on the same official weekly
    sample.  Silent intersection of mismatched date indexes can change the sample
    across model pairs, so the default fails closed on any index or missing-value
    mismatch.  Set ``require_identical_index=False`` only for an explicitly
    intended common-observation analysis.
    """
    aa = _numeric_series(a, "a")
    bb = _numeric_series(b, "b")
    if require_identical_index:
        if not aa.index.equals(bb.index):
            raise ValueError("Paired return streams must have identical indexes in strict mode.")
        pair = pd.concat([aa, bb], axis=1)
        if pair.isna().any().any():
            raise ValueError("Paired return streams contain missing/non-finite observations in strict mode.")
        return pair
    return pd.concat([aa, bb], axis=1).dropna()


def annualized_sharpe(
    returns: Any,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Compute the arithmetic annualized Sharpe ratio with zero risk-free rate."""
    x = _numeric_series(returns).dropna().to_numpy(dtype=float)
    if x.size < 3:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if (not np.isfinite(sd)) or sd <= 0.0:
        return float("nan")
    return float(np.mean(x) / sd * np.sqrt(float(periods_per_year)))


def performance_stats(
    returns: Any,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """Return period count, annualized arithmetic mean, volatility, and Sharpe."""
    x = _numeric_series(returns).dropna()
    if len(x) == 0:
        return {
            "n_periods": 0,
            "ann_mean": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
        }
    mean_period = float(x.mean())
    vol_period = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    sharpe = (
        mean_period / vol_period * np.sqrt(float(periods_per_year))
        if np.isfinite(vol_period) and vol_period > 0.0
        else float("nan")
    )
    return {
        "n_periods": int(len(x)),
        "ann_mean": float(mean_period * periods_per_year),
        "ann_vol": float(vol_period * np.sqrt(float(periods_per_year))) if np.isfinite(vol_period) else float("nan"),
        "sharpe": float(sharpe),
    }


def hac_mean_test(
    returns: Any,
    maxlags: int = DEFAULT_HAC_LAGS,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """HAC test and confidence interval for the mean of a return series."""
    x = _numeric_series(returns).dropna().to_numpy(dtype=float)
    if x.size < 3:
        raise ValueError("hac_mean_test requires at least three finite observations.")
    if int(maxlags) < 0:
        raise ValueError("maxlags must be nonnegative.")
    X = np.ones((len(x), 1), dtype=float)
    fit = sm.OLS(x, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})
    mean = float(fit.params[0])
    se = float(fit.bse[0])
    z = float(stats.norm.ppf(0.5 + float(confidence) / 2.0))
    return {
        "n": int(len(x)),
        "mean_period": mean,
        "hac_se_period": se,
        "hac_t": float(fit.tvalues[0]),
        "p_value_two_sided": float(fit.pvalues[0]),
        "ann_mean": float(mean * periods_per_year),
        "ann_mean_ci_low": float((mean - z * se) * periods_per_year),
        "ann_mean_ci_high": float((mean + z * se) * periods_per_year),
    }


def paired_hac_test(
    a: Any,
    b: Any,
    name_a: str,
    name_b: str,
    maxlags: int = DEFAULT_HAC_LAGS,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    confidence: float = 0.95,
    require_identical_index: bool = True,
) -> Dict[str, float]:
    """Paired HAC test of the mean return difference ``a - b``."""
    pair = _paired_frame(a, b, require_identical_index=require_identical_index)
    if len(pair) < 3:
        raise ValueError("paired_hac_test requires at least three paired observations.")
    out = hac_mean_test(
        pair["a"] - pair["b"],
        maxlags=maxlags,
        periods_per_year=periods_per_year,
        confidence=confidence,
    )
    out.update({"model_a": str(name_a), "model_b": str(name_b)})
    return out


def hac_regression(
    data: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    label: str = "",
    maxlags: int = DEFAULT_HAC_LAGS,
    add_constant: bool = True,
) -> Tuple[pd.DataFrame, Any]:
    """Run a generic HAC OLS regression and return a tidy coefficient table."""
    cols = [str(y_col)] + [str(x) for x in x_cols]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError("Missing regression columns: %s" % missing)
    d = data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < max(5, len(x_cols) + 2):
        raise ValueError("Insufficient finite observations for HAC regression.")
    y = d[y_col].astype(float)
    X = d[list(x_cols)].astype(float)
    if add_constant:
        X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})
    rows: List[Dict[str, Any]] = []
    for var in fit.params.index:
        rows.append(
            {
                "regression": str(label),
                "variable": str(var),
                "n": int(fit.nobs),
                "coef": float(fit.params[var]),
                "hac_se": float(fit.bse[var]),
                "hac_t": float(fit.tvalues[var]),
                "p_value_two_sided": float(fit.pvalues[var]),
                "r_squared": float(fit.rsquared),
            }
        )
    return pd.DataFrame(rows), fit


def spanning_regression(
    return_panel: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    label: str,
    maxlags: int = DEFAULT_HAC_LAGS,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    confidence: float = 0.95,
) -> Tuple[Dict[str, float], Any]:
    """HAC return-spanning regression with annualized intercept (alpha)."""
    cols = [y_col] + list(x_cols)
    missing = [c for c in cols if c not in return_panel.columns]
    if missing:
        raise KeyError("Missing spanning columns: %s" % missing)
    d = return_panel[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < max(5, len(x_cols) + 2):
        raise ValueError("Insufficient finite observations for spanning regression.")
    y = d[y_col].astype(float)
    X = sm.add_constant(d[list(x_cols)].astype(float), has_constant="add")
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})
    alpha = float(fit.params["const"])
    alpha_se = float(fit.bse["const"])
    z = float(stats.norm.ppf(0.5 + float(confidence) / 2.0))
    row: Dict[str, float] = {
        "regression": str(label),
        "n": int(fit.nobs),
        "ann_alpha": float(alpha * periods_per_year),
        "ann_alpha_ci_low": float((alpha - z * alpha_se) * periods_per_year),
        "ann_alpha_ci_high": float((alpha + z * alpha_se) * periods_per_year),
        "alpha_t_hac": float(fit.tvalues["const"]),
        "alpha_p_two_sided": float(fit.pvalues["const"]),
        "r_squared": float(fit.rsquared),
    }
    for x in x_cols:
        row["beta_%s" % x] = float(fit.params[x])
        row["beta_%s_t_hac" % x] = float(fit.tvalues[x])
        row["beta_%s_p_two_sided" % x] = float(fit.pvalues[x])
    return row, fit


def spanning_table(
    return_panel: pd.DataFrame,
    specs: Sequence[Tuple[str, str, Sequence[str]]],
    maxlags: int = DEFAULT_HAC_LAGS,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Run several spanning specifications and return one table."""
    rows = []
    for label, y_col, x_cols in specs:
        row, _ = spanning_regression(
            return_panel,
            y_col=y_col,
            x_cols=x_cols,
            label=label,
            maxlags=maxlags,
            periods_per_year=periods_per_year,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def circular_block_indices(n: int, block_len: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate one circular moving-block bootstrap index of length ``n``."""
    n = int(n)
    block_len = int(block_len)
    if n <= 0:
        raise ValueError("n must be positive.")
    if block_len <= 0:
        raise ValueError("block_len must be positive.")
    n_blocks = int(np.ceil(float(n) / float(block_len)))
    starts = rng.randint(0, n, size=n_blocks)
    idx: List[int] = []
    offsets = np.arange(block_len, dtype=int)
    for start in starts:
        idx.extend(((int(start) + offsets) % n).tolist())
    return np.asarray(idx[:n], dtype=int)


def bootstrap_sharpe_ci(
    returns: Any,
    name: str = "strategy",
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    block_len: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_SEED,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    confidence: float = 0.95,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Circular-block bootstrap confidence interval for one annualized Sharpe."""
    x = _numeric_series(returns).dropna().to_numpy(dtype=float)
    if x.size < 3:
        raise ValueError("bootstrap_sharpe_ci requires at least three finite observations.")
    reps = int(reps)
    if reps <= 0:
        raise ValueError("reps must be positive.")
    observed = annualized_sharpe(x, periods_per_year=periods_per_year)
    rng = np.random.RandomState(int(seed))
    draws = np.empty(reps, dtype=float)
    for r in range(reps):
        idx = circular_block_indices(len(x), int(block_len), rng)
        draws[r] = annualized_sharpe(x[idx], periods_per_year=periods_per_year)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        raise RuntimeError("No finite bootstrap Sharpe draws were produced.")
    alpha = (1.0 - float(confidence)) / 2.0
    ci_low, ci_high = np.quantile(draws, [alpha, 1.0 - alpha])
    row = {
        "model": str(name),
        "n_periods": int(len(x)),
        "block_len": int(block_len),
        "bootstrap_reps": int(len(draws)),
        "observed_sharpe": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_prob_sharpe_gt_0": float(np.mean(draws > 0.0)),
    }
    return row, draws


def bootstrap_sharpe_difference(
    a: Any,
    b: Any,
    name_a: str,
    name_b: str,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    block_len: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_SEED,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    confidence: float = 0.95,
    require_identical_index: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Paired circular-block bootstrap CI for ``Sharpe(a) - Sharpe(b)``."""
    pair = _paired_frame(a, b, require_identical_index=require_identical_index)
    if len(pair) < 3:
        raise ValueError("bootstrap_sharpe_difference requires at least three paired observations.")
    av = pair["a"].to_numpy(dtype=float)
    bv = pair["b"].to_numpy(dtype=float)
    observed = annualized_sharpe(av, periods_per_year) - annualized_sharpe(bv, periods_per_year)
    rng = np.random.RandomState(int(seed))
    deltas = np.empty(int(reps), dtype=float)
    for r in range(int(reps)):
        idx = circular_block_indices(len(pair), int(block_len), rng)
        deltas[r] = (
            annualized_sharpe(av[idx], periods_per_year)
            - annualized_sharpe(bv[idx], periods_per_year)
        )
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        raise RuntimeError("No finite bootstrap Sharpe-difference draws were produced.")
    alpha = (1.0 - float(confidence)) / 2.0
    ci_low, ci_high = np.quantile(deltas, [alpha, 1.0 - alpha])
    row = {
        "model_a": str(name_a),
        "model_b": str(name_b),
        "n_periods": int(len(pair)),
        "block_len": int(block_len),
        "bootstrap_reps": int(len(deltas)),
        "observed_delta_sharpe": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        # Legacy confidence-interval field names retained for compatibility.
        "ci_2_5": float(ci_low) if np.isclose(confidence, 0.95) else float("nan"),
        "ci_97_5": float(ci_high) if np.isclose(confidence, 0.95) else float("nan"),
        "bootstrap_prob_delta_gt_0": float(np.mean(deltas > 0.0)),
    }
    return row, deltas


def bootstrap_block_length_sensitivity(
    a: Any,
    b: Any,
    name_a: str,
    name_b: str,
    block_lengths: Sequence[int],
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    require_identical_index: bool = True,
) -> pd.DataFrame:
    """Repeat paired Sharpe inference across alternative block lengths."""
    rows = []
    for j, block_len in enumerate(block_lengths):
        row, _ = bootstrap_sharpe_difference(
            a,
            b,
            name_a,
            name_b,
            reps=reps,
            block_len=int(block_len),
            seed=int(seed) + j,
            periods_per_year=periods_per_year,
            require_identical_index=require_identical_index,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_inference_table(
    return_panel: pd.DataFrame,
    comparisons: Sequence[Tuple[str, str]],
    maxlags: int = DEFAULT_HAC_LAGS,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    block_len: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_SEED,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    require_identical_index: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired HAC mean-return and paired Sharpe tests for several model pairs."""
    hac_rows = []
    sharpe_rows = []
    for j, (a, b) in enumerate(comparisons):
        if a not in return_panel.columns or b not in return_panel.columns:
            raise KeyError("Missing comparison columns: %s, %s" % (a, b))
        hac_rows.append(
            paired_hac_test(
                return_panel[a],
                return_panel[b],
                a,
                b,
                maxlags=maxlags,
                periods_per_year=periods_per_year,
                require_identical_index=require_identical_index,
            )
        )
        row, _ = bootstrap_sharpe_difference(
            return_panel[a],
            return_panel[b],
            a,
            b,
            reps=reps,
            block_len=block_len,
            seed=int(seed) + j,
            periods_per_year=periods_per_year,
            require_identical_index=require_identical_index,
        )
        sharpe_rows.append(row)
    return pd.DataFrame(hac_rows), pd.DataFrame(sharpe_rows)


def pairwise_inference_by_aum(
    return_panels_by_aum: Mapping[float, pd.DataFrame],
    comparisons: Sequence[Tuple[str, str]],
    maxlags: int = DEFAULT_HAC_LAGS,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    block_len: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_SEED,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    require_identical_index: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply paired net-return and Sharpe inference to each AUM return panel."""
    hac_parts = []
    sharpe_parts = []
    for i, aum in enumerate(sorted(float(x) for x in return_panels_by_aum.keys())):
        panel = return_panels_by_aum[aum]
        hac, sharpe = pairwise_inference_table(
            panel,
            comparisons=comparisons,
            maxlags=maxlags,
            reps=reps,
            block_len=block_len,
            seed=int(seed) + 1000 * i,
            periods_per_year=periods_per_year,
            require_identical_index=require_identical_index,
        )
        for d in (hac, sharpe):
            d.insert(0, "AUM_dollars", float(aum))
            d.insert(1, "AUM_millions", float(aum) / 1e6)
        hac_parts.append(hac)
        sharpe_parts.append(sharpe)
    return pd.concat(hac_parts, ignore_index=True), pd.concat(sharpe_parts, ignore_index=True)


def bootstrap_sharpe_panel(
    return_panel: pd.DataFrame,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    block_len: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_SEED,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Bootstrap a Sharpe confidence interval for every column in a return panel."""
    rows = []
    for j, col in enumerate(return_panel.columns):
        row, _ = bootstrap_sharpe_ci(
            return_panel[col],
            name=str(col),
            reps=reps,
            block_len=block_len,
            seed=int(seed) + j,
            periods_per_year=periods_per_year,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_sharpe_inference(
    returns_by_aum: Mapping[float, Any],
    model: str,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    block_len: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_SEED,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Build a grid-based net-Sharpe confidence table across fixed AUM levels."""
    rows = []
    for j, aum in enumerate(sorted(float(x) for x in returns_by_aum.keys())):
        row, _ = bootstrap_sharpe_ci(
            returns_by_aum[aum],
            name=model,
            reps=reps,
            block_len=block_len,
            seed=int(seed) + j,
            periods_per_year=periods_per_year,
        )
        row.update({"AUM_dollars": float(aum), "AUM_millions": float(aum) / 1e6})
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("AUM_dollars").reset_index(drop=True)
    out["point_estimate_positive"] = out["observed_sharpe"] > 0.0
    out["lower_ci_positive"] = out["ci_low"] > 0.0
    return out


def capacity_thresholds(capacity_table: pd.DataFrame) -> Dict[str, float]:
    """Return grid-based economic and statistical capacity thresholds.

    Economic capacity is the largest tested AUM with positive point-estimate net
    Sharpe. Statistical capacity is the largest tested AUM whose lower bootstrap
    Sharpe confidence bound remains above zero. These are grid-based thresholds,
    not interpolated continuous capacity estimates.
    """
    required = {"AUM_dollars", "observed_sharpe", "ci_low"}
    missing = sorted(required - set(capacity_table.columns))
    if missing:
        raise KeyError("Missing capacity columns: %s" % missing)
    d = capacity_table.copy()
    d["AUM_dollars"] = pd.to_numeric(d["AUM_dollars"], errors="coerce")
    d["observed_sharpe"] = pd.to_numeric(d["observed_sharpe"], errors="coerce")
    d["ci_low"] = pd.to_numeric(d["ci_low"], errors="coerce")
    econ = d.loc[d["observed_sharpe"] > 0.0, "AUM_dollars"].dropna()
    statcap = d.loc[d["ci_low"] > 0.0, "AUM_dollars"].dropna()
    return {
        "economic_capacity_max_tested_aum": float(econ.max()) if len(econ) else float("nan"),
        "statistical_capacity_max_tested_aum": float(statcap.max()) if len(statcap) else float("nan"),
    }


def annual_model_stats(
    return_panel: pd.DataFrame,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Calendar-year annualized mean/volatility/Sharpe for every model column."""
    if not isinstance(return_panel.index, pd.DatetimeIndex):
        raise TypeError("annual_model_stats requires a DatetimeIndex.")
    rows: List[Dict[str, Any]] = []
    for year, g in return_panel.groupby(return_panel.index.year):
        for model in return_panel.columns:
            x = _numeric_series(g[model]).dropna()
            if len(x) < 3:
                continue
            stats_row = performance_stats(x, periods_per_year=periods_per_year)
            rows.append(
                {
                    "year": int(year),
                    "model": str(model),
                    "n_weeks": int(len(x)),
                    "ann_mean": stats_row["ann_mean"],
                    "ann_vol": stats_row["ann_vol"],
                    "sharpe": stats_row["sharpe"],
                }
            )
    return pd.DataFrame(rows)


def year_win_summary(
    annual: pd.DataFrame,
    benchmark: str,
    focal_model: str = "HMM",
) -> Dict[str, float]:
    """Summarize calendar-year return and Sharpe wins versus a benchmark."""
    p = annual.pivot(index="year", columns="model", values=["ann_mean", "sharpe"])
    needed = [
        ("ann_mean", focal_model),
        ("ann_mean", benchmark),
        ("sharpe", focal_model),
        ("sharpe", benchmark),
    ]
    p = p.dropna(subset=needed)
    return {
        "focal_model": str(focal_model),
        "benchmark": str(benchmark),
        "n_years": int(len(p)),
        "focal_higher_return_years": int((p[("ann_mean", focal_model)] > p[("ann_mean", benchmark)]).sum()),
        "focal_higher_sharpe_years": int((p[("sharpe", focal_model)] > p[("sharpe", benchmark)]).sum()),
        "mean_annual_return_difference": float((p[("ann_mean", focal_model)] - p[("ann_mean", benchmark)]).mean()),
        "mean_annual_sharpe_difference": float((p[("sharpe", focal_model)] - p[("sharpe", benchmark)]).mean()),
    }


def rolling_sharpe(
    returns: Any,
    window: int,
    min_periods: Optional[int] = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    s = _numeric_series(returns)
    if int(window) <= 1:
        raise ValueError("window must exceed one period.")
    if min_periods is None:
        min_periods = max(20, int(int(window) * 0.75))
    mean = s.rolling(int(window), min_periods=int(min_periods)).mean()
    vol = s.rolling(int(window), min_periods=int(min_periods)).std(ddof=1)
    return mean / vol * np.sqrt(float(periods_per_year))


def conditional_performance_table(
    data: pd.DataFrame,
    group_col: str,
    model_a: str = "HMM",
    model_b: str = "LogisticStack_WF",
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Conditional return/Sharpe table used by the state-mechanism analysis."""
    rows: List[Dict[str, Any]] = []
    for key, g in data.groupby(group_col, observed=True):
        pair = g[[model_a, model_b]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 5:
            continue
        sr_a = annualized_sharpe(pair[model_a], periods_per_year)
        sr_b = annualized_sharpe(pair[model_b], periods_per_year)
        rows.append(
            {
                group_col: key,
                "n_weeks": int(len(pair)),
                "%s_ann_mean" % model_a: float(pair[model_a].mean() * periods_per_year),
                "%s_ann_mean" % model_b: float(pair[model_b].mean() * periods_per_year),
                "difference_ann_mean": float((pair[model_a] - pair[model_b]).mean() * periods_per_year),
                "%s_sharpe" % model_a: float(sr_a),
                "%s_sharpe" % model_b: float(sr_b),
                "delta_sharpe": float(sr_a - sr_b),
            }
        )
    return pd.DataFrame(rows)


def _long_run_std_hac(x: np.ndarray, maxlags: int) -> float:
    """Estimate long-run SD of sqrt(n) times the sample mean via HAC OLS."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    X = np.ones((len(x), 1), dtype=float)
    fit = sm.OLS(x, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})
    return float(fit.bse[0] * np.sqrt(len(x)))


def hansen_spa_test(
    return_panel: pd.DataFrame,
    benchmark: str,
    candidates: Optional[Sequence[str]] = None,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    block_len: int = DEFAULT_BLOCK_LENGTH,
    maxlags: int = DEFAULT_HAC_LAGS,
    seed: int = DEFAULT_SEED,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Hansen-style Superior Predictive Ability test using return differentials.

    Parameters
    ----------
    return_panel:
        Columns are contemporaneous strategy return streams.  Higher return is
        treated as better performance, so the loss differential is represented
        as candidate return minus benchmark return.
    benchmark:
        Benchmark column under the null.
    candidates:
        Candidate columns tested jointly.  Defaults to every other column.

    Notes
    -----
    The test statistic is the maximum studentized mean differential.  Circular
    blocks preserve serial dependence and use the same resampled weeks for every
    candidate.  Hansen's consistent recentering rule is used to reduce the
    influence of clearly inferior alternatives.  The result is one-sided:

        H0: max_k E[r_k - r_benchmark] <= 0
        H1: at least one candidate has positive expected differential.

    This is intended as a data-snooping robustness diagnostic, not as a claim
    that the candidate set represents every strategy ever considered.
    """
    if benchmark not in return_panel.columns:
        raise KeyError("Benchmark column not found: %s" % benchmark)
    if candidates is None:
        candidates = [c for c in return_panel.columns if c != benchmark]
    candidates = [str(c) for c in candidates if str(c) != str(benchmark)]
    if len(candidates) == 0:
        raise ValueError("At least one candidate strategy is required for SPA.")
    missing = [c for c in candidates if c not in return_panel.columns]
    if missing:
        raise KeyError("Candidate columns not found: %s" % missing)

    cols = [benchmark] + list(candidates)
    d = return_panel[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10:
        raise ValueError("SPA requires at least 10 common finite observations.")
    benchmark_values = d[benchmark].to_numpy(dtype=float)
    diff = np.column_stack([d[c].to_numpy(dtype=float) - benchmark_values for c in candidates])
    n, k = diff.shape
    means = diff.mean(axis=0)
    omega = np.asarray([_long_run_std_hac(diff[:, j], maxlags=maxlags) for j in range(k)], dtype=float)
    valid = np.isfinite(omega) & (omega > 1e-14)
    if not np.any(valid):
        raise RuntimeError("SPA could not estimate positive long-run variance for any candidate.")

    candidates_valid = [c for c, ok in zip(candidates, valid) if ok]
    diff = diff[:, valid]
    means = means[valid]
    omega = omega[valid]
    studentized = np.sqrt(float(n)) * means / omega
    observed = float(max(0.0, np.max(studentized)))

    # Hansen (2005) consistent recentering: very poor alternatives retain their
    # negative estimated mean; competitive alternatives are recentered to zero.
    threshold = -np.sqrt(2.0 * np.log(np.log(float(n)))) if n > np.e else float("-inf")
    mu_consistent = np.where(studentized <= threshold, means, 0.0)

    rng = np.random.RandomState(int(seed))
    draws = np.empty(int(reps), dtype=float)
    for r in range(int(reps)):
        idx = circular_block_indices(n, int(block_len), rng)
        boot_means = diff[idx, :].mean(axis=0)
        centered = boot_means - means + mu_consistent
        t_boot = np.sqrt(float(n)) * centered / omega
        draws[r] = max(0.0, float(np.max(t_boot)))

    p_value = float((1.0 + np.sum(draws >= observed)) / (len(draws) + 1.0))
    best_idx = int(np.argmax(studentized))
    result: Dict[str, Any] = {
        "benchmark": str(benchmark),
        "n_periods": int(n),
        "n_candidates": int(len(candidates_valid)),
        "block_len": int(block_len),
        "bootstrap_reps": int(len(draws)),
        "hac_lags": int(maxlags),
        "spa_statistic": float(observed),
        "p_value_one_sided": p_value,
        "best_candidate": str(candidates_valid[best_idx]),
        "best_candidate_mean_diff_period": float(means[best_idx]),
        "best_candidate_studentized": float(studentized[best_idx]),
        "rejected_at_5pct": bool(p_value < 0.05),
        "candidates_used": tuple(candidates_valid),
    }
    return result, draws


def expected_max_sharpe(
    trial_sharpes: Sequence[float],
    effective_num_trials: Optional[float] = None,
) -> float:
    """Expected maximum Sharpe across multiple trials (Bailey/Lopez de Prado approximation)."""
    sr = np.asarray(list(trial_sharpes), dtype=float)
    sr = sr[np.isfinite(sr)]
    if sr.size == 0:
        return float("nan")
    n_trials = float(effective_num_trials) if effective_num_trials is not None else float(len(sr))
    if (not np.isfinite(n_trials)) or n_trials < 1.0:
        raise ValueError("effective_num_trials must be at least 1.")
    mean_sr = float(np.mean(sr))
    if sr.size < 2 or n_trials <= 1.0:
        return mean_sr
    sd_sr = float(np.std(sr, ddof=1))
    if sd_sr <= 0.0 or not np.isfinite(sd_sr):
        return mean_sr
    p1 = float(np.clip(1.0 - 1.0 / n_trials, 1e-12, 1.0 - 1e-12))
    p2 = float(np.clip(1.0 - 1.0 / (n_trials * np.e), 1e-12, 1.0 - 1e-12))
    z1 = float(stats.norm.ppf(p1))
    z2 = float(stats.norm.ppf(p2))
    return float(mean_sr + sd_sr * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def probabilistic_sharpe_ratio(
    returns: Any,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """Probabilistic Sharpe Ratio relative to an annualized benchmark Sharpe."""
    x = _numeric_series(returns).dropna().to_numpy(dtype=float)
    if x.size < 4:
        raise ValueError("probabilistic_sharpe_ratio requires at least four observations.")
    observed_ann = annualized_sharpe(x, periods_per_year=periods_per_year)
    observed = float(observed_ann / np.sqrt(float(periods_per_year)))
    benchmark = float(benchmark_sharpe) / np.sqrt(float(periods_per_year))
    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, fisher=False, bias=False))
    denom2 = 1.0 - skew * observed + ((kurt - 1.0) / 4.0) * observed * observed
    if (not np.isfinite(denom2)) or denom2 <= 0.0:
        z = float("nan")
        prob = float("nan")
    else:
        z = float((observed - benchmark) * np.sqrt(float(len(x) - 1)) / np.sqrt(denom2))
        prob = float(stats.norm.cdf(z))
    return {
        "n_periods": int(len(x)),
        "observed_sharpe": float(observed_ann),
        "benchmark_sharpe": float(benchmark_sharpe),
        "skewness": skew,
        "kurtosis_nonexcess": kurt,
        "psr_z": z,
        "probability_sharpe_exceeds_benchmark": prob,
    }


def deflated_sharpe_ratio(
    selected_returns: Any,
    trial_sharpes: Sequence[float],
    effective_num_trials: Optional[float] = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """Deflated Sharpe Ratio using the expected maximum trial Sharpe as hurdle."""
    hurdle = expected_max_sharpe(trial_sharpes, effective_num_trials=effective_num_trials)
    out = probabilistic_sharpe_ratio(
        selected_returns,
        benchmark_sharpe=hurdle,
        periods_per_year=periods_per_year,
    )
    out["deflated_sharpe_hurdle"] = float(hurdle)
    out["effective_num_trials"] = float(effective_num_trials) if effective_num_trials is not None else float(len([x for x in trial_sharpes if np.isfinite(x)]))
    out["deflated_sharpe_probability"] = out["probability_sharpe_exceeds_benchmark"]
    return out


def deflated_sharpe_from_panel(
    return_panel: pd.DataFrame,
    selected_model: str,
    trial_models: Optional[Sequence[str]] = None,
    effective_num_trials: Optional[float] = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """Convenience wrapper computing trial Sharpes from a return panel."""
    if selected_model not in return_panel.columns:
        raise KeyError("Selected model not found: %s" % selected_model)
    if trial_models is None:
        trial_models = list(return_panel.columns)
    missing = [c for c in trial_models if c not in return_panel.columns]
    if missing:
        raise KeyError("Trial model columns not found: %s" % missing)
    trial_sharpes = [annualized_sharpe(return_panel[c], periods_per_year) for c in trial_models]
    out = deflated_sharpe_ratio(
        return_panel[selected_model],
        trial_sharpes=trial_sharpes,
        effective_num_trials=effective_num_trials,
        periods_per_year=periods_per_year,
    )
    out["selected_model"] = str(selected_model)
    out["n_trial_models"] = int(len(trial_models))
    return out


__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "DEFAULT_HAC_LAGS",
    "DEFAULT_BOOTSTRAP_REPS",
    "DEFAULT_BLOCK_LENGTH",
    "DEFAULT_SEED",
    "annualized_sharpe",
    "performance_stats",
    "hac_mean_test",
    "paired_hac_test",
    "hac_regression",
    "spanning_regression",
    "spanning_table",
    "circular_block_indices",
    "bootstrap_sharpe_ci",
    "bootstrap_sharpe_difference",
    "bootstrap_block_length_sensitivity",
    "pairwise_inference_table",
    "pairwise_inference_by_aum",
    "bootstrap_sharpe_panel",
    "capacity_sharpe_inference",
    "capacity_thresholds",
    "annual_model_stats",
    "year_win_summary",
    "rolling_sharpe",
    "conditional_performance_table",
    "hansen_spa_test",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "deflated_sharpe_from_panel",
]
