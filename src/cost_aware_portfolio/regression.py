"""Regression checks for preserving gross portfolio outputs across refactors.

The transaction-cost refactor is allowed to change implementation-aware net
returns, costs, and liquidity-constrained holdings.  It must not silently change
an otherwise identical gross portfolio calculation.

This module provides small reusable checks for the final analysis notebook.  A
caller supplies a previously saved gross return stream and the corresponding
new stream.  The functions align dates, report numerical differences, and can
fail closed if the streams are not identical within a chosen tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import math

import numpy as np
import pandas as pd


def periods_per_year(freq: str) -> int:
    if freq == "week":
        return 52
    if freq == "month":
        return 12
    if freq == "quarter":
        return 4
    raise ValueError("freq must be one of: week, month, quarter.")


def _as_return_series(values: pd.Series, *, name: str) -> pd.Series:
    s = pd.Series(values).copy()
    idx = pd.to_datetime(pd.Index(s.index), errors="coerce")
    if pd.isna(idx).any():
        raise ValueError("%s contains invalid dates." % name)
    s.index = pd.DatetimeIndex(idx).normalize()
    if s.index.has_duplicates:
        raise ValueError("%s contains duplicate dates." % name)
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if s.isna().any() or (~np.isfinite(s.to_numpy(dtype=float))).any():
        raise ValueError("%s contains missing or non-finite returns." % name)
    return s.sort_index()


def _stats(s: pd.Series, freq: str) -> Dict[str, float]:
    ppy = float(periods_per_year(freq))
    mean_period = float(s.mean()) if len(s) else float("nan")
    vol_period = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    ann_mean = mean_period * ppy
    ann_vol = vol_period * math.sqrt(ppy)
    sharpe = ann_mean / ann_vol if ann_vol > 0.0 and np.isfinite(ann_vol) else float("nan")
    return {
        "ann_mean": ann_mean,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }


@dataclass(frozen=True)
class GrossRegressionResult:
    label: str
    passed: bool
    n_reference: int
    n_current: int
    n_overlap: int
    missing_from_current: int
    extra_in_current: int
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    correlation: float
    reference_ann_mean: float
    current_ann_mean: float
    reference_ann_vol: float
    current_ann_vol: float
    reference_sharpe: float
    current_sharpe: float
    sharpe_diff: float

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def compare_gross_return_series(
    reference: pd.Series,
    current: pd.Series,
    *,
    label: str = "H-L",
    freq: str = "week",
    atol: float = 1e-12,
    rtol: float = 1e-10,
    require_same_dates: bool = True,
) -> GrossRegressionResult:
    """Compare two gross return streams without changing either one."""
    ref = _as_return_series(reference, name="reference gross returns")
    cur = _as_return_series(current, name="current gross returns")

    missing = ref.index.difference(cur.index)
    extra = cur.index.difference(ref.index)
    overlap = ref.index.intersection(cur.index)
    if len(overlap) == 0:
        raise ValueError("Gross regression check has no overlapping dates.")

    ref_o = ref.reindex(overlap)
    cur_o = cur.reindex(overlap)
    diff = cur_o - ref_o
    abs_diff = diff.abs()
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    rmse = float(np.sqrt(np.mean(np.square(diff.to_numpy(dtype=float)))))

    if len(overlap) > 1 and float(ref_o.std(ddof=1)) > 0.0 and float(cur_o.std(ddof=1)) > 0.0:
        corr = float(ref_o.corr(cur_o))
    else:
        corr = float("nan")

    values_equal = bool(
        np.allclose(
            ref_o.to_numpy(dtype=float),
            cur_o.to_numpy(dtype=float),
            atol=float(atol),
            rtol=float(rtol),
        )
    )
    dates_equal = (len(missing) == 0 and len(extra) == 0)
    passed = values_equal and (dates_equal or not bool(require_same_dates))

    ref_stats = _stats(ref_o, freq)
    cur_stats = _stats(cur_o, freq)
    return GrossRegressionResult(
        label=str(label),
        passed=bool(passed),
        n_reference=int(len(ref)),
        n_current=int(len(cur)),
        n_overlap=int(len(overlap)),
        missing_from_current=int(len(missing)),
        extra_in_current=int(len(extra)),
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        rmse=rmse,
        correlation=corr,
        reference_ann_mean=float(ref_stats["ann_mean"]),
        current_ann_mean=float(cur_stats["ann_mean"]),
        reference_ann_vol=float(ref_stats["ann_vol"]),
        current_ann_vol=float(cur_stats["ann_vol"]),
        reference_sharpe=float(ref_stats["sharpe"]),
        current_sharpe=float(cur_stats["sharpe"]),
        sharpe_diff=float(cur_stats["sharpe"] - ref_stats["sharpe"]),
    )


def assert_gross_return_series_unchanged(
    reference: pd.Series,
    current: pd.Series,
    *,
    label: str = "H-L",
    freq: str = "week",
    atol: float = 1e-12,
    rtol: float = 1e-10,
    require_same_dates: bool = True,
) -> GrossRegressionResult:
    """Run the comparison and raise if the gross stream changed unexpectedly."""
    result = compare_gross_return_series(
        reference,
        current,
        label=label,
        freq=freq,
        atol=atol,
        rtol=rtol,
        require_same_dates=require_same_dates,
    )
    if not result.passed:
        raise AssertionError(
            "Gross regression failed for %s: missing=%d extra=%d max_abs_diff=%.6e "
            "mean_abs_diff=%.6e sharpe_diff=%.6e"
            % (
                result.label,
                result.missing_from_current,
                result.extra_in_current,
                result.max_abs_diff,
                result.mean_abs_diff,
                result.sharpe_diff,
            )
        )
    return result


def compare_gross_return_frames(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: Optional[Iterable[str]] = None,
    freq: str = "week",
    atol: float = 1e-12,
    rtol: float = 1e-10,
    require_same_dates: bool = True,
) -> pd.DataFrame:
    """Compare matching gross-return columns and return one audit row per column."""
    if columns is None:
        cols = [str(c) for c in reference.columns if c in current.columns]
    else:
        cols = [str(c) for c in columns]
    if not cols:
        raise ValueError("No gross-return columns were selected for comparison.")

    rows = []
    for col in cols:
        if col not in reference.columns or col not in current.columns:
            raise KeyError("Gross regression column %r is missing from one frame." % col)
        result = compare_gross_return_series(
            reference[col],
            current[col],
            label=col,
            freq=freq,
            atol=atol,
            rtol=rtol,
            require_same_dates=require_same_dates,
        )
        rows.append(result.to_dict())
    return pd.DataFrame(rows)


__all__ = [
    "GrossRegressionResult",
    "assert_gross_return_series_unchanged",
    "compare_gross_return_frames",
    "compare_gross_return_series",
    "periods_per_year",
]
