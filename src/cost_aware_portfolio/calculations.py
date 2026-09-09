"""Small statistical and portfolio helper functions.

This module contains shared helpers used by the portfolio manager and strategy
classes for correlations, decile membership, scaling, and turnover.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class PortfolioMath:
    """Collection of lightweight portfolio and diagnostic utilities."""

    # Coerce two series to aligned numeric vectors and drop NaN and inf pairs.
    @staticmethod
    def _aligned_numeric(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        xs = pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan)
        ys = pd.to_numeric(pd.Series(y), errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = xs.notna() & ys.notna()
        return xs[mask], ys[mask]

    # Compute a Pearson correlation from two numeric arrays.
    @staticmethod
    def _pearson_from_arrays(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    # Compute a NaN-safe Pearson correlation between two series.
    @staticmethod
    def _safe_pearson(x: pd.Series, y: pd.Series) -> float:
        xs, ys = PortfolioMath._aligned_numeric(x, y)
        if len(xs) < 3 or xs.nunique() < 2 or ys.nunique() < 2:
            return float("nan")
        return PortfolioMath._pearson_from_arrays(xs.to_numpy(dtype=float), ys.to_numpy(dtype=float))

    # Compute a NaN-safe Spearman correlation between two series.
    @staticmethod
    def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
        xs, ys = PortfolioMath._aligned_numeric(x, y)
        if len(xs) < 3 or xs.nunique() < 2 or ys.nunique() < 2:
            return float("nan")
        rx = xs.rank(method="average")
        ry = ys.rank(method="average")
        if rx.nunique() < 2 or ry.nunique() < 2:
            return float("nan")
        return PortfolioMath._pearson_from_arrays(rx.to_numpy(dtype=float), ry.to_numpy(dtype=float))

    # Shift a series to strictly positive support.
    @staticmethod
    def _normalize_alpha(series: pd.Series, eps: float = 1e-8) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").astype(float)
        return (s - float(s.min())) + float(eps)

    # Compute the percentile bounds for one decile bucket.
    @staticmethod
    def _decile_bounds(up: pd.Series, cut: int, decile_idx: int) -> Tuple[float, float]:
        low = float(np.percentile(up, decile_idx * 100.0 / cut))
        high = float(np.percentile(up, (decile_idx + 1) * 100.0 / cut))
        return low, high

    # Build the membership mask for one decile bucket.
    @staticmethod
    def _decile_mask(up: pd.Series, low: float, high: float, is_lowest: bool) -> pd.Series:
        return ((up >= low) & (up <= high)) if is_lowest else ((up > low) & (up <= high))

    # Reindex a numeric series to a target index with a fill value.
    @staticmethod
    def _safe_reindex(s: pd.Series, idx: pd.Index, fill: float = 0.0) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").reindex(idx).fillna(float(fill))

    # Rescale a series to a target gross exposure.
    @staticmethod
    def _scale_to_gross(w: pd.Series, gross_target: float) -> pd.Series:
        ww = pd.to_numeric(w, errors="coerce").astype(float)
        gross = float(np.abs(ww).sum())
        if gross <= 0 or not np.isfinite(gross):
            return ww
        return (float(gross_target) / gross) * ww

    # Compute one-way turnover after drifting prior signed weights and re-normalizing gross exposure.
    @staticmethod
    def _gross_normalized_turnover(cur_w: pd.Series, prev_w: pd.Series, prev_rets: pd.Series) -> float:
        cur = pd.to_numeric(cur_w, errors="coerce").astype(float)
        prev = pd.to_numeric(prev_w, errors="coerce").reindex(cur.index).fillna(0.0).astype(float)
        rets = pd.to_numeric(prev_rets, errors="coerce").reindex(cur.index).fillna(0.0).astype(float)
        drifted = prev * (1.0 + rets)
        gross_cur = float(np.abs(cur).sum())
        gross_drifted = float(np.abs(drifted).sum())
        if gross_cur <= 0 or not np.isfinite(gross_cur):
            gross_cur = gross_drifted
        if gross_drifted > 0 and np.isfinite(gross_drifted) and gross_cur > 0 and np.isfinite(gross_cur):
            drifted = drifted * (gross_cur / gross_drifted)
        return 0.5 * float(np.abs(cur - drifted).sum())

    # Compute a long-only-style leg return after normalizing by absolute weight sum.
    @staticmethod
    def _leg_return_longonly_style(weights: pd.Series, rets: pd.Series) -> float:
        ww = pd.to_numeric(weights, errors="coerce").abs()
        total = float(ww.sum())
        if total <= 0 or not np.isfinite(total):
            return 0.0
        ww = ww / total
        rr = pd.to_numeric(rets, errors="coerce").reindex(ww.index).fillna(0.0)
        return float((ww * rr).sum())
