"""
CovarianceEngine: rolling and EWMA covariance estimation for portfolio construction.

This module builds Date×StockID return pivots and returns stabilized covariance
matrices for a requested rebalance date and asset subset.

Design goals
------------
1) Keep the public interface unchanged for PortfolioManager and strategy classes.
2) Fail loudly on true structural problems such as missing returns, too few assets,
   or too little total history.
3) Repair normal numerical pathologies that arise in sparse rolling covariance
   estimation, especially:
   - undefined off-diagonal entries from insufficient pairwise overlap,
   - non-positive or non-finite diagonal entries,
   - mild non-PSD covariance matrices.
4) Preserve a small ridge parameter (`mw_ridge`) as a numerical stabilizer rather
   than turning it into a broad fallback system.

Stabilization policy
--------------------
After estimating a covariance matrix, the engine applies a narrow repair layer:
- symmetrize,
- repair bad diagonal entries using the median positive variance,
- set remaining bad off-diagonal entries to 0.0,
- apply spectral PSD repair by clipping negative eigenvalues to a tiny floor,
- re-symmetrize and add the configured ridge.

This is intentionally stricter than broad silent fallbacks and is designed to
prevent sparse pairwise overlap from crashing Markowitz-style portfolio runs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class CovarianceEngine:
    """
    Date-aware covariance engine used by portfolio strategies.

    The engine maintains a cached Date×StockID return pivot and produces
    covariance matrices Σ_t(S) for a requested date t and asset subset S.

    Supported modes
    ---------------
    rolling
        Pairwise-available sample covariance on the trailing `mw_window`,
        requiring at least `mw_min_periods` observations at the asset level.

    ewma
        Exponentially weighted covariance over the full return history up to the
        requested date, with optional mean centering.

    Important behavior
    ------------------
    - Missing data are handled pairwise during estimation.
    - Undefined off-diagonal entries are explicitly cleaned rather than allowed
      to crash downstream optimizers.
    - A small diagonal ridge (`mw_ridge`) is added after covariance repair.
    """

    # Initialize configuration, pivot placeholders, EWMA state, and caches.
    def __init__(
        self,
        *,
        mw_window: int,
        mw_min_periods: int,
        mw_ridge: float,
        cov_mode: str,
        ewma_alpha: float,
        ewma_center: bool,
    ) -> None:
        self.mw_window = int(mw_window)
        self.mw_min_periods = int(mw_min_periods)
        self.mw_ridge = float(mw_ridge)
        if cov_mode not in ("rolling", "ewma"):
            raise ValueError(f"Unsupported cov_mode={cov_mode!r}; expected 'rolling' or 'ewma'.")
        self.cov_mode = cov_mode
        self.ewma_alpha = float(ewma_alpha)
        self.ewma_center = bool(ewma_center)

        self._ret_pivot: Optional[pd.DataFrame] = None
        self._ret_pivot_name: Optional[str] = None
        self._dates_sorted: Optional[pd.DatetimeIndex] = None

        self._ewma_state: Optional[dict] = None
        self._ewma_memo_enabled: bool = True

        self._col_indexer: Dict[str, int] = {}

    # Attach a return panel and rebuild the pivot when the return column changes.
    def attach_returns_panel(self, df: pd.DataFrame, ret_name: str) -> None:
        self._ensure_pivot(df, ret_name)

    # Build a stabilized covariance matrix for the requested date and asset subset.
    def build_cov(
        self,
        date: pd.Timestamp,
        asset_ids: pd.Index,
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        piv = self._ret_pivot
        dates = self._dates_sorted
        if piv is None or dates is None:
            raise RuntimeError("CovarianceEngine.build_cov called before a returns pivot was attached.")

        cols = [str(c) for c in pd.Index(asset_ids) if str(c) in piv.columns]
        if len(cols) < 2:
            raise RuntimeError(
                f"CovarianceEngine.build_cov requires at least two assets in the return pivot; got {len(cols)}."
            )

        i = dates.searchsorted(pd.Timestamp(date))
        if i < self.mw_min_periods:
            raise RuntimeError(
                f"CovarianceEngine.build_cov has insufficient history for date={pd.Timestamp(date)}."
            )

        if self.cov_mode == "rolling":
            start = max(0, i - self.mw_window)
            win_dates = dates[start:i]
            if len(win_dates) < self.mw_min_periods:
                raise RuntimeError(
                    f"CovarianceEngine.build_cov rolling window is too short for date={pd.Timestamp(date)}."
                )
            r_sub = piv.loc[win_dates, cols].to_numpy(dtype=np.float32, copy=False)
            sigma = self._compute_sample_cov(r_sub, min_periods=self.mw_min_periods)
            return sigma, cols

        if not self._ewma_memo_enabled:
            start = max(0, i - self.mw_window)
            win_dates = dates[start:i]
            if len(win_dates) < self.mw_min_periods:
                raise RuntimeError(
                    f"CovarianceEngine.build_cov EWMA window is too short for date={pd.Timestamp(date)}."
                )
            r_sub = piv.loc[win_dates, cols].to_numpy(dtype=np.float32, copy=False)
            sigma = self._compute_ewma_cov(r_sub, alpha=self.ewma_alpha, center=self.ewma_center)
            return sigma, cols

        idx = [self._col_indexer[c] for c in cols]
        sigma = self._ewma_cov_for_subset(i, idx)
        if sigma is None:
            raise RuntimeError(
                f"CovarianceEngine.build_cov could not materialize the EWMA covariance for date={pd.Timestamp(date)}."
            )
        return sigma, cols

    # Build or refresh the Date×StockID return pivot and reset dependent state.
    def _ensure_pivot(self, df: pd.DataFrame, ret_name: str) -> None:
        if self._ret_pivot is not None and self._ret_pivot_name == ret_name:
            return

        frame = df.reset_index()
        if ret_name not in frame.columns:
            raise KeyError(f"Return column not found in frame: {ret_name}")
        tmp = frame[["Date", "StockID", ret_name]].copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp["StockID"] = tmp["StockID"].astype(str)
        tmp = tmp.dropna(subset=["Date"])

        self._ret_pivot = tmp.pivot(index="Date", columns="StockID", values=ret_name).sort_index()
        self._ret_pivot_name = ret_name
        self._dates_sorted = self._ret_pivot.index
        self._ewma_state = None
        self._col_indexer = {str(c): i for i, c in enumerate(self._ret_pivot.columns)}

    # Return a repaired, symmetric, PSD covariance matrix with ridge added.
    def _stabilize_covariance(self, sigma: np.ndarray, *, diag_floor: Optional[float] = None) -> np.ndarray:
        c = np.asarray(sigma, dtype=np.float64)
        if c.ndim != 2 or c.shape[0] != c.shape[1]:
            raise RuntimeError(f"Covariance stabilization requires a square matrix; got shape {c.shape}.")

        n = c.shape[0]
        if n == 0:
            raise RuntimeError("Covariance stabilization received an empty matrix.")

        # Symmetrize first so later repairs operate on a symmetric matrix.
        c = 0.5 * (c + c.T)

        d = np.diag(c).copy()
        pos = np.isfinite(d) & (d > 0)
        if diag_floor is None:
            if not np.any(pos):
                raise RuntimeError(
                    "Covariance stabilization found no positive finite variances; "
                    "refusing to synthesize an arbitrary diagonal scale."
                )
            diag_floor = float(np.nanmedian(d[pos]))
        if (not np.isfinite(float(diag_floor))) or float(diag_floor) <= 0.0:
            raise RuntimeError("Covariance stabilization requires a positive finite diagonal repair scale.")
        diag_floor = max(float(diag_floor), 1e-10)

        # Repair bad diagonal entries using the median positive variance.
        bad_diag = ~np.isfinite(d) | (d <= 0)
        if np.any(bad_diag):
            d[bad_diag] = diag_floor
            np.fill_diagonal(c, d)

        # Replace any remaining bad off-diagonal entries with 0.0.
        bad = ~np.isfinite(c)
        if np.any(bad):
            offdiag_bad = bad.copy()
            np.fill_diagonal(offdiag_bad, False)
            if np.any(offdiag_bad):
                c[offdiag_bad] = 0.0

        # Re-symmetrize after off-diagonal cleanup.
        c = 0.5 * (c + c.T)

        # Spectral PSD repair by clipping eigenvalues to a tiny floor.
        try:
            evals, evecs = np.linalg.eigh(c)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Covariance stabilization failed during eigendecomposition.") from exc

        eig_floor = max(float(self.mw_ridge), 1e-10)
        evals_clipped = np.clip(evals, eig_floor, None)
        c = (evecs * evals_clipped) @ evecs.T

        # Final symmetrization and diagonal ridge.
        c = 0.5 * (c + c.T)
        c = c + float(self.mw_ridge) * np.eye(n, dtype=np.float64)

        final_diag = np.diag(c)
        if not np.isfinite(c).all():
            bad_count = int((~np.isfinite(c)).sum())
            raise RuntimeError(
                f"CovarianceEngine._stabilize_covariance produced {bad_count} non-finite entries after repair."
            )
        if np.any(final_diag <= 0) or not np.isfinite(final_diag).all():
            raise RuntimeError("CovarianceEngine._stabilize_covariance produced a non-positive or non-finite diagonal.")

        return c.astype(np.float32, copy=False)

    # Ensure the cached EWMA state matches the current pivot and config.
    def _ensure_ewma_state_current(self) -> None:
        if self._ret_pivot is None:
            self._ewma_state = None
            return

        cols = list(self._ret_pivot.columns)
        need_reset = False
        if self._ewma_state is None:
            need_reset = True
        else:
            st = self._ewma_state
            if abs(st["alpha"] - float(self.ewma_alpha)) > 1e-12:
                need_reset = True
            if bool(st["center"]) != bool(self.ewma_center):
                need_reset = True
            if st["cols"] != cols:
                need_reset = True

        if need_reset:
            n = len(cols)
            self._ewma_state = {
                "alpha": float(self.ewma_alpha),
                "beta": 1.0 - float(self.ewma_alpha),
                "center": bool(self.ewma_center),
                "cols": cols,
                "S": np.zeros((n, n), dtype=np.float32),
                "W": np.zeros((n, n), dtype=np.float32),
                "mu": np.zeros(n, dtype=np.float32),
                "last_i": 0,
            }

    # Advance the memoized EWMA state up to the requested pivot row.
    def _ewma_advance_to(self, i_target: int) -> None:
        st = self._ewma_state
        piv = self._ret_pivot
        if st is None or piv is None:
            return

        beta = st["beta"]
        center = st["center"]
        s_mat, w_mat, mu = st["S"], st["W"], st["mu"]
        block = 1024

        for t in range(st["last_i"], i_target):
            row = piv.iloc[t].to_numpy(dtype=np.float32, copy=False)
            msk = np.isfinite(row)
            idx = np.flatnonzero(msk)

            s_mat *= beta
            w_mat *= beta
            if idx.size == 0:
                continue

            r_obs = row[idx].astype(np.float32, copy=False)
            if center:
                mu[idx] = beta * mu[idx] + (1.0 - beta) * r_obs
                x_obs = r_obs - mu[idx]
            else:
                x_obs = r_obs

            tw = (1.0 - beta)
            nobs = idx.size

            for a in range(0, nobs, block):
                ia = idx[a:a + block]
                for b in range(a, nobs, block):
                    ib = idx[b:b + block]
                    w_mat[np.ix_(ia, ib)] += tw
                    if b != a:
                        w_mat[np.ix_(ib, ia)] += tw

            for a in range(0, nobs, block):
                ia = idx[a:a + block]
                va = x_obs[a:a + block][:, None]
                for b in range(a, nobs, block):
                    ib = idx[b:b + block]
                    vb = x_obs[b:b + block][None, :]
                    s_mat[np.ix_(ia, ib)] += tw * (va * vb)
                    if b != a:
                        s_mat[np.ix_(ib, ia)] += tw * (vb.T * va.T)

        st["mu"] = mu
        st["last_i"] = i_target

    # Materialize a stabilized EWMA covariance matrix for the requested subset.
    def _ewma_cov_for_subset(self, i: int, idx: List[int]) -> Optional[np.ndarray]:
        self._ensure_ewma_state_current()
        if self._ewma_state is None or i < self.mw_min_periods:
            return None

        self._ewma_advance_to(i)

        st = self._ewma_state
        s_mat = st["S"]
        w_mat = st["W"]
        tiny = 1e-12

        s_sub = s_mat[np.ix_(idx, idx)]
        w_sub = w_mat[np.ix_(idx, idx)]
        c = s_sub / np.maximum(w_sub, tiny)

        d_all = np.diag(s_mat) / np.maximum(np.diag(w_mat), tiny)
        d_all = d_all.astype(np.float32, copy=False)
        pos_all = np.isfinite(d_all) & (d_all > 0)
        if not np.any(pos_all):
            raise RuntimeError(
                "EWMA covariance state contains no positive finite variances for diagonal repair."
            )
        med_all = float(np.nanmedian(d_all[pos_all]))
        return self._stabilize_covariance(c, diag_floor=med_all)

    # Compute a stabilized finite-window EWMA covariance matrix on a T×n block.
    def _compute_ewma_cov(self, r: np.ndarray, alpha: float, center: bool) -> np.ndarray:
        r = np.asarray(r, dtype=np.float32)
        t_count, n = r.shape
        beta = 1.0 - float(alpha)
        s_mat = np.zeros((n, n), dtype=np.float32)
        w_mat = np.zeros((n, n), dtype=np.float32)
        mu = np.zeros(n, dtype=np.float32)
        block = 1024

        for t in range(t_count):
            row = r[t]
            msk = np.isfinite(row)
            idx = np.flatnonzero(msk)

            s_mat *= beta
            w_mat *= beta
            if idx.size == 0:
                continue

            r_obs = row[idx].astype(np.float32, copy=False)
            if center:
                mu[idx] = beta * mu[idx] + (1.0 - beta) * r_obs
                x_obs = r_obs - mu[idx]
            else:
                x_obs = r_obs

            tw = (1.0 - beta)
            nobs = idx.size
            for a in range(0, nobs, block):
                ia = idx[a:a + block]
                for b in range(a, nobs, block):
                    ib = idx[b:b + block]
                    w_mat[np.ix_(ia, ib)] += tw
                    if b != a:
                        w_mat[np.ix_(ib, ia)] += tw

            for a in range(0, nobs, block):
                ia = idx[a:a + block]
                va = x_obs[a:a + block][:, None]
                for b in range(a, nobs, block):
                    ib = idx[b:b + block]
                    vb = x_obs[b:b + block][None, :]
                    s_mat[np.ix_(ia, ib)] += tw * (va * vb)
                    if b != a:
                        s_mat[np.ix_(ib, ia)] += tw * (vb.T * va.T)

        tiny = 1e-12
        c = s_mat / np.maximum(w_mat, tiny)
        d_all = np.diag(s_mat) / np.maximum(np.diag(w_mat), tiny)
        pos_all = np.isfinite(d_all) & (d_all > 0)
        if not np.any(pos_all):
            raise RuntimeError(
                "EWMA covariance state contains no positive finite variances for diagonal repair."
            )
        med_all = float(np.nanmedian(d_all[pos_all]))
        return self._stabilize_covariance(c, diag_floor=med_all)

    # Compute a stabilized NaN-safe rolling sample covariance on a T×n block.
    def _compute_sample_cov(self, r: np.ndarray, min_periods: int) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        t_count, n = r.shape
        mask = np.isfinite(r)
        cnt = mask.sum(axis=0).astype(np.int64)

        with np.errstate(invalid="ignore"):
            mu = np.where(cnt > 0, np.nansum(np.where(mask, r, 0.0), axis=0) / np.maximum(cnt, 1), 0.0)

        s_mat = np.zeros((n, n), dtype=np.float64)
        w_mat = np.zeros((n, n), dtype=np.float64)
        for t in range(t_count):
            m = mask[t].astype(np.float64)
            if m.sum() == 0:
                continue
            x = np.where(m > 0, r[t] - mu, 0.0)
            mm_t = np.outer(m, m)
            s_mat += np.outer(x, x) * mm_t
            w_mat += mm_t

        denom = np.maximum(w_mat - 1.0, 1.0)
        c = s_mat / denom
        c[w_mat < max(int(min_periods), 2)] = np.nan
        return self._stabilize_covariance(c)

    # Return a heuristic shrinkage intensity based on effective sample size and dimension.
    def _auto_lambda(self, t_eff: int, n: int) -> float:
        if t_eff <= 0:
            return 0.5
        return float(np.clip(n / float(t_eff), 0.0, 0.99))

    # Apply diagonal shrinkage and then add the configured ridge.
    def _apply_shrinkage(self, sigma: np.ndarray, lam: Optional[float]) -> np.ndarray:
        n = sigma.shape[0]
        ident = np.eye(n, dtype=np.asarray(sigma).dtype)
        if lam is None:
            return np.asarray(sigma, dtype=np.float32) + self.mw_ridge * ident
        lam = float(np.clip(lam, 0.0, 1.0))
        diag = np.diag(np.diag(sigma))
        return ((1.0 - lam) * sigma + lam * diag + self.mw_ridge * ident).astype(np.float32, copy=False)

    # Project raw weights into the dollar-neutral subspace using a precomputed inverse.
    def _project_to_dollar_neutral(self, inv: np.ndarray, ones: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
        denom = float(ones.T @ inv @ ones)
        c = 0.0 if abs(denom) < 1e-12 else float((ones.T @ w_vec) / denom)
        return w_vec - c * (inv @ ones)
