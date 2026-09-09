
"""Portfolio diagnostics helpers used by PortfolioManager and transaction-cost strategies.

This module centralizes diagnostics-only logic so reporting can evolve without
changing portfolio construction, covariance estimation, or transaction-cost
optimization behavior.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cost_aware_portfolio.state import RebalanceResult
from cost_aware_portfolio.transaction_costs import TC_ADV_DAILY_COL, TC_SIGMA_DAILY_COL


class PortfolioDiagnostics:
    """Diagnostics-only helpers for portfolio runs and TC debug output."""

    # Return the diagnostic spread series in bps for one sleeve frame.
    @staticmethod
    def tc_diag_spread_bps(*, frame: pd.DataFrame) -> pd.Series:
        if "spread_oneway" in frame.columns:
            return 1e4 * pd.to_numeric(frame["spread_oneway"], errors="coerce")
        if "bid_ask_bps" in frame.columns:
            return pd.to_numeric(frame["bid_ask_bps"], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype=float)

    # Return diagnostic ADV in millions. Historical ADV is retained only for legacy/shadow
    # diagnostics; active nonlinear optimization is required to use TC_ADV_DAILY_COL.
    @staticmethod
    def tc_diag_adv_musd(*, ctx, frame: pd.DataFrame) -> pd.Series:
        adv_col = TC_ADV_DAILY_COL if bool(getattr(ctx, "tc_use_forecast_inputs", False)) else "DollarVol_20d"
        if adv_col not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return 1e-6 * pd.to_numeric(frame[adv_col], errors="coerce")

    # Return diagnostic sigma. Historical mw_var is retained only for legacy/shadow
    # diagnostics; active nonlinear optimization is required to use TC_SIGMA_DAILY_COL.
    @staticmethod
    def tc_diag_sigma(*, ctx, frame: pd.DataFrame) -> pd.Series:
        sigma_col = TC_SIGMA_DAILY_COL if bool(getattr(ctx, "tc_use_forecast_inputs", False)) else "mw_var"
        if sigma_col not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        sigma = pd.to_numeric(frame[sigma_col], errors="coerce")
        if sigma_col == "mw_var":
            sigma = np.sqrt(sigma.clip(lower=0.0))
        return sigma.astype(float)

    # Return the diagnostic market-cap series in billions of dollars for one sleeve frame.
    @staticmethod
    def diag_mcap_busd(*, frame: pd.DataFrame) -> pd.Series:
        if "MarketCap" not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return 1e-9 * pd.to_numeric(frame["MarketCap"], errors="coerce").abs()

    # Return the diagnostic log-market-cap series for one sleeve frame.
    @staticmethod
    def diag_log_mcap(*, frame: pd.DataFrame) -> pd.Series:
        if "MarketCap" not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        mcap = pd.to_numeric(frame["MarketCap"], errors="coerce").abs()
        mcap = mcap.where(mcap > 0.0, np.nan)
        return np.log(mcap)

    # Return the first matching column from a frame, case-insensitively.
    @staticmethod
    def _find_first_col(frame: pd.DataFrame, names: Tuple[str, ...]) -> Optional[str]:
        low_map = {str(c).lower(): c for c in frame.columns}
        for name in names:
            hit = low_map.get(str(name).lower())
            if hit is not None:
                return hit
        return None

    # Return a clean sector label series for one sleeve frame.
    @staticmethod
    def diag_sector_labels(*, frame: pd.DataFrame) -> pd.Series:
        sector_name_col = PortfolioDiagnostics._find_first_col(frame, ("sector_name",))
        if sector_name_col is not None:
            raw = frame[sector_name_col]
            labels = raw.astype("object").where(pd.notna(raw), "Unknown").astype(str)
            labels = labels.where(labels.str.len() > 0, "Unknown")
            return pd.Series(labels, index=frame.index, dtype=object)

        sector_id_col = PortfolioDiagnostics._find_first_col(frame, ("sector_id",))
        if sector_id_col is not None:
            vals = pd.to_numeric(frame[sector_id_col], errors="coerce")
            labels = vals.map(lambda x: f"S{int(x)}" if pd.notna(x) else "Unknown")
            return pd.Series(labels, index=frame.index, dtype=object)

        return pd.Series("Unknown", index=frame.index, dtype=object)

    # Return the share of sleeve weight with a known sector label.
    @staticmethod
    def sector_known_weight_share(*, frame: pd.DataFrame, weights_abs: pd.Series) -> float:
        labels = PortfolioDiagnostics.diag_sector_labels(frame=frame)
        return PortfolioDiagnostics.weighted_share(labels != "Unknown", weights_abs)

    # Compute a NaN-safe mean over the selected subset of a series.
    @staticmethod
    def tc_masked_mean(series: pd.Series, mask: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        m = pd.Series(mask, index=series.index).fillna(False).astype(bool)
        x = s[m].dropna()
        return float(x.mean()) if len(x) else float("nan")

    # Compute a NaN-safe median over the selected subset of a series.
    @staticmethod
    def masked_median(series: pd.Series, mask: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        m = pd.Series(mask, index=series.index).fillna(False).astype(bool)
        x = s[m].dropna()
        return float(x.median()) if len(x) else float("nan")

    # Compute a NaN-safe percentile over the selected subset of a series.
    @staticmethod
    def masked_percentile(series: pd.Series, mask: pd.Series, q: float) -> float:
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        m = pd.Series(mask, index=series.index).fillna(False).astype(bool)
        x = s[m].dropna().to_numpy(dtype=float)
        if x.size == 0:
            return float("nan")
        return float(np.percentile(x, q))

    # Compute a weight-based share of names satisfying a Boolean condition.
    @staticmethod
    def weighted_share(condition: pd.Series, weights: pd.Series) -> float:
        cond = pd.Series(condition).fillna(False).astype(bool)
        ww = pd.to_numeric(weights, errors="coerce").fillna(0.0).abs()
        ww = ww.reindex(cond.index).fillna(0.0)
        total = float(ww.sum())
        if total <= 0 or not np.isfinite(total):
            return float("nan")
        return float(ww[cond].sum() / total)

    # Compute the effective number of names from absolute sleeve weights.
    @staticmethod
    def effective_n(weights: pd.Series) -> float:
        ww = pd.to_numeric(weights, errors="coerce").fillna(0.0).abs()
        gross = float(ww.sum())
        if gross <= 0 or not np.isfinite(gross):
            return float("nan")
        wn = ww / gross
        denom = float(np.square(wn).sum())
        if denom <= 0 or not np.isfinite(denom):
            return float("nan")
        return float(1.0 / denom)

    # Return the share held in the top-k names of a sleeve.
    @staticmethod
    def top_k_weight_share(weights: pd.Series, k: int) -> float:
        ww = pd.to_numeric(weights, errors="coerce").fillna(0.0).abs().sort_values(ascending=False)
        gross = float(ww.sum())
        if gross <= 0 or not np.isfinite(gross):
            return float("nan")
        return float(ww.head(int(k)).sum() / gross)

    # Return the maximum absolute weight in a sleeve.
    @staticmethod
    def max_weight(weights: pd.Series) -> float:
        ww = pd.to_numeric(weights, errors="coerce").fillna(0.0).abs()
        return float(ww.max()) if len(ww) else float("nan")

    # Return the median active absolute weight in a sleeve.
    @staticmethod
    def median_active_weight(weights: pd.Series) -> float:
        ww = pd.to_numeric(weights, errors="coerce").fillna(0.0).abs()
        ww = ww[ww > 0.0]
        return float(ww.median()) if len(ww) else float("nan")

    # Return the 95th percentile active absolute weight in a sleeve.
    @staticmethod
    def p95_active_weight(weights: pd.Series) -> float:
        ww = pd.to_numeric(weights, errors="coerce").fillna(0.0).abs()
        ww = ww[ww > 0.0]
        return float(np.percentile(ww.to_numpy(dtype=float), 95.0)) if len(ww) else float("nan")

    # Build a normalized sector-weight map from sleeve weights, excluding unknown labels.
    @staticmethod
    def sector_weight_map(*, frame: pd.DataFrame, weights_abs: pd.Series) -> Dict[str, float]:
        if len(frame) == 0:
            return {}
        ww = pd.to_numeric(weights_abs.reindex(frame.index), errors="coerce").fillna(0.0).abs()
        labels = PortfolioDiagnostics.diag_sector_labels(frame=frame)
        known_mask = labels != "Unknown"
        ww = ww[known_mask]
        labels = labels[known_mask]
        gross = float(ww.sum())
        if gross <= 0 or not np.isfinite(gross):
            return {}
        grp = ww.groupby(labels).sum() / gross
        grp = grp.sort_values(ascending=False)
        return {str(k): float(v) for k, v in grp.items() if float(v) > 0.0}

    # Compute the HHI from a sector-weight map.
    @staticmethod
    def sector_hhi(weight_map: Dict[str, float]) -> float:
        if not weight_map:
            return float("nan")
        vals = np.asarray(list(weight_map.values()), dtype=float)
        return float(np.square(vals).sum())

    # Compute the dominant sector share from a sector-weight map.
    @staticmethod
    def dominant_sector_share(weight_map: Dict[str, float]) -> float:
        if not weight_map:
            return float("nan")
        return float(max(weight_map.values()))

    # Compute the top-k overlap between two weighted sleeves.
    @staticmethod
    def top_k_overlap(cur: pd.Series, prev: pd.Series, k: int) -> float:
        cur_abs = pd.to_numeric(cur, errors="coerce").fillna(0.0).abs().sort_values(ascending=False)
        prev_abs = pd.to_numeric(prev, errors="coerce").fillna(0.0).abs().sort_values(ascending=False)
        cur_ids = set(cur_abs[cur_abs > 0.0].head(int(k)).index.astype(str))
        prev_ids = set(prev_abs[prev_abs > 0.0].head(int(k)).index.astype(str))
        denom = max(len(cur_ids), len(prev_ids))
        if denom == 0:
            return float("nan")
        return float(len(cur_ids & prev_ids) / denom)

    # Compute the Jaccard overlap of active names in two sleeves.
    @staticmethod
    def active_jaccard(cur: pd.Series, prev: pd.Series) -> float:
        cur_ids = set(pd.Index(pd.to_numeric(cur, errors="coerce").fillna(0.0).loc[lambda s: s.abs() > 0.0].index).astype(str))
        prev_ids = set(pd.Index(pd.to_numeric(prev, errors="coerce").fillna(0.0).loc[lambda s: s.abs() > 0.0].index).astype(str))
        denom = len(cur_ids | prev_ids)
        if denom == 0:
            return float("nan")
        return float(len(cur_ids & prev_ids) / denom)

    # Build the generic tradable mask used by shadow-cost diagnostics for all methods.
    @staticmethod
    def generic_tradable_mask(*, ctx, frame: pd.DataFrame) -> pd.Series:
        idx = frame.index
        mask = pd.Series(True, index=idx, dtype=bool)

        adv_col = TC_ADV_DAILY_COL if bool(getattr(ctx, "tc_use_forecast_inputs", False)) else "DollarVol_20d"
        if adv_col not in frame.columns:
            return pd.Series(False, index=idx, dtype=bool)
        adv = pd.to_numeric(frame[adv_col], errors="coerce")
        mask &= adv.notna() & np.isfinite(adv) & (adv > 0.0)

        sigma_col = TC_SIGMA_DAILY_COL if bool(getattr(ctx, "tc_use_forecast_inputs", False)) else "mw_var"
        if sigma_col not in frame.columns:
            return pd.Series(False, index=idx, dtype=bool)
        sigma_raw = pd.to_numeric(frame[sigma_col], errors="coerce")
        mask &= sigma_raw.notna() & np.isfinite(sigma_raw) & (sigma_raw >= 0.0)

        if "spread_oneway" in frame.columns:
            spread = pd.to_numeric(frame["spread_oneway"], errors="coerce")
            mask &= spread.notna() & np.isfinite(spread) & (spread >= 0.0)
        elif "bid_ask_bps" in frame.columns:
            spread_bps = pd.to_numeric(frame["bid_ask_bps"], errors="coerce")
            mask &= spread_bps.notna() & np.isfinite(spread_bps) & (spread_bps >= 0.0)
        else:
            return pd.Series(False, index=idx, dtype=bool)

        if "Price" in frame.columns:
            price = pd.to_numeric(frame["Price"], errors="coerce")
            mask &= price.notna() & np.isfinite(price) & (price > 0.0)

        if "Volume" in frame.columns:
            volume = pd.to_numeric(frame["Volume"], errors="coerce")
            mask &= volume.notna() & np.isfinite(volume) & (volume > 0.0)

        return mask.fillna(False)

    # Build aligned one-way spreads in fraction-of-notional units without failing on missing rows.
    @staticmethod
    def generic_spreads_fraction(*, frame: pd.DataFrame) -> pd.Series:
        if "spread_oneway" in frame.columns:
            return pd.to_numeric(frame["spread_oneway"], errors="coerce")
        if "bid_ask_bps" in frame.columns:
            return 1e-4 * pd.to_numeric(frame["bid_ask_bps"], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype=float)

    # Compute additive ex-post trading-cost series on the tradable subset for any portfolio method.
    @staticmethod
    def generic_trade_cost_series(
        *,
        ctx,
        frame: pd.DataFrame,
        cur_w: pd.Series,
        prev_w: pd.Series,
        aum_override: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        idx = pd.Index(frame.index)
        cur = pd.to_numeric(cur_w.reindex(idx), errors="coerce").fillna(0.0)
        prev = pd.to_numeric(prev_w.reindex(idx), errors="coerce").fillna(0.0)
        trad_mask = PortfolioDiagnostics.generic_tradable_mask(ctx=ctx, frame=frame)

        if not bool(trad_mask.any()):
            zeros = pd.Series(0.0, index=idx, dtype=float)
            return zeros, zeros, zeros, trad_mask

        adv_col = TC_ADV_DAILY_COL if bool(getattr(ctx, "tc_use_forecast_inputs", False)) else "DollarVol_20d"
        sigma_col = TC_SIGMA_DAILY_COL if bool(getattr(ctx, "tc_use_forecast_inputs", False)) else "mw_var"

        spreads = PortfolioDiagnostics.generic_spreads_fraction(frame=frame)
        adv = pd.to_numeric(frame[adv_col], errors="coerce")
        sigma_raw = pd.to_numeric(frame[sigma_col], errors="coerce")
        sigma = sigma_raw if sigma_col == TC_SIGMA_DAILY_COL else np.sqrt(sigma_raw.clip(lower=0.0))
        aum = float(ctx.tc_aum_dollars) if aum_override is None else float(aum_override)

        delta = (cur - prev).abs()
        linear = pd.Series(0.0, index=idx, dtype=float)
        impact = pd.Series(0.0, index=idx, dtype=float)

        tm = trad_mask.fillna(False)
        linear.loc[tm] = (pd.to_numeric(spreads.reindex(idx), errors="coerce").fillna(0.0).loc[tm] * delta.loc[tm]).astype(float)
        execution_days = float(getattr(ctx, "tc_execution_days", 1.0))
        execution_volume = adv.loc[tm].to_numpy(dtype=float) * execution_days
        coeff = ctx.tc_impact_kappa * sigma.loc[tm].to_numpy(dtype=float) * np.sqrt(aum / execution_volume)
        impact.loc[tm] = coeff * np.power(delta.loc[tm].to_numpy(dtype=float), float(ctx.tc_impact_exponent))
        total = (linear + impact).astype(float)
        return linear, impact, total, trad_mask

    # Compute costs from the canonical RebalanceResult trade vector.
    @staticmethod
    def trade_cost_series_from_rebalance_result(
        *,
        ctx,
        frame: pd.DataFrame,
        rebalance_result: RebalanceResult,
        aum_override: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        idx = pd.Index(rebalance_result.delta_w.index).astype(str)
        use_frame = frame.copy()
        use_frame.index = pd.Index(use_frame.index).astype(str)
        use_frame = use_frame.reindex(idx)
        trad_mask = PortfolioDiagnostics.generic_tradable_mask(ctx=ctx, frame=use_frame)

        zeros = pd.Series(0.0, index=idx, dtype=float)
        if not bool(trad_mask.any()):
            return zeros.copy(), zeros.copy(), zeros.copy(), trad_mask

        adv_col = TC_ADV_DAILY_COL if bool(getattr(ctx, "tc_use_forecast_inputs", False)) else "DollarVol_20d"
        sigma_col = TC_SIGMA_DAILY_COL if bool(getattr(ctx, "tc_use_forecast_inputs", False)) else "mw_var"
        spreads = PortfolioDiagnostics.generic_spreads_fraction(frame=use_frame).reindex(idx)
        adv = pd.to_numeric(use_frame[adv_col], errors="coerce").reindex(idx)
        sigma_raw = pd.to_numeric(use_frame[sigma_col], errors="coerce").reindex(idx)
        sigma = sigma_raw if sigma_col == TC_SIGMA_DAILY_COL else np.sqrt(sigma_raw.clip(lower=0.0))
        aum = float(rebalance_result.aum) if aum_override is None else float(aum_override)

        # This is the only trade-size definition used in the canonical TC path.
        delta = pd.to_numeric(rebalance_result.delta_w.reindex(idx), errors="coerce").fillna(0.0).abs()
        linear = zeros.copy()
        impact = zeros.copy()
        tm = trad_mask.fillna(False)
        linear.loc[tm] = (spreads.fillna(0.0).loc[tm] * delta.loc[tm]).astype(float)
        execution_days = float(getattr(ctx, "tc_execution_days", 1.0))
        execution_volume = adv.loc[tm].to_numpy(dtype=float) * execution_days
        coeff = (
            float(ctx.tc_impact_kappa)
            * sigma.loc[tm].to_numpy(dtype=float)
            * np.sqrt(aum / execution_volume)
        )
        impact.loc[tm] = coeff * np.power(
            delta.loc[tm].to_numpy(dtype=float),
            float(ctx.tc_impact_exponent),
        )
        total = (linear + impact).astype(float)
        return linear, impact, total, trad_mask

    # Decompose turnover directly from the canonical pre/post trade transition.
    @staticmethod
    def turnover_decomposition_from_rebalance_result(
        rebalance_result: RebalanceResult,
    ) -> Dict[str, float]:
        idx = pd.Index(rebalance_result.delta_w.index).astype(str)
        pre = pd.to_numeric(rebalance_result.pretrade_weights.reindex(idx), errors="coerce").fillna(0.0)
        post = pd.to_numeric(rebalance_result.posttrade_weights.reindex(idx), errors="coerce").fillna(0.0)
        delta_abs = pd.to_numeric(rebalance_result.delta_w.reindex(idx), errors="coerce").fillna(0.0).abs()
        pre_abs = pre.abs()
        post_abs = post.abs()

        enter_mask = (post_abs > 0.0) & (pre_abs <= 0.0)
        exit_mask = (post_abs <= 0.0) & (pre_abs > 0.0)
        resize_mask = (post_abs > 0.0) & (pre_abs > 0.0)
        entry_to = 0.5 * float(delta_abs[enter_mask].sum())
        exit_to = 0.5 * float(delta_abs[exit_mask].sum())
        resize_to = 0.5 * float(delta_abs[resize_mask].sum())
        return {
            "total_turnover": float(rebalance_result.turnover_oneway),
            "entry_turnover": entry_to,
            "exit_turnover": exit_to,
            "resize_turnover": resize_to,
        }

    # Drift prior signed weights using realized returns and normalize to current gross.
    @staticmethod
    def drifted_prev_weights(*, cur_w: pd.Series, prev_w: pd.Series, prev_rets: pd.Series) -> pd.Series:
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
        return drifted.astype(float)

    # Decompose one-way turnover into entry/exit and resize components.
    @staticmethod
    def turnover_decomposition(*, cur_w: pd.Series, prev_w: pd.Series, prev_rets: pd.Series) -> Dict[str, float]:
        idx = pd.Index(np.unique(list(pd.Index(cur_w.index).astype(str)) + list(pd.Index(prev_w.index).astype(str))))
        cur = pd.to_numeric(cur_w.reindex(idx), errors="coerce").fillna(0.0).astype(float)
        prev = pd.to_numeric(prev_w.reindex(idx), errors="coerce").fillna(0.0).astype(float)
        rets = pd.to_numeric(prev_rets.reindex(idx), errors="coerce").fillna(0.0).astype(float)
        drifted = PortfolioDiagnostics.drifted_prev_weights(cur_w=cur, prev_w=prev, prev_rets=rets)

        cur_abs = cur.abs()
        drift_abs = drifted.abs()
        diff_abs = (cur - drifted).abs()

        enter_mask = (cur_abs > 0.0) & (drift_abs <= 0.0)
        exit_mask = (cur_abs <= 0.0) & (drift_abs > 0.0)
        common_mask = (cur_abs > 0.0) & (drift_abs > 0.0)

        entry_to = 0.5 * float(diff_abs[enter_mask].sum())
        exit_to = 0.5 * float(diff_abs[exit_mask].sum())
        resize_to = 0.5 * float(diff_abs[common_mask].sum())
        total_to = entry_to + exit_to + resize_to

        return {
            "total_turnover": float(total_to),
            "entry_turnover": float(entry_to),
            "exit_turnover": float(exit_to),
            "resize_turnover": float(resize_to),
        }

    # Append a paired top/bottom sleeve diagnostic row without changing portfolio logic.
    @staticmethod
    def append_paired_tc_debug_row(
        *,
        ctx,
        prev_state: Dict,
        date: pd.Timestamp,
        common_target_sum: Optional[float],
        low_frame: pd.DataFrame,
        high_frame: pd.DataFrame,
        low_weights_abs: pd.Series,
        high_weights: pd.Series,
        low_cost: float,
        high_cost: float,
        low_trad_mask: pd.Series,
        high_trad_mask: pd.Series,
    ) -> None:
        debug_rows = prev_state.setdefault("tc_debug_rows", [])
        low_abs = pd.to_numeric(low_weights_abs.reindex(low_frame.index), errors="coerce").fillna(0.0).abs()
        high_abs = pd.to_numeric(high_weights.reindex(high_frame.index), errors="coerce").fillna(0.0).abs()
        low_gross = float(low_abs.sum())
        high_gross = float(high_abs.sum())
        target_sum = float(common_target_sum) if common_target_sum is not None else float("nan")

        # Each paired sleeve is a long-only sub-portfolio whose requested full
        # investment is one unit of NAV.  Cash/unfilled reporting therefore has
        # a direct accounting interpretation even though the final H-L book is
        # represented as a signed portfolio.
        low_cash = 1.0 - low_gross
        high_cash = 1.0 - high_gross
        low_unfilled = max(0.0, 1.0 - low_gross)
        high_unfilled = max(0.0, 1.0 - high_gross)
        mean_unfilled = 0.5 * (low_unfilled + high_unfilled)
        capacity_target_shortfall = max(0.0, 1.0 - target_sum) if np.isfinite(target_sum) else float("nan")
        low_execution_shortfall = max(0.0, target_sum - low_gross) if np.isfinite(target_sum) else float("nan")
        high_execution_shortfall = max(0.0, target_sum - high_gross) if np.isfinite(target_sum) else float("nan")

        debug_rows.append({
            "date": pd.Timestamp(date),
            "execution_days": float(getattr(ctx, "tc_execution_days", 1.0)),
            "unmatched_feasibility": bool(common_target_sum is None),
            "common_target_sum": target_sum,
            "capacity_target_shortfall": capacity_target_shortfall,
            "low_realized_gross": low_gross,
            "high_realized_gross": high_gross,
            "low_posttrade_cash_weight": low_cash,
            "high_posttrade_cash_weight": high_cash,
            "low_unfilled_target_share": low_unfilled,
            "high_unfilled_target_share": high_unfilled,
            "mean_unfilled_target_share": mean_unfilled,
            "low_execution_shortfall_vs_common_target": low_execution_shortfall,
            "high_execution_shortfall_vs_common_target": high_execution_shortfall,
            "low_cost": float(low_cost),
            "high_cost": float(high_cost),
            "total_cost": float(low_cost) + float(high_cost),
            "low_total_names": int(len(low_frame)),
            "low_tradable_names": int(pd.Series(low_trad_mask, index=low_frame.index).fillna(False).sum()),
            "high_total_names": int(len(high_frame)),
            "high_tradable_names": int(pd.Series(high_trad_mask, index=high_frame.index).fillna(False).sum()),
            "low_avg_spread_bps": PortfolioDiagnostics.tc_masked_mean(
                PortfolioDiagnostics.tc_diag_spread_bps(frame=low_frame),
                low_trad_mask,
            ),
            "high_avg_spread_bps": PortfolioDiagnostics.tc_masked_mean(
                PortfolioDiagnostics.tc_diag_spread_bps(frame=high_frame),
                high_trad_mask,
            ),
            "low_avg_adv_musd": PortfolioDiagnostics.tc_masked_mean(
                PortfolioDiagnostics.tc_diag_adv_musd(ctx=ctx, frame=low_frame),
                low_trad_mask,
            ),
            "high_avg_adv_musd": PortfolioDiagnostics.tc_masked_mean(
                PortfolioDiagnostics.tc_diag_adv_musd(ctx=ctx, frame=high_frame),
                high_trad_mask,
            ),
            "low_avg_sigma": PortfolioDiagnostics.tc_masked_mean(
                PortfolioDiagnostics.tc_diag_sigma(ctx=ctx, frame=low_frame),
                low_trad_mask,
            ),
            "high_avg_sigma": PortfolioDiagnostics.tc_masked_mean(
                PortfolioDiagnostics.tc_diag_sigma(ctx=ctx, frame=high_frame),
                high_trad_mask,
            ),
        })

    # Append all-method shadow-cost and sleeve-composition diagnostics for one rebalance date.
    @staticmethod
    def append_portfolio_debug_row(
        *,
        ctx,
        prev_state: Dict,
        date: pd.Timestamp,
        reb_frame: pd.DataFrame,
        to_df: pd.DataFrame,
        prev_to_df: Optional[pd.DataFrame],
        ret_name: str,
        rebalance_result: Optional[RebalanceResult] = None,
    ) -> None:
        debug_rows = prev_state.setdefault("portfolio_debug_rows", [])
        if to_df is None or len(to_df) == 0:
            debug_rows.append({"date": pd.Timestamp(date)})
            return

        if rebalance_result is not None:
            union_idx = pd.Index(rebalance_result.delta_w.index).astype(str)
            union_frame = reb_frame.copy()
            union_frame.index = pd.Index(union_frame.index).astype(str)
            union_frame = union_frame.reindex(union_idx).copy()
            cur_union = pd.to_numeric(rebalance_result.posttrade_weights.reindex(union_idx), errors="coerce").fillna(0.0)
            prev_union = pd.to_numeric(rebalance_result.pretrade_weights.reindex(union_idx), errors="coerce").fillna(0.0)
            prev_rets = pd.Series(0.0, index=union_idx, dtype=float)
            linear_cost, impact_cost, total_cost, trad_mask_union = PortfolioDiagnostics.trade_cost_series_from_rebalance_result(
                ctx=ctx, frame=union_frame, rebalance_result=rebalance_result
            )
            _, _, total_cost_2x, _ = PortfolioDiagnostics.trade_cost_series_from_rebalance_result(
                ctx=ctx, frame=union_frame, rebalance_result=rebalance_result,
                aum_override=float(rebalance_result.aum) * 2.0,
            )
            _, _, total_cost_5x, _ = PortfolioDiagnostics.trade_cost_series_from_rebalance_result(
                ctx=ctx, frame=union_frame, rebalance_result=rebalance_result,
                aum_override=float(rebalance_result.aum) * 5.0,
            )
        else:
            cur_signed = pd.to_numeric(to_df.get("weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            prev_signed = pd.Series(dtype=float)
            prev_rets_raw = pd.Series(dtype=float)
            if prev_to_df is not None and len(prev_to_df) > 0:
                prev_signed = pd.to_numeric(prev_to_df.get("weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                prev_rets_raw = pd.to_numeric(prev_to_df.get(ret_name, pd.Series(dtype=float)), errors="coerce").fillna(0.0)

            union_idx = pd.Index(np.unique(list(cur_signed.index.astype(str)) + list(prev_signed.index.astype(str))))
            union_frame = reb_frame.reindex(union_idx).copy()
            cur_union = pd.to_numeric(cur_signed.reindex(union_idx), errors="coerce").fillna(0.0)
            prev_union = pd.to_numeric(prev_signed.reindex(union_idx), errors="coerce").fillna(0.0)
            prev_rets = pd.to_numeric(prev_rets_raw.reindex(union_idx), errors="coerce").fillna(0.0)
            linear_cost, impact_cost, total_cost, trad_mask_union = PortfolioDiagnostics.generic_trade_cost_series(
                ctx=ctx, frame=union_frame, cur_w=cur_union, prev_w=prev_union
            )
            _, _, total_cost_2x, _ = PortfolioDiagnostics.generic_trade_cost_series(
                ctx=ctx, frame=union_frame, cur_w=cur_union, prev_w=prev_union,
                aum_override=float(ctx.tc_aum_dollars) * 2.0,
            )
            _, _, total_cost_5x, _ = PortfolioDiagnostics.generic_trade_cost_series(
                ctx=ctx, frame=union_frame, cur_w=cur_union, prev_w=prev_union,
                aum_override=float(ctx.tc_aum_dollars) * 5.0,
            )

        low_cost_attr = (cur_union < 0.0) | ((cur_union == 0.0) & (prev_union < 0.0))
        high_cost_attr = (cur_union > 0.0) | ((cur_union == 0.0) & (prev_union > 0.0))
        low_ids = cur_union.index[cur_union < 0.0]
        high_ids = cur_union.index[cur_union > 0.0]
        low_frame = reb_frame.reindex(low_ids).copy()
        high_frame = reb_frame.reindex(high_ids).copy()
        low_weights_abs = cur_union.reindex(low_ids).fillna(0.0).abs()
        high_weights_abs = cur_union.reindex(high_ids).fillna(0.0).abs()
        low_trad_mask = PortfolioDiagnostics.generic_tradable_mask(ctx=ctx, frame=low_frame) if len(low_frame) else pd.Series(False, index=low_frame.index, dtype=bool)
        high_trad_mask = PortfolioDiagnostics.generic_tradable_mask(ctx=ctx, frame=high_frame) if len(high_frame) else pd.Series(False, index=high_frame.index, dtype=bool)
        low_mcap = PortfolioDiagnostics.diag_mcap_busd(frame=low_frame)
        high_mcap = PortfolioDiagnostics.diag_mcap_busd(frame=high_frame)
        low_log_mcap = PortfolioDiagnostics.diag_log_mcap(frame=low_frame)
        high_log_mcap = PortfolioDiagnostics.diag_log_mcap(frame=high_frame)

        low_sector_known_weight_share = PortfolioDiagnostics.sector_known_weight_share(frame=low_frame, weights_abs=low_weights_abs)
        high_sector_known_weight_share = PortfolioDiagnostics.sector_known_weight_share(frame=high_frame, weights_abs=high_weights_abs)
        low_sector_map = PortfolioDiagnostics.sector_weight_map(frame=low_frame, weights_abs=low_weights_abs)
        high_sector_map = PortfolioDiagnostics.sector_weight_map(frame=high_frame, weights_abs=high_weights_abs)
        sector_keys = sorted(set(low_sector_map.keys()) | set(high_sector_map.keys()))
        net_sector_map = {k: float(high_sector_map.get(k, 0.0) - low_sector_map.get(k, 0.0)) for k in sector_keys}

        if rebalance_result is not None:
            drifted_union = prev_union.copy()
            to_decomp = PortfolioDiagnostics.turnover_decomposition_from_rebalance_result(rebalance_result)
        else:
            drifted_union = PortfolioDiagnostics.drifted_prev_weights(cur_w=cur_union, prev_w=prev_union, prev_rets=prev_rets)
            to_decomp = PortfolioDiagnostics.turnover_decomposition(cur_w=cur_union, prev_w=prev_union, prev_rets=prev_rets)

        low_scope = pd.Index(np.unique(list(pd.Index(cur_union[cur_union < 0.0].index).astype(str)) + list(pd.Index(prev_union[prev_union < 0.0].index).astype(str))))
        high_scope = pd.Index(np.unique(list(pd.Index(cur_union[cur_union > 0.0].index).astype(str)) + list(pd.Index(prev_union[prev_union > 0.0].index).astype(str))))
        low_turnover = float("nan")
        high_turnover = float("nan")
        if len(low_scope) > 0:
            low_turnover = 0.5 * float((cur_union.reindex(low_scope).fillna(0.0) - drifted_union.reindex(low_scope).fillna(0.0)).abs().sum())
        if len(high_scope) > 0:
            high_turnover = 0.5 * float((cur_union.reindex(high_scope).fillna(0.0) - drifted_union.reindex(high_scope).fillna(0.0)).abs().sum())

        low_prev = prev_union.reindex(low_ids).fillna(0.0).abs()
        high_prev = prev_union.reindex(high_ids).fillna(0.0).clip(lower=0.0)
        low_jaccard = PortfolioDiagnostics.active_jaccard(low_weights_abs, low_prev)
        high_jaccard = PortfolioDiagnostics.active_jaccard(high_weights_abs, high_prev)
        low_top10_overlap = PortfolioDiagnostics.top_k_overlap(low_weights_abs, low_prev, 10)
        high_top10_overlap = PortfolioDiagnostics.top_k_overlap(high_weights_abs, high_prev, 10)

        low_cur_ids = set(pd.Index(low_weights_abs[low_weights_abs > 0.0].index).astype(str))
        low_prev_ids = set(pd.Index(low_prev[low_prev > 0.0].index).astype(str))
        high_cur_ids = set(pd.Index(high_weights_abs[high_weights_abs > 0.0].index).astype(str))
        high_prev_ids = set(pd.Index(high_prev[high_prev > 0.0].index).astype(str))
        low_replacement_frac = float(1.0 - len(low_cur_ids & low_prev_ids) / max(len(low_cur_ids), 1)) if len(low_cur_ids) else float("nan")
        high_replacement_frac = float(1.0 - len(high_cur_ids & high_prev_ids) / max(len(high_cur_ids), 1)) if len(high_cur_ids) else float("nan")

        adv_col = TC_ADV_DAILY_COL if bool(getattr(ctx, "tc_use_forecast_inputs", False)) else "DollarVol_20d"
        adv_union = pd.to_numeric(union_frame.get(adv_col, pd.Series(index=union_idx, dtype=float)), errors="coerce").reindex(union_idx)
        delta_union = (
            pd.to_numeric(rebalance_result.delta_w.reindex(union_idx), errors="coerce").fillna(0.0).abs()
            if rebalance_result is not None
            else (cur_union - prev_union).abs()
        )
        # Position capacity remains measured against one day of ADV.  Only the
        # scheduled rebalance trade cap scales with the explicit execution window.
        pos_cap = float(ctx.tc_pos_adv_cap) * adv_union / float(ctx.tc_aum_dollars)
        execution_days = float(getattr(ctx, "tc_execution_days", 1.0))
        trade_cap = (
            float(ctx.tc_trade_adv_cap)
            * adv_union
            * execution_days
            / float(ctx.tc_aum_dollars)
        )
        pos_bind_mask = trad_mask_union & (cur_union.abs() > pos_cap)
        trade_bind_mask = trad_mask_union & (delta_union > trade_cap)

        debug_rows.append({
            "date": pd.Timestamp(date),
            "execution_days": float(getattr(ctx, "tc_execution_days", 1.0)),
            "pretrade_cash_weight": (
                float(rebalance_result.pretrade_cash_weight)
                if rebalance_result is not None else float("nan")
            ),
            "posttrade_cash_weight": (
                float(rebalance_result.posttrade_cash_weight)
                if rebalance_result is not None else float("nan")
            ),
            "pretrade_net_exposure": (
                float(rebalance_result.net_pretrade)
                if rebalance_result is not None else float("nan")
            ),
            "posttrade_net_exposure": (
                float(rebalance_result.net_posttrade)
                if rebalance_result is not None else float("nan")
            ),
            "pretrade_gross_exposure": (
                float(rebalance_result.gross_pretrade)
                if rebalance_result is not None else float("nan")
            ),
            "posttrade_gross_exposure": (
                float(rebalance_result.gross_posttrade)
                if rebalance_result is not None else float("nan")
            ),
            "canonical_turnover_oneway": (
                float(rebalance_result.turnover_oneway)
                if rebalance_result is not None else float("nan")
            ),
            "canonical_oneway_dollar_turnover": (
                float(rebalance_result.one_way_dollar_turnover)
                if rebalance_result is not None else float("nan")
            ),
            "canonical_total_abs_dollar_trades": (
                float(rebalance_result.total_absolute_dollar_trades)
                if rebalance_result is not None else float("nan")
            ),
            "low_realized_gross": float(low_weights_abs.sum()),
            "high_realized_gross": float(high_weights_abs.sum()),
            "linear_cost": float(linear_cost.sum()),
            "impact_cost": float(impact_cost.sum()),
            "total_cost": float(total_cost.sum()),
            "total_cost_2x": float(total_cost_2x.sum()),
            "total_cost_5x": float(total_cost_5x.sum()),
            "low_cost": float(total_cost[low_cost_attr].sum()),
            "high_cost": float(total_cost[high_cost_attr].sum()),
            "cost_attr_residual": float(total_cost.sum() - total_cost[low_cost_attr].sum() - total_cost[high_cost_attr].sum()),
            "low_total_names": int(len(low_frame)),
            "high_total_names": int(len(high_frame)),
            "low_tradable_names": int(low_trad_mask.sum()) if len(low_frame) else 0,
            "high_tradable_names": int(high_trad_mask.sum()) if len(high_frame) else 0,
            "low_avg_spread_bps": PortfolioDiagnostics.tc_masked_mean(PortfolioDiagnostics.tc_diag_spread_bps(frame=low_frame), low_trad_mask),
            "high_avg_spread_bps": PortfolioDiagnostics.tc_masked_mean(PortfolioDiagnostics.tc_diag_spread_bps(frame=high_frame), high_trad_mask),
            "low_avg_adv_musd": PortfolioDiagnostics.tc_masked_mean(PortfolioDiagnostics.tc_diag_adv_musd(ctx=ctx, frame=low_frame), low_trad_mask),
            "high_avg_adv_musd": PortfolioDiagnostics.tc_masked_mean(PortfolioDiagnostics.tc_diag_adv_musd(ctx=ctx, frame=high_frame), high_trad_mask),
            "low_avg_sigma": PortfolioDiagnostics.tc_masked_mean(PortfolioDiagnostics.tc_diag_sigma(ctx=ctx, frame=low_frame), low_trad_mask),
            "high_avg_sigma": PortfolioDiagnostics.tc_masked_mean(PortfolioDiagnostics.tc_diag_sigma(ctx=ctx, frame=high_frame), high_trad_mask),
            "low_avg_mcap_busd": PortfolioDiagnostics.tc_masked_mean(low_mcap, low_weights_abs > 0.0),
            "high_avg_mcap_busd": PortfolioDiagnostics.tc_masked_mean(high_mcap, high_weights_abs > 0.0),
            "low_median_mcap_busd": PortfolioDiagnostics.masked_median(low_mcap, low_weights_abs > 0.0),
            "high_median_mcap_busd": PortfolioDiagnostics.masked_median(high_mcap, high_weights_abs > 0.0),
            "low_p10_mcap_busd": PortfolioDiagnostics.masked_percentile(low_mcap, low_weights_abs > 0.0, 10.0),
            "high_p10_mcap_busd": PortfolioDiagnostics.masked_percentile(high_mcap, high_weights_abs > 0.0, 10.0),
            "low_p90_mcap_busd": PortfolioDiagnostics.masked_percentile(low_mcap, low_weights_abs > 0.0, 90.0),
            "high_p90_mcap_busd": PortfolioDiagnostics.masked_percentile(high_mcap, high_weights_abs > 0.0, 90.0),
            "low_avg_log_mcap": PortfolioDiagnostics.tc_masked_mean(low_log_mcap, low_weights_abs > 0.0),
            "high_avg_log_mcap": PortfolioDiagnostics.tc_masked_mean(high_log_mcap, high_weights_abs > 0.0),
            "low_micro_weight_share": PortfolioDiagnostics.weighted_share(low_mcap < 0.3, low_weights_abs),
            "high_micro_weight_share": PortfolioDiagnostics.weighted_share(high_mcap < 0.3, high_weights_abs),
            "low_smallcap_weight_share": PortfolioDiagnostics.weighted_share((low_mcap >= 0.3) & (low_mcap < 2.0), low_weights_abs),
            "high_smallcap_weight_share": PortfolioDiagnostics.weighted_share((high_mcap >= 0.3) & (high_mcap < 2.0), high_weights_abs),
            "low_midcap_weight_share": PortfolioDiagnostics.weighted_share((low_mcap >= 2.0) & (low_mcap < 10.0), low_weights_abs),
            "high_midcap_weight_share": PortfolioDiagnostics.weighted_share((high_mcap >= 2.0) & (high_mcap < 10.0), high_weights_abs),
            "low_largecap_weight_share": PortfolioDiagnostics.weighted_share(low_mcap >= 10.0, low_weights_abs),
            "high_largecap_weight_share": PortfolioDiagnostics.weighted_share(high_mcap >= 10.0, high_weights_abs),
            "low_effective_n": PortfolioDiagnostics.effective_n(low_weights_abs),
            "high_effective_n": PortfolioDiagnostics.effective_n(high_weights_abs),
            "low_top5_weight_share": PortfolioDiagnostics.top_k_weight_share(low_weights_abs, 5),
            "high_top5_weight_share": PortfolioDiagnostics.top_k_weight_share(high_weights_abs, 5),
            "low_top10_weight_share": PortfolioDiagnostics.top_k_weight_share(low_weights_abs, 10),
            "high_top10_weight_share": PortfolioDiagnostics.top_k_weight_share(high_weights_abs, 10),
            "low_max_weight": PortfolioDiagnostics.max_weight(low_weights_abs),
            "high_max_weight": PortfolioDiagnostics.max_weight(high_weights_abs),
            "low_median_active_weight": PortfolioDiagnostics.median_active_weight(low_weights_abs),
            "high_median_active_weight": PortfolioDiagnostics.median_active_weight(high_weights_abs),
            "low_p95_active_weight": PortfolioDiagnostics.p95_active_weight(low_weights_abs),
            "high_p95_active_weight": PortfolioDiagnostics.p95_active_weight(high_weights_abs),
            "total_turnover": float(to_decomp["total_turnover"]),
            "entry_turnover": float(to_decomp["entry_turnover"]),
            "exit_turnover": float(to_decomp["exit_turnover"]),
            "resize_turnover": float(to_decomp["resize_turnover"]),
            "low_turnover": float(low_turnover),
            "high_turnover": float(high_turnover),
            "low_jaccard": float(low_jaccard),
            "high_jaccard": float(high_jaccard),
            "low_top10_overlap": float(low_top10_overlap),
            "high_top10_overlap": float(high_top10_overlap),
            "low_replacement_frac": float(low_replacement_frac),
            "high_replacement_frac": float(high_replacement_frac),
            "any_pos_cap_bind": bool(pos_bind_mask.any()),
            "any_trade_cap_bind": bool(trade_bind_mask.any()),
            "pos_cap_bind_weight_share": PortfolioDiagnostics.weighted_share(pos_bind_mask, cur_union.abs()),
            "trade_cap_bind_weight_share": PortfolioDiagnostics.weighted_share(trade_bind_mask, delta_union),
            "low_sector_known_weight_share": float(low_sector_known_weight_share),
            "high_sector_known_weight_share": float(high_sector_known_weight_share),
            "low_sector_map": low_sector_map,
            "high_sector_map": high_sector_map,
            "net_sector_map": net_sector_map,
            "low_sector_hhi": PortfolioDiagnostics.sector_hhi(low_sector_map),
            "high_sector_hhi": PortfolioDiagnostics.sector_hhi(high_sector_map),
            "low_dom_sector_share": PortfolioDiagnostics.dominant_sector_share(low_sector_map),
            "high_dom_sector_share": PortfolioDiagnostics.dominant_sector_share(high_sector_map),
        })

    # Aggregate dictionary-valued diagnostics across dates by simple averages.
    @staticmethod
    def average_dicts(series: pd.Series) -> Dict[str, float]:
        maps = [x for x in series.tolist() if isinstance(x, dict)]
        if not maps:
            return {}
        keys = sorted({k for mp in maps for k in mp.keys()})
        out: Dict[str, float] = {}
        n = float(len(maps))
        for k in keys:
            out[str(k)] = float(sum(float(mp.get(k, 0.0)) for mp in maps) / n)
        return out

    # Format the top-k entries from a weight/exposure map.
    @staticmethod
    def format_top_dict(weight_map: Dict[str, float], k: int = 3, signed: bool = False) -> str:
        if not weight_map:
            return "n/a"
        items = list(weight_map.items())
        if signed:
            items = sorted(items, key=lambda kv: abs(float(kv[1])), reverse=True)
            return ", ".join([f"{k}:{100.0 * float(v):+.1f}%" for k, v in items[: int(k)]])
        items = sorted(items, key=lambda kv: float(kv[1]), reverse=True)
        return ", ".join([f"{k}:{100.0 * float(v):.1f}%" for k, v in items[: int(k)]])

    # Format market-cap values in readable units.
    @staticmethod
    def format_busd(x: float) -> str:
        if not np.isfinite(x):
            return "n/a"
        ax = abs(float(x))
        if ax < 0.1:
            return f"${1000.0 * float(x):.1f}mm"
        return f"${float(x):.2f}bn"

    # Compute annualized return/std/sharpe and tail metrics for a return series.
    @staticmethod
    def annualized_return_stats(series: pd.Series, freq: str) -> Dict[str, float]:
        x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        period = 52 if freq == "week" else 12 if freq == "month" else 4
        if len(x) == 0:
            return {
                "ann_ret": float("nan"),
                "ann_std": float("nan"),
                "ann_sr": float("nan"),
                "worst_period": float("nan"),
                "p5_period": float("nan"),
                "downside_vol_ann": float("nan"),
                "hit_rate": float("nan"),
            }
        ann_ret = float(x.mean() * period)
        ann_std = float(x.std() * np.sqrt(period))
        ann_sr = float(ann_ret / ann_std) if ann_std > 0 and np.isfinite(ann_std) else float("nan")
        downside = x.copy()
        downside = downside.where(downside < 0.0, 0.0)
        downside_vol_ann = float(downside.std() * np.sqrt(period)) if len(downside) else float("nan")
        return {
            "ann_ret": ann_ret,
            "ann_std": ann_std,
            "ann_sr": ann_sr,
            "worst_period": float(x.min()),
            "p5_period": float(np.percentile(x.to_numpy(dtype=float), 5.0)),
            "downside_vol_ann": downside_vol_ann,
            "hit_rate": float((x > 0.0).mean()),
        }

    # Summarize raw TC debug rows into a dataframe and compact metrics dictionary.
    @staticmethod
    def summarize_tc_debug(weight_type: str, debug_rows: List[Dict]) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        canonical = str(weight_type).lower().strip()
        if not debug_rows:
            return None, None
        dbg = pd.DataFrame(debug_rows).copy()
        if "date" in dbg.columns:
            dbg["date"] = pd.to_datetime(dbg["date"], errors="coerce")
            dbg = dbg.sort_values("date").reset_index(drop=True)
        unmatched = dbg.get("unmatched_feasibility", pd.Series(False, index=dbg.index)).fillna(False).astype(bool)
        common = pd.to_numeric(dbg.get("common_target_sum", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        low_gross = pd.to_numeric(dbg.get("low_realized_gross", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        high_gross = pd.to_numeric(dbg.get("high_realized_gross", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        total_cost_bps = 1e4 * pd.to_numeric(dbg.get("total_cost", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        low_cost_bps = 1e4 * pd.to_numeric(dbg.get("low_cost", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        high_cost_bps = 1e4 * pd.to_numeric(dbg.get("high_cost", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        low_total = pd.to_numeric(dbg.get("low_total_names", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        low_trad = pd.to_numeric(dbg.get("low_tradable_names", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        high_total = pd.to_numeric(dbg.get("high_total_names", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        high_trad = pd.to_numeric(dbg.get("high_tradable_names", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        low_trad_share = np.where(low_total > 0, low_trad / low_total, np.nan)
        high_trad_share = np.where(high_total > 0, high_trad / high_total, np.nan)

        def _mean(colname: str) -> float:
            x = pd.to_numeric(dbg.get(colname, pd.Series(np.nan, index=dbg.index)), errors="coerce")
            return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

        summary = {
            "weight_type": canonical,
            "n_dates": int(len(dbg)),
            "execution_days": _mean("execution_days"),
            "unmatched_feasibility_count": int(unmatched.sum()),
            "unmatched_feasibility_pct": float(unmatched.mean()) if len(dbg) else float("nan"),
            "common_target_sum_mean": float(np.nanmean(common)) if np.isfinite(common).any() else float("nan"),
            "common_target_sum_median": float(np.nanmedian(common)) if np.isfinite(common).any() else float("nan"),
            "common_target_sum_lt_0_8_pct": float(np.nanmean(common < 0.8)) if np.isfinite(common).any() else float("nan"),
            "common_target_sum_lt_0_6_pct": float(np.nanmean(common < 0.6)) if np.isfinite(common).any() else float("nan"),
            "common_target_sum_lt_0_4_pct": float(np.nanmean(common < 0.4)) if np.isfinite(common).any() else float("nan"),
            "low_realized_gross_mean": float(np.nanmean(low_gross)) if np.isfinite(low_gross).any() else float("nan"),
            "high_realized_gross_mean": float(np.nanmean(high_gross)) if np.isfinite(high_gross).any() else float("nan"),
            "capacity_target_shortfall_mean": _mean("capacity_target_shortfall"),
            "low_posttrade_cash_weight_mean": _mean("low_posttrade_cash_weight"),
            "high_posttrade_cash_weight_mean": _mean("high_posttrade_cash_weight"),
            "low_unfilled_target_share_mean": _mean("low_unfilled_target_share"),
            "high_unfilled_target_share_mean": _mean("high_unfilled_target_share"),
            "mean_unfilled_target_share": _mean("mean_unfilled_target_share"),
            "low_execution_shortfall_vs_common_target_mean": _mean("low_execution_shortfall_vs_common_target"),
            "high_execution_shortfall_vs_common_target_mean": _mean("high_execution_shortfall_vs_common_target"),
            "total_cost_bps_mean": float(np.nanmean(total_cost_bps)) if np.isfinite(total_cost_bps).any() else float("nan"),
            "low_cost_bps_mean": float(np.nanmean(low_cost_bps)) if np.isfinite(low_cost_bps).any() else float("nan"),
            "high_cost_bps_mean": float(np.nanmean(high_cost_bps)) if np.isfinite(high_cost_bps).any() else float("nan"),
            "low_tradable_share_mean": float(np.nanmean(low_trad_share)) if np.isfinite(low_trad_share).any() else float("nan"),
            "high_tradable_share_mean": float(np.nanmean(high_trad_share)) if np.isfinite(high_trad_share).any() else float("nan"),
            "low_avg_spread_bps_mean": _mean("low_avg_spread_bps"),
            "high_avg_spread_bps_mean": _mean("high_avg_spread_bps"),
            "low_avg_adv_musd_mean": _mean("low_avg_adv_musd"),
            "high_avg_adv_musd_mean": _mean("high_avg_adv_musd"),
            "low_avg_sigma_mean": _mean("low_avg_sigma"),
            "high_avg_sigma_mean": _mean("high_avg_sigma"),
        }
        return dbg, summary

    # Summarize all-method shadow-cost, composition, turnover, and return diagnostics.
    @staticmethod
    def summarize_portfolio_debug(
        weight_type: str,
        debug_rows: List[Dict],
        *,
        portfolio_ret_eval: Optional[pd.DataFrame] = None,
        freq: str = "week",
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        canonical = str(weight_type).lower().strip()
        if not debug_rows:
            return None, None
        dbg = pd.DataFrame(debug_rows).copy()
        if "date" in dbg.columns:
            dbg["date"] = pd.to_datetime(dbg["date"], errors="coerce")
            dbg = dbg.sort_values("date").reset_index(drop=True)

        if portfolio_ret_eval is not None and len(portfolio_ret_eval) > 0:
            eval_dates = pd.to_datetime(pd.Index(portfolio_ret_eval.index), errors="coerce")
            dbg = dbg.loc[dbg["date"].isin(eval_dates)].copy().reset_index(drop=True)

        low_gross = pd.to_numeric(dbg.get("low_realized_gross", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        high_gross = pd.to_numeric(dbg.get("high_realized_gross", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        linear_cost_bps = 1e4 * pd.to_numeric(dbg.get("linear_cost", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        impact_cost_bps = 1e4 * pd.to_numeric(dbg.get("impact_cost", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        total_cost_bps = 1e4 * pd.to_numeric(dbg.get("total_cost", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        total_cost_bps_2x = 1e4 * pd.to_numeric(dbg.get("total_cost_2x", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        total_cost_bps_5x = 1e4 * pd.to_numeric(dbg.get("total_cost_5x", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        cost_attr_residual_bps = 1e4 * pd.to_numeric(dbg.get("cost_attr_residual", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        low_cost_bps = 1e4 * pd.to_numeric(dbg.get("low_cost", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        high_cost_bps = 1e4 * pd.to_numeric(dbg.get("high_cost", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        low_total = pd.to_numeric(dbg.get("low_total_names", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        low_trad = pd.to_numeric(dbg.get("low_tradable_names", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        high_total = pd.to_numeric(dbg.get("high_total_names", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        high_trad = pd.to_numeric(dbg.get("high_tradable_names", pd.Series(np.nan, index=dbg.index)), errors="coerce")
        low_trad_share = np.where(low_total > 0, low_trad / low_total, np.nan)
        high_trad_share = np.where(high_total > 0, high_trad / high_total, np.nan)

        def _mean(colname: str) -> float:
            x = pd.to_numeric(dbg.get(colname, pd.Series(np.nan, index=dbg.index)), errors="coerce")
            return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

        low_sector_avg = PortfolioDiagnostics.average_dicts(dbg.get("low_sector_map", pd.Series([], dtype=object)))
        high_sector_avg = PortfolioDiagnostics.average_dicts(dbg.get("high_sector_map", pd.Series([], dtype=object)))
        net_sector_avg = PortfolioDiagnostics.average_dicts(dbg.get("net_sector_map", pd.Series([], dtype=object)))

        return_view = {}
        if portfolio_ret_eval is not None and len(portfolio_ret_eval) > 0:
            pf = portfolio_ret_eval.copy()
            if not isinstance(pf.index, pd.DatetimeIndex):
                pf.index = pd.to_datetime(pf.index, errors="coerce")
            cost_by_date = dbg.set_index("date")["total_cost"].reindex(pf.index).fillna(0.0)
            tc_weight_types = {
                "ew_tc",
                "vw_tc",
                "mw_full_tc_nl",
                "mw_spread_full_tc_nl",
                "mw_spread_shrink_tc_nl",
                "mw_h_plus_lowneg_tc_nl",
            }
            if canonical in tc_weight_types:
                # Active TC strategies already subtract realized cost inside the
                # sleeve return.  Reconstruct pre-cost executed gross by adding
                # the canonical realized cost back exactly once.
                net_hl = pd.to_numeric(pf["H-L"], errors="coerce")
                gross_hl = net_hl + cost_by_date
            else:
                gross_hl = pd.to_numeric(pf["H-L"], errors="coerce")
                net_hl = gross_hl - cost_by_date
            gross_hl_stats = PortfolioDiagnostics.annualized_return_stats(gross_hl, freq)
            net_hl_stats = PortfolioDiagnostics.annualized_return_stats(net_hl, freq)
            high_stats = PortfolioDiagnostics.annualized_return_stats(pf[pf.columns[-2]], freq)
            low_bucket_stats = PortfolioDiagnostics.annualized_return_stats(pf[pf.columns[0]], freq)
            low_short_stats = PortfolioDiagnostics.annualized_return_stats(-pf[pf.columns[0]], freq)
            period = 52 if freq == "week" else 12 if freq == "month" else 4
            return_view = {
                "gross_hl_ann_ret": gross_hl_stats["ann_ret"],
                "gross_hl_ann_std": gross_hl_stats["ann_std"],
                "gross_hl_ann_sr": gross_hl_stats["ann_sr"],
                "shadow_net_hl_ann_ret": net_hl_stats["ann_ret"],
                "shadow_net_hl_ann_std": net_hl_stats["ann_std"],
                "shadow_net_hl_ann_sr": net_hl_stats["ann_sr"],
                "annualized_cost_drag": float(cost_by_date.mean() * period),
                "high_long_ann_ret": high_stats["ann_ret"],
                "low_bucket_ann_ret": low_bucket_stats["ann_ret"],
                "low_short_ann_ret": low_short_stats["ann_ret"],
                "hl_hit_rate": gross_hl_stats["hit_rate"],
                "hl_worst_period": gross_hl_stats["worst_period"],
                "hl_p5_period": gross_hl_stats["p5_period"],
                "hl_downside_vol_ann": gross_hl_stats["downside_vol_ann"],
            }

        worst_cost_dates = "n/a"
        if "date" in dbg.columns and np.isfinite(total_cost_bps).any():
            tmp = pd.DataFrame({"date": dbg["date"], "total_cost_bps": total_cost_bps}).dropna()
            tmp = tmp.sort_values("total_cost_bps", ascending=False).head(3)
            if len(tmp):
                worst_cost_dates = ", ".join([
                    f"{pd.Timestamp(d).date()}:{float(c):.1f}bps"
                    for d, c in zip(tmp["date"], tmp["total_cost_bps"])
                ])

        summary = {
            "weight_type": canonical,
            "n_dates": int(len(dbg)),
            "execution_days": _mean("execution_days"),
            "pretrade_cash_weight_mean": _mean("pretrade_cash_weight"),
            "posttrade_cash_weight_mean": _mean("posttrade_cash_weight"),
            "pretrade_net_exposure_mean": _mean("pretrade_net_exposure"),
            "posttrade_net_exposure_mean": _mean("posttrade_net_exposure"),
            "pretrade_gross_exposure_mean": _mean("pretrade_gross_exposure"),
            "posttrade_gross_exposure_mean": _mean("posttrade_gross_exposure"),
            "canonical_turnover_oneway_mean": _mean("canonical_turnover_oneway"),
            "canonical_oneway_dollar_turnover_mean": _mean("canonical_oneway_dollar_turnover"),
            "canonical_total_abs_dollar_trades_mean": _mean("canonical_total_abs_dollar_trades"),
            "low_realized_gross_mean": float(np.nanmean(low_gross)) if np.isfinite(low_gross).any() else float("nan"),
            "high_realized_gross_mean": float(np.nanmean(high_gross)) if np.isfinite(high_gross).any() else float("nan"),
            "linear_cost_bps_mean": float(np.nanmean(linear_cost_bps)) if np.isfinite(linear_cost_bps).any() else float("nan"),
            "impact_cost_bps_mean": float(np.nanmean(impact_cost_bps)) if np.isfinite(impact_cost_bps).any() else float("nan"),
            "total_cost_bps_mean": float(np.nanmean(total_cost_bps)) if np.isfinite(total_cost_bps).any() else float("nan"),
            "total_cost_bps_median": float(np.nanmedian(total_cost_bps)) if np.isfinite(total_cost_bps).any() else float("nan"),
            "total_cost_bps_p90": float(np.nanpercentile(total_cost_bps, 90.0)) if np.isfinite(total_cost_bps).any() else float("nan"),
            "total_cost_bps_p95": float(np.nanpercentile(total_cost_bps, 95.0)) if np.isfinite(total_cost_bps).any() else float("nan"),
            "total_cost_bps_2x_mean": float(np.nanmean(total_cost_bps_2x)) if np.isfinite(total_cost_bps_2x).any() else float("nan"),
            "total_cost_bps_5x_mean": float(np.nanmean(total_cost_bps_5x)) if np.isfinite(total_cost_bps_5x).any() else float("nan"),
            "low_cost_bps_mean": float(np.nanmean(low_cost_bps)) if np.isfinite(low_cost_bps).any() else float("nan"),
            "high_cost_bps_mean": float(np.nanmean(high_cost_bps)) if np.isfinite(high_cost_bps).any() else float("nan"),
            "low_tradable_share_mean": float(np.nanmean(low_trad_share)) if np.isfinite(low_trad_share).any() else float("nan"),
            "high_tradable_share_mean": float(np.nanmean(high_trad_share)) if np.isfinite(high_trad_share).any() else float("nan"),
            "low_avg_spread_bps_mean": _mean("low_avg_spread_bps"),
            "high_avg_spread_bps_mean": _mean("high_avg_spread_bps"),
            "low_avg_adv_musd_mean": _mean("low_avg_adv_musd"),
            "high_avg_adv_musd_mean": _mean("high_avg_adv_musd"),
            "low_avg_sigma_mean": _mean("low_avg_sigma"),
            "high_avg_sigma_mean": _mean("high_avg_sigma"),
            "low_avg_mcap_busd_mean": _mean("low_avg_mcap_busd"),
            "high_avg_mcap_busd_mean": _mean("high_avg_mcap_busd"),
            "low_median_mcap_busd_mean": _mean("low_median_mcap_busd"),
            "high_median_mcap_busd_mean": _mean("high_median_mcap_busd"),
            "low_p10_mcap_busd_mean": _mean("low_p10_mcap_busd"),
            "high_p10_mcap_busd_mean": _mean("high_p10_mcap_busd"),
            "low_p90_mcap_busd_mean": _mean("low_p90_mcap_busd"),
            "high_p90_mcap_busd_mean": _mean("high_p90_mcap_busd"),
            "low_avg_log_mcap_mean": _mean("low_avg_log_mcap"),
            "high_avg_log_mcap_mean": _mean("high_avg_log_mcap"),
            "low_micro_weight_share_mean": _mean("low_micro_weight_share"),
            "high_micro_weight_share_mean": _mean("high_micro_weight_share"),
            "low_smallcap_weight_share_mean": _mean("low_smallcap_weight_share"),
            "high_smallcap_weight_share_mean": _mean("high_smallcap_weight_share"),
            "low_midcap_weight_share_mean": _mean("low_midcap_weight_share"),
            "high_midcap_weight_share_mean": _mean("high_midcap_weight_share"),
            "low_largecap_weight_share_mean": _mean("low_largecap_weight_share"),
            "high_largecap_weight_share_mean": _mean("high_largecap_weight_share"),
            "low_effective_n_mean": _mean("low_effective_n"),
            "high_effective_n_mean": _mean("high_effective_n"),
            "low_top5_weight_share_mean": _mean("low_top5_weight_share"),
            "high_top5_weight_share_mean": _mean("high_top5_weight_share"),
            "low_top10_weight_share_mean": _mean("low_top10_weight_share"),
            "high_top10_weight_share_mean": _mean("high_top10_weight_share"),
            "low_max_weight_mean": _mean("low_max_weight"),
            "high_max_weight_mean": _mean("high_max_weight"),
            "low_median_active_weight_mean": _mean("low_median_active_weight"),
            "high_median_active_weight_mean": _mean("high_median_active_weight"),
            "low_p95_active_weight_mean": _mean("low_p95_active_weight"),
            "high_p95_active_weight_mean": _mean("high_p95_active_weight"),
            "low_active_names_mean": float(np.nanmean(low_total)) if np.isfinite(low_total).any() else float("nan"),
            "high_active_names_mean": float(np.nanmean(high_total)) if np.isfinite(high_total).any() else float("nan"),
            "total_turnover_mean": _mean("total_turnover"),
            "entry_turnover_mean": _mean("entry_turnover"),
            "exit_turnover_mean": _mean("exit_turnover"),
            "resize_turnover_mean": _mean("resize_turnover"),
            "entry_turnover_share_mean": float(np.nanmean(pd.to_numeric(dbg.get("entry_turnover", pd.Series(np.nan, index=dbg.index)), errors="coerce") / pd.to_numeric(dbg.get("total_turnover", pd.Series(np.nan, index=dbg.index)), errors="coerce"))) if "entry_turnover" in dbg.columns else float("nan"),
            "exit_turnover_share_mean": float(np.nanmean(pd.to_numeric(dbg.get("exit_turnover", pd.Series(np.nan, index=dbg.index)), errors="coerce") / pd.to_numeric(dbg.get("total_turnover", pd.Series(np.nan, index=dbg.index)), errors="coerce"))) if "exit_turnover" in dbg.columns else float("nan"),
            "resize_turnover_share_mean": float(np.nanmean(pd.to_numeric(dbg.get("resize_turnover", pd.Series(np.nan, index=dbg.index)), errors="coerce") / pd.to_numeric(dbg.get("total_turnover", pd.Series(np.nan, index=dbg.index)), errors="coerce"))) if "resize_turnover" in dbg.columns else float("nan"),
            "low_turnover_mean": _mean("low_turnover"),
            "high_turnover_mean": _mean("high_turnover"),
            "low_jaccard_mean": _mean("low_jaccard"),
            "high_jaccard_mean": _mean("high_jaccard"),
            "low_top10_overlap_mean": _mean("low_top10_overlap"),
            "high_top10_overlap_mean": _mean("high_top10_overlap"),
            "low_replacement_frac_mean": _mean("low_replacement_frac"),
            "high_replacement_frac_mean": _mean("high_replacement_frac"),
            "any_pos_cap_bind_pct": float(np.nanmean(pd.Series(dbg.get("any_pos_cap_bind", pd.Series(False, index=dbg.index))).astype(float))),
            "any_trade_cap_bind_pct": float(np.nanmean(pd.Series(dbg.get("any_trade_cap_bind", pd.Series(False, index=dbg.index))).astype(float))),
            "pos_cap_bind_weight_share_mean": _mean("pos_cap_bind_weight_share"),
            "trade_cap_bind_weight_share_mean": _mean("trade_cap_bind_weight_share"),
            "low_sector_hhi_mean": _mean("low_sector_hhi"),
            "high_sector_hhi_mean": _mean("high_sector_hhi"),
            "low_dom_sector_share_mean": _mean("low_dom_sector_share"),
            "high_dom_sector_share_mean": _mean("high_dom_sector_share"),
            "cost_attr_residual_bps_mean": float(np.nanmean(cost_attr_residual_bps)) if np.isfinite(cost_attr_residual_bps).any() else float("nan"),
            "low_sector_known_weight_share_mean": _mean("low_sector_known_weight_share"),
            "high_sector_known_weight_share_mean": _mean("high_sector_known_weight_share"),
            "low_dom_sector_gt20_pct": float(np.nanmean(pd.to_numeric(dbg.get("low_dom_sector_share", pd.Series(np.nan, index=dbg.index)), errors="coerce") > 0.20)),
            "high_dom_sector_gt20_pct": float(np.nanmean(pd.to_numeric(dbg.get("high_dom_sector_share", pd.Series(np.nan, index=dbg.index)), errors="coerce") > 0.20)),
            "low_dom_sector_gt30_pct": float(np.nanmean(pd.to_numeric(dbg.get("low_dom_sector_share", pd.Series(np.nan, index=dbg.index)), errors="coerce") > 0.30)),
            "high_dom_sector_gt30_pct": float(np.nanmean(pd.to_numeric(dbg.get("high_dom_sector_share", pd.Series(np.nan, index=dbg.index)), errors="coerce") > 0.30)),
            "low_sector_top3": PortfolioDiagnostics.format_top_dict(low_sector_avg, 3, signed=False),
            "high_sector_top3": PortfolioDiagnostics.format_top_dict(high_sector_avg, 3, signed=False),
            "net_sector_top3": PortfolioDiagnostics.format_top_dict(net_sector_avg, 3, signed=True),
            "worst_cost_dates": worst_cost_dates,
        }
        summary.update(return_view)
        return dbg, summary

    # Print the compact TC diagnostics summary.
    @staticmethod
    def print_tc_debug_summary(summary: Optional[Dict]) -> None:
        if not summary:
            return
        print(
            f"[TC Diagnostics | {summary['weight_type']}] dates={summary['n_dates']:,} | "
            f"unmatched_feasibility={summary['unmatched_feasibility_count']:,} "
            f"({100.0 * summary['unmatched_feasibility_pct']:.1f}%)"
        )
        if np.isfinite(summary.get("execution_days", float("nan"))):
            print(
                "[TC Diagnostics] execution convention "
                f"scheduled rebalance execution={summary['execution_days']:.2f} trading day(s)"
            )
        hardening_keys = (
            "optimizer_solve_calls",
            "optimizer_fallback_used_events",
            "optimizer_nonconverged_events",
            "insufficient_covariance_no_trade_events",
            "zero_alpha_no_trade_events",
            "frozen_nontradable_name_instances",
            "covariance_excluded_name_instances",
        )
        if any(int(summary.get(k, 0) or 0) != 0 for k in hardening_keys):
            print(
                "[TC Diagnostics] hardening events "
                + " | ".join(f"{k}={int(summary.get(k, 0) or 0)}" for k in hardening_keys)
            )
        if np.isfinite(summary["common_target_sum_mean"]):
            print(
                "[TC Diagnostics] common_target_sum "
                f"mean={summary['common_target_sum_mean']:.3f} | median={summary['common_target_sum_median']:.3f} "
                f"| <0.8={100.0 * summary['common_target_sum_lt_0_8_pct']:.1f}% "
                f"| <0.6={100.0 * summary['common_target_sum_lt_0_6_pct']:.1f}% "
                f"| <0.4={100.0 * summary['common_target_sum_lt_0_4_pct']:.1f}%"
            )
        print(
            "[TC Diagnostics] realized gross "
            f"low={summary['low_realized_gross_mean']:.3f} | high={summary['high_realized_gross_mean']:.3f}"
        )
        if np.isfinite(summary.get("mean_unfilled_target_share", float("nan"))):
            print(
                "[TC Diagnostics] cash / unfilled target "
                f"low cash={100.0 * summary['low_posttrade_cash_weight_mean']:.2f}% | "
                f"high cash={100.0 * summary['high_posttrade_cash_weight_mean']:.2f}% | "
                f"mean unfilled={100.0 * summary['mean_unfilled_target_share']:.2f}% | "
                f"capacity target shortfall={100.0 * summary['capacity_target_shortfall_mean']:.2f}%"
            )
        print(
            "[TC Diagnostics] realized cost per rebalance (bps) "
            f"total={summary['total_cost_bps_mean']:.2f} | low={summary['low_cost_bps_mean']:.2f} | high={summary['high_cost_bps_mean']:.2f}"
        )
        if np.isfinite(summary["low_tradable_share_mean"]) or np.isfinite(summary["high_tradable_share_mean"]):
            print(
                "[TC Diagnostics] tradable share "
                f"low={100.0 * summary['low_tradable_share_mean']:.1f}% | high={100.0 * summary['high_tradable_share_mean']:.1f}%"
            )
        if np.isfinite(summary["low_avg_spread_bps_mean"]) or np.isfinite(summary["high_avg_spread_bps_mean"]):
            print(
                "[TC Diagnostics] avg spread (bps, tradable names) "
                f"low={summary['low_avg_spread_bps_mean']:.2f} | high={summary['high_avg_spread_bps_mean']:.2f}"
            )
        if np.isfinite(summary["low_avg_adv_musd_mean"]) or np.isfinite(summary["high_avg_adv_musd_mean"]):
            print(
                "[TC Diagnostics] avg ADV ($mm, tradable names) "
                f"low={summary['low_avg_adv_musd_mean']:.2f} | high={summary['high_avg_adv_musd_mean']:.2f}"
            )
        if np.isfinite(summary["low_avg_sigma_mean"]) or np.isfinite(summary["high_avg_sigma_mean"]):
            print(
                "[TC Diagnostics] avg sigma (tradable names) "
                f"low={summary['low_avg_sigma_mean']:.4f} | high={summary['high_avg_sigma_mean']:.4f}"
            )

    # Print the all-method shadow-cost, composition, turnover, and return summary.
    @staticmethod
    def print_portfolio_debug_summary(summary: Optional[Dict]) -> None:
        if not summary:
            return
        print(f"[Portfolio Diagnostics | {summary['weight_type']}] dates={summary['n_dates']:,}")
        if np.isfinite(summary.get("execution_days", float("nan"))):
            print(
                "[Portfolio Diagnostics] execution convention "
                f"scheduled rebalance execution={summary['execution_days']:.2f} trading day(s)"
            )
        print(
            "[Portfolio Diagnostics] realized gross "
            f"low={summary['low_realized_gross_mean']:.3f} | high={summary['high_realized_gross_mean']:.3f}"
        )
        print(
            "[Portfolio Diagnostics] ex-post cost per rebalance (bps) "
            f"total={summary['total_cost_bps_mean']:.2f} | linear={summary['linear_cost_bps_mean']:.2f} | impact={summary['impact_cost_bps_mean']:.2f} "
            f"| low={summary['low_cost_bps_mean']:.2f} | high={summary['high_cost_bps_mean']:.2f} | residual={summary['cost_attr_residual_bps_mean']:.2f}"
        )
        print(
            "[Portfolio Diagnostics] cost distribution / capacity "
            f"median={summary['total_cost_bps_median']:.2f}bps | p90={summary['total_cost_bps_p90']:.2f}bps | p95={summary['total_cost_bps_p95']:.2f}bps "
            f"| 2xAUM={summary['total_cost_bps_2x_mean']:.2f}bps | 5xAUM={summary['total_cost_bps_5x_mean']:.2f}bps"
        )
        print(f"[Portfolio Diagnostics] worst rebalance dates by cost {summary['worst_cost_dates']}")
        if np.isfinite(summary["low_tradable_share_mean"]) or np.isfinite(summary["high_tradable_share_mean"]):
            print(
                "[Portfolio Diagnostics] tradable share "
                f"low={100.0 * summary['low_tradable_share_mean']:.1f}% | high={100.0 * summary['high_tradable_share_mean']:.1f}%"
            )
        if np.isfinite(summary["low_avg_spread_bps_mean"]) or np.isfinite(summary["high_avg_spread_bps_mean"]):
            print(
                "[Portfolio Diagnostics] avg spread (bps, active names) "
                f"low={summary['low_avg_spread_bps_mean']:.2f} | high={summary['high_avg_spread_bps_mean']:.2f}"
            )
        if np.isfinite(summary["low_avg_adv_musd_mean"]) or np.isfinite(summary["high_avg_adv_musd_mean"]):
            print(
                "[Portfolio Diagnostics] avg ADV ($mm, active names) "
                f"low={summary['low_avg_adv_musd_mean']:.2f} | high={summary['high_avg_adv_musd_mean']:.2f}"
            )
        if np.isfinite(summary["low_avg_sigma_mean"]) or np.isfinite(summary["high_avg_sigma_mean"]):
            print(
                "[Portfolio Diagnostics] avg sigma (active names) "
                f"low={summary['low_avg_sigma_mean']:.4f} | high={summary['high_avg_sigma_mean']:.4f}"
            )
        if np.isfinite(summary["low_avg_mcap_busd_mean"]) or np.isfinite(summary["high_avg_mcap_busd_mean"]):
            print(
                "[Portfolio Diagnostics] MarketCap avg / median / p10 / p90 "
                f"low={PortfolioDiagnostics.format_busd(summary['low_avg_mcap_busd_mean'])} / "
                f"{PortfolioDiagnostics.format_busd(summary['low_median_mcap_busd_mean'])} / "
                f"{PortfolioDiagnostics.format_busd(summary['low_p10_mcap_busd_mean'])} / "
                f"{PortfolioDiagnostics.format_busd(summary['low_p90_mcap_busd_mean'])} "
                f"| high={PortfolioDiagnostics.format_busd(summary['high_avg_mcap_busd_mean'])} / "
                f"{PortfolioDiagnostics.format_busd(summary['high_median_mcap_busd_mean'])} / "
                f"{PortfolioDiagnostics.format_busd(summary['high_p10_mcap_busd_mean'])} / "
                f"{PortfolioDiagnostics.format_busd(summary['high_p90_mcap_busd_mean'])}"
            )
        if np.isfinite(summary["low_avg_log_mcap_mean"]) or np.isfinite(summary["high_avg_log_mcap_mean"]):
            print(
                "[Portfolio Diagnostics] avg log MarketCap "
                f"low={summary['low_avg_log_mcap_mean']:.2f} | high={summary['high_avg_log_mcap_mean']:.2f}"
            )
        print(
            "[Portfolio Diagnostics] size-bucket weight share "
            f"low micro/small/mid/large={100.0 * summary['low_micro_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['low_smallcap_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['low_midcap_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['low_largecap_weight_share_mean']:.1f}% "
            f"| high={100.0 * summary['high_micro_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['high_smallcap_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['high_midcap_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['high_largecap_weight_share_mean']:.1f}%"
        )
        print(
            "[Portfolio Diagnostics] effective N / active names "
            f"low={summary['low_effective_n_mean']:.1f} / {summary['low_active_names_mean']:.1f} "
            f"| high={summary['high_effective_n_mean']:.1f} / {summary['high_active_names_mean']:.1f}"
        )
        print(
            "[Portfolio Diagnostics] concentration top5 / top10 / max / median / p95 weight "
            f"low={100.0 * summary['low_top5_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['low_top10_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['low_max_weight_mean']:.2f}% / "
            f"{100.0 * summary['low_median_active_weight_mean']:.2f}% / "
            f"{100.0 * summary['low_p95_active_weight_mean']:.2f}% "
            f"| high={100.0 * summary['high_top5_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['high_top10_weight_share_mean']:.1f}% / "
            f"{100.0 * summary['high_max_weight_mean']:.2f}% / "
            f"{100.0 * summary['high_median_active_weight_mean']:.2f}% / "
            f"{100.0 * summary['high_p95_active_weight_mean']:.2f}%"
        )
        print(
            "[Portfolio Diagnostics] known sector weight share "
            f"low={100.0 * summary['low_sector_known_weight_share_mean']:.1f}% | high={100.0 * summary['high_sector_known_weight_share_mean']:.1f}%"
        )
        print(
            "[Portfolio Diagnostics] sector top3 / net tilts "
            f"low={summary['low_sector_top3']} | high={summary['high_sector_top3']} | H-L={summary['net_sector_top3']}"
        )
        print(
            "[Portfolio Diagnostics] sector concentration HHI / dominant share "
            f"low={summary['low_sector_hhi_mean']:.3f} / {100.0 * summary['low_dom_sector_share_mean']:.1f}% "
            f"| high={summary['high_sector_hhi_mean']:.3f} / {100.0 * summary['high_dom_sector_share_mean']:.1f}%"
        )
        print(
            "[Portfolio Diagnostics] dominant-sector >20% / >30% of sleeve "
            f"low={100.0 * summary['low_dom_sector_gt20_pct']:.1f}% / {100.0 * summary['low_dom_sector_gt30_pct']:.1f}% "
            f"| high={100.0 * summary['high_dom_sector_gt20_pct']:.1f}% / {100.0 * summary['high_dom_sector_gt30_pct']:.1f}%"
        )
        print(
            "[Portfolio Diagnostics] turnover mean / entry / exit / resize "
            f"total={summary['total_turnover_mean']:.3f} | entry={summary['entry_turnover_mean']:.3f} "
            f"| exit={summary['exit_turnover_mean']:.3f} | resize={summary['resize_turnover_mean']:.3f}"
        )
        print(
            "[Portfolio Diagnostics] turnover share entry / exit / resize "
            f"{100.0 * summary['entry_turnover_share_mean']:.1f}% / "
            f"{100.0 * summary['exit_turnover_share_mean']:.1f}% / "
            f"{100.0 * summary['resize_turnover_share_mean']:.1f}%"
        )
        print(
            "[Portfolio Diagnostics] sleeve turnover / overlap / replacement "
            f"low turnover={summary['low_turnover_mean']:.3f}, jaccard={summary['low_jaccard_mean']:.3f}, top10 overlap={summary['low_top10_overlap_mean']:.3f}, replaced={100.0 * summary['low_replacement_frac_mean']:.1f}% "
            f"| high turnover={summary['high_turnover_mean']:.3f}, jaccard={summary['high_jaccard_mean']:.3f}, top10 overlap={summary['high_top10_overlap_mean']:.3f}, replaced={100.0 * summary['high_replacement_frac_mean']:.1f}%"
        )
        print(
            "[Portfolio Diagnostics] cap stress dates / violating share "
            f"pos cap={100.0 * summary['any_pos_cap_bind_pct']:.1f}% ({100.0 * summary['pos_cap_bind_weight_share_mean']:.1f}% weight) "
            f"| trade cap={100.0 * summary['any_trade_cap_bind_pct']:.1f}% ({100.0 * summary['trade_cap_bind_weight_share_mean']:.1f}% trade)"
        )
        if "gross_hl_ann_ret" in summary:
            print(
                "[Portfolio Diagnostics] H-L gross / shadow-net annualized ret,std,SR "
                f"gross={summary['gross_hl_ann_ret']:.3f}, {summary['gross_hl_ann_std']:.3f}, {summary['gross_hl_ann_sr']:.3f} "
                f"| net={summary['shadow_net_hl_ann_ret']:.3f}, {summary['shadow_net_hl_ann_std']:.3f}, {summary['shadow_net_hl_ann_sr']:.3f}"
            )
            print(
                "[Portfolio Diagnostics] annualized contribution view "
                f"high long={summary['high_long_ann_ret']:.3f} | low bucket={summary['low_bucket_ann_ret']:.3f} "
                f"| low short={summary['low_short_ann_ret']:.3f} | cost drag={summary['annualized_cost_drag']:.3f}"
            )
            print(
                "[Portfolio Diagnostics] H-L tail / hit-rate "
                f"worst period={summary['hl_worst_period']:.4f} | p5={summary['hl_p5_period']:.4f} "
                f"| downside vol ann={summary['hl_downside_vol_ann']:.3f} | hit-rate={100.0 * summary['hl_hit_rate']:.1f}%"
            )

    # Print the probability/return correlation summary used during portfolio runs.
    @staticmethod
    def print_prediction_correlation_summary(
        *,
        prob_ret_corr_eval: np.ndarray,
        prob_ret_pearson_eval: np.ndarray,
        prob_inv_ret_corr_eval: Optional[np.ndarray] = None,
        prob_inv_ret_pearson_eval: Optional[np.ndarray] = None,
    ) -> None:
        print(f"Spearman Corr(up_prob, ret) : {np.nanmean(prob_ret_corr_eval):.4f}")
        print(f"Pearson  Corr(up_prob, ret) : {np.nanmean(prob_ret_pearson_eval):.4f}")
        if prob_inv_ret_corr_eval is not None and prob_inv_ret_pearson_eval is not None:
            print(f"Spearman Corr(up_prob, inv_ret Top/Bottom) : {np.nanmean(prob_inv_ret_corr_eval):.4f}")
            print(f"Pearson  Corr(up_prob, inv_ret Top/Bottom) : {np.nanmean(prob_inv_ret_pearson_eval):.4f}")

    # Build the annualized portfolio summary table and print it.
    @staticmethod
    def build_portfolio_summary_table(*, portfolio_ret: pd.DataFrame, turnover: float, freq: str, cut: int = 10) -> pd.DataFrame:
        avg = portfolio_ret.mean().to_numpy()
        std = portfolio_ret.std().to_numpy()
        period = 52 if freq == "week" else 12 if freq == "month" else 4
        res = np.zeros((cut + 1, 3))
        res[:, 0] = avg * period
        res[:, 1] = std * np.sqrt(period)
        res[:, 2] = np.divide(res[:, 0], res[:, 1], out=np.zeros_like(res[:, 0]), where=res[:, 1] != 0)
        summary = pd.DataFrame(res, columns=["ret", "std", "SR"])
        summary.index = pd.Index(["Low"] + list(range(2, cut)) + ["High", "H-L"])
        summary.loc["Turnover", "ret"] = turnover
        summary.loc["Turnover", "std"] = np.nan
        summary.loc["Turnover", "SR"] = turnover * period
        print(summary)
        return summary
