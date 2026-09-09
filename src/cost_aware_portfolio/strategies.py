"""Portfolio strategy implementations for base and active cost-aware portfolios.

This module keeps the original equal-weight, value-weight, and Markowitz
portfolio logic while replacing the old EW/VW transaction-cost heuristics with
updated nonlinear target-tracking rebalancing. The active transaction-cost
strategies are:
- `ew_tc`
- `vw_tc`
- `mw_full_tc_nl`
- `mw_spread_full_tc_nl`
- `mw_spread_shrink_tc_nl`
- `mw_h_plus_lowneg_tc_nl`

This rewritten version also coordinates the top and bottom sleeves when the
final reported portfolio is long top-decile and short bottom-decile. For the
active transaction-cost strategies:
- the top and bottom sleeves first compute their own feasible long-only
  liquidity intervals,
- if the two intervals overlap, both sleeves are solved to the same common
  gross so the reported H-L portfolio remains matched,
- if the intervals do not overlap, the code falls back to independent partial
  fill rather than crashing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cost_aware_portfolio.diagnostics import PortfolioDiagnostics
from cost_aware_portfolio.transaction_costs import (
    TC_ADV_DAILY_COL,
    TC_SIGMA_DAILY_COL,
)
from cost_aware_portfolio.state import (
    PreTradeState,
    RebalanceResult,
    build_pretrade_state,
    build_rebalance_result,
    build_rebalance_universe,
)


class PortfolioStrategy:
    """Base interface for one-date portfolio-construction strategies."""

    # Initialize the strategy with a portfolio-manager context object.
    def __init__(self, ctx) -> None:
        self.ctx = ctx

    # Compute a finite percentile after dropping non-numeric and infinite values.
    @staticmethod
    def _finite_percentile(s: pd.Series, q: float) -> float:
        x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if x.size == 0:
            return float("nan")
        return float(np.percentile(x, q))

    # Require distinct finite signal tails before forming low/high portfolios.
    @staticmethod
    def _require_tail_thresholds(
        reb: pd.DataFrame,
        *,
        date: pd.Timestamp,
    ) -> Tuple[float, float]:
        if "up_prob" not in reb.columns:
            raise KeyError("Signal frame is missing required column 'up_prob'.")

        q10 = PortfolioStrategy._finite_percentile(reb["up_prob"], 10)
        q90 = PortfolioStrategy._finite_percentile(reb["up_prob"], 90)
        if (not np.isfinite(q10)) or (not np.isfinite(q90)) or q10 >= q90:
            raise RuntimeError(
                "degenerate signal tails on "
                f"{pd.Timestamp(date).date()}: q10={q10}, q90={q90}."
            )
        return float(q10), float(q90)

    # Build value weights without silently substituting another weighting rule.
    @staticmethod
    def _strict_value_weights(
        frame: pd.DataFrame,
        *,
        date: pd.Timestamp,
        label: str,
    ) -> pd.Series:
        if "MarketCap" not in frame.columns:
            raise RuntimeError(
                f"{label}: MarketCap is unavailable on {pd.Timestamp(date).date()}."
            )

        mcap = pd.to_numeric(frame["MarketCap"], errors="coerce").astype(float)
        finite = mcap.notna() & np.isfinite(mcap)
        if not bool(finite.all()):
            raise RuntimeError(
                f"{label}: MarketCap contains missing or non-finite values on "
                f"{pd.Timestamp(date).date()}."
            )
        if bool((mcap < 0.0).any()):
            raise RuntimeError(
                f"{label}: MarketCap contains negative values on {pd.Timestamp(date).date()}."
            )

        total = float(mcap.sum())
        if (not np.isfinite(total)) or total <= 0.0:
            raise RuntimeError(
                f"{label}: MarketCap sum is non-positive on {pd.Timestamp(date).date()}."
            )
        return (mcap / total).astype(float)

    # Restrict a covariance block to names with enough strictly prior return history.
    def _eligible_cov_block(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        ret_name: str,
        frame: pd.DataFrame,
        *,
        leg_name: str,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        ctx = self.ctx
        if len(frame) == 0:
            raise RuntimeError(f"{self.__class__.__name__}: empty frame passed to _eligible_cov_block for {leg_name}.")

        ctx._ensure_pivot(df, ret_name)
        elig = getattr(ctx, "_eligible_ids", lambda ids, d: list(pd.Index(ids).astype(str)))(
            pd.Index(frame.index),
            date,
        )
        if len(elig) < 2:
            raise RuntimeError(
                f"{self.__class__.__name__}: insufficient eligible {leg_name} names for date={date}."
            )

        elig_set = {str(x) for x in elig}
        filtered = frame.loc[[str(i) in elig_set for i in frame.index]].copy()
        if len(filtered) < 2:
            raise RuntimeError(
                f"{self.__class__.__name__}: filtered {leg_name} block has fewer than two names for date={date}."
            )

        sigma, cols = ctx._build_cov_matrix(df, date, pd.Index(filtered.index).astype(str), ret_name)
        if sigma is None or cols is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: covariance matrix unavailable for {leg_name} on date={date}."
            )
        if len(cols) < 2:
            raise RuntimeError(
                f"{self.__class__.__name__}: covariance block for {leg_name} has fewer than two names for date={date}."
            )
        return filtered, sigma, cols

    # Build one rebalance date of outputs for the concrete strategy.
    def compute_for_date(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        cut: int,
        ret_name: str,
        prev_state: Dict,
    ) -> Tuple[np.ndarray, pd.DataFrame, Optional[pd.DataFrame]]:
        raise NotImplementedError



class TCOptimizerMixin:
    """Shared helpers for active nonlinear transaction-cost strategies."""

    # Validate that the covariance block returned by the context is usable.
    @staticmethod
    def _require_covariance(
        *,
        strategy_name: str,
        Sigma: Optional[np.ndarray],
        cols: Optional[List[str]],
    ) -> List[str]:
        if Sigma is None or cols is None:
            raise RuntimeError(f"{strategy_name}: covariance matrix is unavailable for the transaction-cost block.")
        if len(cols) < 2:
            raise RuntimeError(f"{strategy_name}: covariance block contains fewer than two names.")
        return list(cols)

    # Build nonlinear transaction-cost parameter settings for one optimizer-backed leg.
    def _build_tc_params(
        self,
        ctx,
        *,
        long_only: bool,
        dollar_neutral: bool,
        gross_target: float,
    ):
        from cost_aware_portfolio.optimizer import CostAwareParams

        return CostAwareParams(
            lambda_risk=ctx.tc_lambda_risk,
            eta_tc=ctx.tc_eta,
            eta_linear=ctx.tc_eta_linear,
            eta_impact=ctx.tc_eta_impact,
            impact_kappa=ctx.tc_impact_kappa,
            impact_exponent=ctx.tc_impact_exponent,
            impact_smooth_eps=ctx.tc_impact_smooth_eps,
            spread_bps_default=ctx.tc_spread_bps_default,
            dollar_neutral=bool(dollar_neutral),
            long_only=bool(long_only),
            gross_target=float(gross_target),
            pos_adv_cap=ctx.tc_pos_adv_cap,
            trade_adv_cap=ctx.tc_trade_adv_cap,
            execution_days=float(getattr(ctx, "tc_execution_days", 1.0)),
            box_abs=ctx.tc_box_abs,
            allow_partial_long_only=True,
        )

    # Identify names with complete canonical execution inputs on this rebalance.
    def _tc_tradable_mask(self, *, ctx, frame: pd.DataFrame) -> pd.Series:
        idx = frame.index
        mask = pd.Series(True, index=idx, dtype=bool)

        # Active transaction-cost strategies require the frozen OOS DAILY ADV /
        # DAILY sigma pair.  Historical DollarVol_20d and weekly mw_var are not
        # accepted as substitutes here.
        if TC_ADV_DAILY_COL not in frame.columns or TC_SIGMA_DAILY_COL not in frame.columns:
            return pd.Series(False, index=idx, dtype=bool)

        adv = pd.to_numeric(frame[TC_ADV_DAILY_COL], errors="coerce")
        sigma = pd.to_numeric(frame[TC_SIGMA_DAILY_COL], errors="coerce")
        mask &= adv.notna() & np.isfinite(adv) & (adv > 0.0)
        mask &= sigma.notna() & np.isfinite(sigma) & (sigma >= 0.0)

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

    # Return the persistent store used for canonical transaction-cost leg states.
    @staticmethod
    def _tc_leg_state_store(prev_state: Dict) -> Dict[str, Dict[str, object]]:
        return prev_state.setdefault("tc_leg_states", {})

    # Use the pre-screen execution-support panel for legacy holdings whenever it
    # is available.  New names are still selected from the screened signal frame;
    # this broader panel is only for carrying, pricing, and liquidating positions
    # that were valid when entered but later failed a screen.
    @staticmethod
    def _tc_execution_support_frame(*, ctx, date: pd.Timestamp, selection_frame: pd.DataFrame) -> pd.DataFrame:
        getter = getattr(ctx, "_tc_execution_frame", None)
        if not callable(getter):
            raise RuntimeError(
                "Stateful transaction-cost execution requires ctx._tc_execution_frame; "
                "refusing to fall back to the screened selection panel."
            )
        out = getter(pd.Timestamp(date)).copy()
        out.index = pd.Index(out.index).astype(str)
        return out

    # Build the actual drifted pre-trade state for one strategy sleeve.
    def _tc_pretrade_state(
        self,
        *,
        ctx,
        prev_state: Dict,
        state_key: str,
        date: pd.Timestamp,
    ) -> PreTradeState:
        store = self._tc_leg_state_store(prev_state)
        saved = store.get(str(state_key))
        if saved is None:
            return build_pretrade_state(
                date=pd.Timestamp(date),
                aum=float(ctx.tc_aum_dollars),
            )

        return build_pretrade_state(
            date=pd.Timestamp(date),
            aum=float(ctx.tc_aum_dollars),
            previous_posttrade_weights=pd.Series(saved["posttrade_weights"], dtype=float),
            realized_returns=pd.Series(saved["realized_returns"], dtype=float),
            previous_cash_weight=float(saved["cash_weight"]),
            previous_date=pd.Timestamp(saved["date"]),
        )

    # Expand the current desired block to include every nonzero legacy holding.
    @staticmethod
    def _tc_stateful_frame(
        *,
        reb_frame: pd.DataFrame,
        desired_frame: pd.DataFrame,
        pretrade_state: PreTradeState,
    ) -> pd.DataFrame:
        desired_index = pd.Index(desired_frame.index).astype(str)
        universe = build_rebalance_universe(pretrade_state.weights, desired_index)
        if len(universe) == 0:
            out = reb_frame.iloc[0:0].copy()
            out["_tc_desired"] = pd.Series(dtype=bool)
            return out

        reb = reb_frame.copy()
        reb.index = pd.Index(reb.index).astype(str)
        out = reb.reindex(universe).copy()
        out["_tc_desired"] = out.index.isin(desired_index)
        return out

    # Store the canonical post-trade result and the returns needed to drift it next period.
    def _tc_store_leg_state(
        self,
        *,
        prev_state: Dict,
        state_key: str,
        pretrade_state: PreTradeState,
        posttrade_weights: pd.Series,
        frame: pd.DataFrame,
        ret_name: str,
        transaction_cost: float = 0.0,
    ) -> RebalanceResult:
        """Persist one sleeve's state, including bounded handling of missing realized returns.

        A missing realized return is an accounting-data problem, not an optimizer
        signal.  For a nonzero legacy holding, the position is marked flat for up
        to ``tc_missing_return_max_carry_weeks`` consecutive periods.  If the
        return remains unavailable beyond that grace window, the residual
        position is removed from the *carried state* at its last marked value and
        the corresponding value is transferred to cash.  This terminal
        mark-to-cash adjustment is not treated as an executed trade and therefore
        does not create artificial turnover or transaction cost.

        Every missing-return stock-period is recorded in
        ``prev_state['tc_missing_return_events']``.  Exposure above
        ``tc_missing_return_warn_weight`` is flagged for diagnostics but does not
        abort the run.
        """
        result = build_rebalance_result(
            pretrade_state=pretrade_state,
            posttrade_weights=pd.to_numeric(posttrade_weights, errors="coerce").fillna(0.0),
        )

        transaction_cost_value = float(transaction_cost)
        if (not np.isfinite(transaction_cost_value)) or transaction_cost_value < -1e-12:
            raise ValueError("transaction_cost must be finite and non-negative.")
        transaction_cost_value = max(0.0, transaction_cost_value)

        frame_use = frame.copy()
        frame_use.index = pd.Index(frame_use.index).astype(str)
        if ret_name not in frame_use.columns:
            raise RuntimeError(
                f"{self.__class__.__name__}: missing return column {ret_name!r} while storing rebalance state."
            )

        held = result.posttrade_weights[result.posttrade_weights.abs() > 1e-12].copy()
        held.index = pd.Index(held.index).astype(str)
        realized = pd.to_numeric(
            frame_use[ret_name].reindex(result.posttrade_weights.index),
            errors="coerce",
        )
        realized.index = pd.Index(realized.index).astype(str)
        missing_held = pd.Index(held.index[realized.reindex(held.index).isna()]).astype(str)

        max_carry_weeks = int(getattr(self.ctx, "tc_missing_return_max_carry_weeks", 4))
        warn_weight = float(getattr(self.ctx, "tc_missing_return_warn_weight", 0.005))
        if max_carry_weeks < 0:
            raise ValueError("tc_missing_return_max_carry_weeks must be non-negative.")
        if (not np.isfinite(warn_weight)) or warn_weight < 0.0:
            raise ValueError("tc_missing_return_warn_weight must be finite and non-negative.")

        streak_store = prev_state.setdefault("tc_missing_return_streaks", {})
        leg_streaks = dict(streak_store.get(str(state_key), {}))
        held_ids = pd.Index(held.index).astype(str)

        # A streak only matters while the stock remains a nonzero holding in this
        # sleeve.  Observed returns reset it immediately.
        leg_streaks = {
            str(sid): int(count)
            for sid, count in leg_streaks.items()
            if str(sid) in held_ids
        }
        missing_set = set(missing_held.tolist())
        for sid in held_ids:
            sid = str(sid)
            if sid in missing_set:
                leg_streaks[sid] = int(leg_streaks.get(sid, 0)) + 1
            else:
                leg_streaks.pop(sid, None)

        # The canonical rebalance result is left untouched.  If a return remains
        # missing beyond the grace window, only the state carried into the next
        # rebalance is closed at the last marked value.
        state_weights = result.posttrade_weights.copy().astype(float)
        terminal_ids = [
            sid
            for sid in missing_held
            if int(leg_streaks.get(str(sid), 0)) > max_carry_weeks
        ]
        if terminal_ids:
            state_weights.loc[pd.Index(terminal_ids)] = 0.0

        # Missing held returns are explicitly marked flat for this holding period.
        realized = realized.fillna(0.0).astype(float)

        events = prev_state.setdefault("tc_missing_return_events", [])
        missing_weight_total = float(held.reindex(missing_held).abs().sum()) if len(missing_held) else 0.0
        materiality_warning = bool(missing_weight_total > warn_weight)
        terminal_set = set(map(str, terminal_ids))
        for sid in missing_held:
            sid = str(sid)
            abs_weight = float(abs(held.loc[sid]))
            streak = int(leg_streaks.get(sid, 0))
            terminal = sid in terminal_set
            events.append(
                {
                    "date": pd.Timestamp(pretrade_state.date),
                    "state_key": str(state_key),
                    "stock_id": sid,
                    "abs_sleeve_weight": abs_weight,
                    "missing_abs_sleeve_weight_total": missing_weight_total,
                    "consecutive_missing_periods": streak,
                    "action": "terminal_mark_to_cash" if terminal else "mark_flat",
                    "materiality_warning": materiality_warning,
                }
            )

        if materiality_warning and bool(getattr(self.ctx, "verbose", False)):
            sample = list(missing_held[:5])
            print(
                "[Missing return warning] "
                f"date={pd.Timestamp(pretrade_state.date).date()} "
                f"state={state_key} missing_abs_sleeve_weight={missing_weight_total:.3%} "
                f"n_names={len(missing_held)} sample={sample}"
            )

        # Terminally closed names must not continue accumulating a missing streak.
        for sid in terminal_ids:
            leg_streaks.pop(str(sid), None)
        streak_store[str(state_key)] = leg_streaks

        # Trading cost is paid from cash at the completed rebalance. If a
        # persistently missing holding is terminally marked to cash, transfer its
        # last marked signed value into cash without recording an artificial trade.
        terminal_transfer = (
            float(result.posttrade_weights.reindex(pd.Index(terminal_ids)).fillna(0.0).sum())
            if terminal_ids
            else 0.0
        )
        state_cash_weight = (
            float(result.posttrade_cash_weight)
            - transaction_cost_value
            + terminal_transfer
        )
        if not np.isfinite(state_cash_weight):
            raise RuntimeError("Missing-return state adjustment produced a non-finite cash weight.")

        store = self._tc_leg_state_store(prev_state)
        store[str(state_key)] = {
            "date": pd.Timestamp(pretrade_state.date),
            "posttrade_weights": state_weights.copy(),
            "cash_weight": float(state_cash_weight),
            "realized_returns": realized.copy(),
        }
        prev_state.setdefault("tc_rebalance_results", {})[str(state_key)] = result
        return result

    # Build an alpha vector that rewards only names in the current desired set.
    @staticmethod
    def _tc_desired_alpha(
        *,
        frame: pd.DataFrame,
        desired_alpha: pd.Series,
    ) -> pd.Series:
        out = pd.Series(0.0, index=frame.index, dtype=float)
        vals = pd.to_numeric(desired_alpha, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        common = out.index.intersection(pd.Index(vals.index).astype(str))
        if len(common) > 0:
            vals = vals.copy()
            vals.index = pd.Index(vals.index).astype(str)
            out.loc[common] = vals.reindex(common).fillna(0.0).astype(float)
        return out

    # Split a block into tradable names and frozen legacy holdings for this rebalance.
    def _tc_split_block(
        self,
        *,
        ctx,
        frame: pd.DataFrame,
        prev_w: pd.Series,
        use_abs_prev: bool = False,
        long_only: bool = True,
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.Index]:
        prev_full = pd.to_numeric(prev_w.reindex(frame.index), errors="coerce").fillna(0.0)
        if use_abs_prev:
            prev_full = prev_full.abs()

        tradable_mask = self._tc_tradable_mask(ctx=ctx, frame=frame)
        tradable_idx = frame.index[tradable_mask]
        frozen_idx = frame.index[~tradable_mask]

        frozen_w = prev_full.reindex(frozen_idx).fillna(0.0)
        if long_only:
            frozen_w = frozen_w.clip(lower=0.0)

        tradable_frame = frame.loc[tradable_idx].copy()
        return prev_full, frozen_w, tradable_frame, tradable_idx

    # Compute the feasible long-only gross interval for one transaction-cost sleeve.
    def _tc_feasible_interval(
        self,
        *,
        ctx,
        frame: pd.DataFrame,
        prev_w: pd.Series,
        use_abs_prev: bool = False,
    ) -> Tuple[float, float]:
        from cost_aware_portfolio.optimizer import compute_long_only_feasible_interval

        if len(frame) == 0:
            return 0.0, 0.0

        prev_full, frozen_w, tradable_frame, tradable_idx = self._tc_split_block(
            ctx=ctx,
            frame=frame,
            prev_w=prev_w,
            use_abs_prev=use_abs_prev,
            long_only=True,
        )
        frozen_sum = float(frozen_w.sum())

        if len(tradable_frame) == 0:
            return frozen_sum, frozen_sum

        adv = ctx._tc_adv_series(tradable_idx, tradable_frame)
        prev_w_sub = prev_full.reindex(tradable_idx).fillna(0.0)

        params = self._build_tc_params(
            ctx,
            long_only=True,
            dollar_neutral=False,
            gross_target=1.0,
        )
        lo, hi = compute_long_only_feasible_interval(
            w_prev=prev_w_sub,
            adv=adv,
            aum=ctx.tc_aum_dollars,
            params=params,
        )
        return frozen_sum + float(lo), frozen_sum + float(hi)


    # Compute a common feasible long-only target sum for paired top and bottom sleeves when possible.
    def _paired_common_long_only_sum(
        self,
        *,
        ctx,
        low_frame: pd.DataFrame,
        high_frame: pd.DataFrame,
        prev_w_low: pd.Series,
        prev_w_high: pd.Series,
        use_abs_low_prev: bool = False,
    ) -> Optional[float]:
        low_min, low_max = self._tc_feasible_interval(
            ctx=ctx,
            frame=low_frame,
            prev_w=prev_w_low,
            use_abs_prev=use_abs_low_prev,
        )
        high_min, high_max = self._tc_feasible_interval(
            ctx=ctx,
            frame=high_frame,
            prev_w=prev_w_high,
            use_abs_prev=False,
        )

        common_low = max(float(low_min), float(high_min), 0.0)
        common_high = min(1.0, float(low_max), float(high_max))
        if common_high + 1e-12 < common_low:
            return None
        return float(common_high)


    # Return the diagnostic spread series in bps for one sleeve frame.
    def _tc_diag_spread_bps(self, *, frame: pd.DataFrame) -> pd.Series:
        return PortfolioDiagnostics.tc_diag_spread_bps(frame=frame)

    # Return the diagnostic ADV series in millions of dollars for one sleeve frame.
    def _tc_diag_adv_musd(self, *, ctx, frame: pd.DataFrame) -> pd.Series:
        return PortfolioDiagnostics.tc_diag_adv_musd(ctx=ctx, frame=frame)

    # Return the diagnostic sigma series using the same convention as the optimizer.
    def _tc_diag_sigma(self, *, ctx, frame: pd.DataFrame) -> pd.Series:
        return PortfolioDiagnostics.tc_diag_sigma(ctx=ctx, frame=frame)

    # Compute a NaN-safe mean over the tradable subset of a sleeve frame.
    @staticmethod
    def _tc_masked_mean(series: pd.Series, mask: pd.Series) -> float:
        return PortfolioDiagnostics.tc_masked_mean(series, mask)

    # Append a paired top/bottom sleeve diagnostic row without changing portfolio logic.
    def _append_paired_tc_debug_row(
        self,
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
    ) -> None:
        low_trad_mask = self._tc_tradable_mask(ctx=ctx, frame=low_frame) if len(low_frame) else pd.Series(False, index=low_frame.index, dtype=bool)
        high_trad_mask = self._tc_tradable_mask(ctx=ctx, frame=high_frame) if len(high_frame) else pd.Series(False, index=high_frame.index, dtype=bool)
        PortfolioDiagnostics.append_paired_tc_debug_row(
            ctx=ctx,
            prev_state=prev_state,
            date=date,
            common_target_sum=common_target_sum,
            low_frame=low_frame,
            high_frame=high_frame,
            low_weights_abs=low_weights_abs,
            high_weights=high_weights,
            low_cost=low_cost,
            high_cost=high_cost,
            low_trad_mask=low_trad_mask,
            high_trad_mask=high_trad_mask,
        )

    # Compute additive realized trading cost by name using the same ex-post formulas as the optimizer helper.
    def _tc_trade_cost_series(
        self,
        *,
        ctx,
        w: pd.Series,
        prev_w: pd.Series,
        frame: pd.DataFrame,
    ) -> pd.Series:
        idx = pd.Index(frame.index.astype(str))
        w_cur = pd.to_numeric(w.reindex(idx), errors="coerce").fillna(0.0)
        w_old = pd.to_numeric(prev_w.reindex(idx), errors="coerce").fillna(0.0)
        spreads = ctx._tc_spreads_vector(idx, frame)
        sigma = ctx._tc_sigma_series(idx, frame)
        adv = ctx._tc_adv_series(idx, frame)

        delta = (w_cur - w_old).abs()
        linear = spreads * delta
        execution_volume = adv.to_numpy(dtype=float) * float(getattr(ctx, "tc_execution_days", 1.0))
        coeff = pd.Series(
            ctx.tc_impact_kappa
            * sigma.to_numpy(dtype=float)
            * np.sqrt(float(ctx.tc_aum_dollars) / execution_volume),
            index=idx,
            dtype=float,
        )
        impact = coeff * np.power(delta.to_numpy(dtype=float), float(ctx.tc_impact_exponent))
        return (linear + impact).astype(float)

    # Solve one alpha-driven nonlinear optimizer block and return aligned weights plus realized costs.
    def _solve_tc_block(
        self,
        *,
        ctx,
        frame: pd.DataFrame,
        alpha: pd.Series,
        prev_w: pd.Series,
        Sigma: Optional[np.ndarray],
        cols: Optional[List[str]],
        long_only: bool,
        dollar_neutral: bool,
        gross_target: float,
        use_abs_prev: bool = False,
        desired_target_sum: Optional[float] = None,
        shrink: bool = False,
    ) -> Tuple[pd.Series, float]:
        from cost_aware_portfolio.optimizer import (
            compute_trade_cost_breakdown,
            solve_cost_aware_nonlinear_pg,
        )

        cols_req = self._require_covariance(
            strategy_name=self.__class__.__name__,
            Sigma=Sigma,
            cols=cols,
        )
        prev_full, frozen_w, tradable_frame_all, tradable_idx_all = self._tc_split_block(
            ctx=ctx,
            frame=frame,
            prev_w=prev_w,
            use_abs_prev=use_abs_prev,
            long_only=long_only,
        )

        out = pd.Series(0.0, index=frame.index, dtype=float)
        if len(frozen_w) > 0:
            out.loc[frozen_w.index] = frozen_w.astype(float)

        tradable_cols = [c for c in cols_req if c in tradable_frame_all.index]
        cov_excluded = pd.Index(tradable_idx_all).difference(pd.Index(tradable_cols))
        if len(cov_excluded) > 0:
            keep_prev = prev_full.reindex(cov_excluded).fillna(0.0)
            if long_only:
                keep_prev = keep_prev.clip(lower=0.0)
            out.loc[cov_excluded] = keep_prev.astype(float)

        if len(tradable_cols) == 0:
            return out, 0.0

        frozen_effective = pd.concat([
            frozen_w.astype(float),
            out.reindex(cov_excluded).fillna(0.0).astype(float),
        ])
        frozen_effective = frozen_effective[~frozen_effective.index.duplicated(keep="last")]
        frozen_abs = float(np.abs(frozen_effective).sum())
        gross_target_sub = max(0.0, float(gross_target) - frozen_abs)

        desired_target_sum_sub = desired_target_sum
        if desired_target_sum_sub is not None and long_only:
            desired_target_sum_sub = max(0.0, float(desired_target_sum_sub) - float(frozen_effective.sum()))

        if gross_target_sub <= 1e-12:
            return out, 0.0
        if long_only and desired_target_sum_sub is not None and desired_target_sum_sub <= 1e-12:
            return out, 0.0

        # If too few tradable covariance names remain, keep current holdings and do not force a trade.
        if len(tradable_cols) < 2:
            keep_prev = prev_full.reindex(tradable_cols).fillna(0.0)
            if long_only:
                keep_prev = keep_prev.clip(lower=0.0)
            out.loc[tradable_cols] = keep_prev.astype(float)
            return out, 0.0

        sub = tradable_frame_all.loc[tradable_cols].copy()
        loc_map = {str(c): i for i, c in enumerate(cols_req)}
        locs = [loc_map[str(c)] for c in tradable_cols]
        Sigma_sub = np.asarray(Sigma, dtype=float)[np.ix_(locs, locs)]
        if shrink:
            lam = ctx.mw_shrinkage
            if lam is None and ctx.mw_auto_shrinkage:
                lam = ctx._auto_lambda(ctx.mw_window, len(tradable_cols))
            lam = 0.5 if lam is None else lam
            lam = min(max(lam, 0.0), 1.0)
            Sigma_use = ctx._apply_shrinkage(Sigma_sub, lam)
        else:
            Sigma_use = ctx._apply_shrinkage(Sigma_sub, None)

        alpha_sub = pd.to_numeric(alpha.reindex(sub.index), errors="coerce")
        if alpha_sub.isna().any():
            raise RuntimeError(
                f"{self.__class__.__name__}: alpha contains missing values after alignment."
            )
        if float(alpha_sub.abs().sum()) <= 0.0:
            out.loc[sub.index] = prev_full.reindex(sub.index).fillna(0.0).clip(lower=0.0 if long_only else None)
            return out, 0.0

        adv = ctx._tc_adv_series(sub.index, sub)
        sigma = ctx._tc_sigma_series(sub.index, sub)
        spreads = ctx._tc_spreads_vector(sub.index, sub)
        w_prev_sub = pd.to_numeric(prev_full.reindex(sub.index), errors="coerce").fillna(0.0)
        if long_only:
            w_prev_sub = w_prev_sub.clip(lower=0.0)

        params = self._build_tc_params(
            ctx,
            long_only=long_only,
            dollar_neutral=dollar_neutral,
            gross_target=gross_target_sub,
        )

        try:
            w_opt = solve_cost_aware_nonlinear_pg(
                alpha=alpha_sub,
                Sigma=Sigma_use,
                w_prev=w_prev_sub,
                adv=adv,
                sigma=sigma,
                aum=ctx.tc_aum_dollars,
                spreads_oneway=spreads,
                params=params,
                desired_target_sum=desired_target_sum_sub,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{self.__class__.__name__}: nonlinear Markowitz TC solve failed."
            ) from exc

        breakdown = compute_trade_cost_breakdown(
            w=w_opt,
            w_prev=w_prev_sub,
            spreads_oneway=spreads,
            sigma=sigma,
            adv=adv,
            aum=ctx.tc_aum_dollars,
            impact_kappa=ctx.tc_impact_kappa,
            impact_exponent=ctx.tc_impact_exponent,
            execution_days=float(getattr(ctx, "tc_execution_days", 1.0)),
        )
        out.loc[sub.index] = w_opt.reindex(sub.index).fillna(0.0).astype(float)
        return out.reindex(frame.index).fillna(0.0), float(breakdown.total_cost)

    # Solve a nonlinear rebalance toward a fixed target weight vector and return realized costs.
    def _solve_tc_target_tracking(
        self,
        *,
        ctx,
        frame: pd.DataFrame,
        target_w: pd.Series,
        prev_w: pd.Series,
        desired_target_sum: Optional[float] = None,
    ) -> Tuple[pd.Series, float]:
        from cost_aware_portfolio.optimizer import (
            compute_trade_cost_breakdown,
            solve_rebalance_to_target_nonlinear_pg,
        )

        prev_full, frozen_w, tradable_frame, tradable_idx = self._tc_split_block(
            ctx=ctx,
            frame=frame,
            prev_w=prev_w,
            use_abs_prev=False,
            long_only=True,
        )
        out = pd.Series(0.0, index=frame.index, dtype=float)
        if len(frozen_w) > 0:
            out.loc[frozen_w.index] = frozen_w.astype(float)

        if len(tradable_frame) == 0:
            return out, 0.0

        target_full = pd.to_numeric(target_w.reindex(frame.index), errors="coerce").fillna(0.0).clip(lower=0.0)
        prev_w_sub = prev_full.reindex(tradable_idx).fillna(0.0).clip(lower=0.0)
        target_sub_raw = target_full.reindex(tradable_idx).fillna(0.0).clip(lower=0.0)

        total_target_sum = float(desired_target_sum) if (desired_target_sum is not None) else float(target_full.sum())
        tradable_target_sum = max(0.0, total_target_sum - float(frozen_w.sum()))
        if tradable_target_sum <= 1e-12:
            return out, 0.0

        raw_sum = float(target_sub_raw.sum())
        if raw_sum > 0.0 and np.isfinite(raw_sum):
            target_sub = target_sub_raw * (tradable_target_sum / raw_sum)
        else:
            target_sub = pd.Series(
                float(tradable_target_sum) / max(len(tradable_frame), 1),
                index=tradable_frame.index,
                dtype=float,
            )

        adv = ctx._tc_adv_series(tradable_frame.index, tradable_frame)
        sigma = ctx._tc_sigma_series(tradable_frame.index, tradable_frame)
        spreads = ctx._tc_spreads_vector(tradable_frame.index, tradable_frame)

        params = self._build_tc_params(
            ctx,
            long_only=True,
            dollar_neutral=False,
            gross_target=max(0.0, tradable_target_sum),
        )

        w_opt = solve_rebalance_to_target_nonlinear_pg(
            w_target=target_sub,
            w_prev=prev_w_sub,
            adv=adv,
            sigma=sigma,
            aum=ctx.tc_aum_dollars,
            spreads_oneway=spreads,
            params=params,
            desired_target_sum=tradable_target_sum,
        )
        breakdown = compute_trade_cost_breakdown(
            w=w_opt,
            w_prev=prev_w_sub,
            spreads_oneway=spreads,
            sigma=sigma,
            adv=adv,
            aum=ctx.tc_aum_dollars,
            impact_kappa=ctx.tc_impact_kappa,
            impact_exponent=ctx.tc_impact_exponent,
            execution_days=float(getattr(ctx, "tc_execution_days", 1.0)),
        )
        out.loc[tradable_frame.index] = w_opt.reindex(tradable_frame.index).fillna(0.0).astype(float)
        return out.reindex(frame.index).fillna(0.0), float(breakdown.total_cost)


class EWStrategy(PortfolioStrategy):
    """Equal-weight within each decile."""

    # Build the equal-weight portfolio outputs for one rebalance date.
    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        reb = df.loc[date].copy()
        row = np.zeros(cut, dtype=float)
        kept: Dict[int, pd.DataFrame] = {}

        for j in range(cut):
            up = reb["up_prob"].astype(float)
            lo, hi = self.ctx._decile_bounds(up, cut, j)
            mask = self.ctx._decile_mask(up, lo, hi, is_lowest=(j == 0))
            ddf = reb[mask].copy()

            if len(ddf) == 0:
                ddf = pd.DataFrame(index=[])
                ddf["weight"] = 0.0
                ddf[ret_name] = 0.0
                ddf["inv_ret"] = 0.0
            else:
                ddf["weight"] = 1.0 / max(len(ddf), 1)
                ddf["inv_ret"] = ddf["weight"] * ddf[ret_name]

            if self.ctx.transaction_cost:
                fee = ddf.get("transaction_fee", 0.0)
                if j == cut - 1:
                    ddf["inv_ret"] -= ddf["weight"] * fee * 2
                elif j == 0:
                    ddf["inv_ret"] += ddf["weight"] * fee * 2

            kept[j] = ddf
            row[j] = float(ddf["inv_ret"].sum())

        sell, buy = kept[0].copy(), kept[cut - 1].copy()
        sell[["weight", "inv_ret"]] *= -1
        to_df = pd.concat([sell, buy], axis=0)
        diag = pd.concat([kept[0], kept[cut - 1]], axis=0)[["up_prob", "inv_ret"]]
        return row, to_df, diag


class VWStrategy(PortfolioStrategy):
    """Value-weight within each decile using market capitalization."""

    # Build the value-weight portfolio outputs for one rebalance date.
    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        reb = df.loc[date].copy()
        row = np.zeros(cut, dtype=float)
        kept: Dict[int, pd.DataFrame] = {}

        for j in range(cut):
            up = reb["up_prob"].astype(float)
            lo, hi = self.ctx._decile_bounds(up, cut, j)
            mask = self.ctx._decile_mask(up, lo, hi, is_lowest=(j == 0))
            ddf = reb[mask].copy()

            if len(ddf):
                ddf["weight"] = self._strict_value_weights(
                    ddf,
                    date=pd.Timestamp(date),
                    label=f"VW decile={j}",
                )
            else:
                ddf["weight"] = 0.0

            ddf["inv_ret"] = ddf["weight"] * ddf[ret_name]

            if self.ctx.transaction_cost:
                fee = ddf.get("transaction_fee", 0.0)
                if j == cut - 1:
                    ddf["inv_ret"] -= ddf["weight"] * fee * 2
                elif j == 0:
                    ddf["inv_ret"] += ddf["weight"] * fee * 2

            kept[j] = ddf
            row[j] = float(ddf["inv_ret"].sum())

        sell, buy = kept[0].copy(), kept[cut - 1].copy()
        sell[["weight", "inv_ret"]] *= -1
        to_df = pd.concat([sell, buy], axis=0)
        diag = pd.concat([kept[0], kept[cut - 1]], axis=0)[["up_prob", "inv_ret"]]
        return row, to_df, diag


class MWPerDecileStrategy(PortfolioStrategy):
    """Per-decile Markowitz portfolio with optional shrinkage."""

    # Initialize the strategy and record whether shrinkage is enabled.
    def __init__(self, ctx, shrink: bool):
        super().__init__(ctx)
        self.shrink = bool(shrink)

    # Solve one decile under the Markowitz optimization logic.
    def _solve_one_decile(self, df, date, ret_name, ddf, decile_idx):
        ctx = self.ctx
        if len(ddf) == 0:
            out = ddf.copy()
            out["weight"] = 0.0
            out["inv_ret"] = 0.0
            return out

        _, Sigma, cols = self._eligible_cov_block(
            df,
            date,
            ret_name,
            ddf,
            leg_name=f"decile={decile_idx}",
        )

        alpha_sign = -1.0 if (ctx.mw_invert_low_decile and decile_idx == 0) else 1.0
        sub = ddf.loc[cols]
        a = alpha_sign * ctx._normalize_alpha(sub["up_prob"]).to_numpy(dtype=float)
        if self.shrink:
            lam = ctx.mw_shrinkage
            if lam is None and ctx.mw_auto_shrinkage:
                lam = ctx._auto_lambda(ctx.mw_window, len(cols))
            lam = 0.5 if lam is None else lam
            Sigma_use = ctx._apply_shrinkage(Sigma, lam)
        else:
            Sigma_use = ctx._apply_shrinkage(Sigma, None)

        try:
            w_vec, _, _ = ctx._solve_sigma(Sigma_use, a)
            w = pd.Series(w_vec, index=sub.index)
        except Exception as exc:
            raise RuntimeError(
                f"MWPerDecileStrategy: linear solve failed for date={date} decile={decile_idx}."
            ) from exc

        if ctx.mw_long_only:
            w = w.clip(lower=0.0)
        total = float(np.nansum(w))
        if total <= 0 or not np.isfinite(total):
            raise RuntimeError(
                f"MWPerDecileStrategy: non-finite or zero weight sum for date={date} decile={decile_idx}."
            )
        w = w / total

        out = ddf.copy()
        out["weight"] = out.index.to_series().astype(str).map(pd.Series(w, dtype=float)).fillna(0.0).astype(float)
        out["inv_ret"] = out["weight"] * out[ret_name]
        return out

    # Build all deciles for one rebalance date under per-decile Markowitz.
    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        reb = df.loc[date].copy()
        row = np.zeros(cut, dtype=float)
        kept: Dict[int, pd.DataFrame] = {}

        for j in range(cut):
            up = reb["up_prob"].astype(float)
            lo, hi = self.ctx._decile_bounds(up, cut, j)
            mask = self.ctx._decile_mask(up, lo, hi, is_lowest=(j == 0))
            ddf = reb[mask].copy()
            dec = self._solve_one_decile(df, date, ret_name, ddf, j)

            if self.ctx.transaction_cost:
                fee = dec.get("transaction_fee", 0.0)
                if j == cut - 1:
                    dec["inv_ret"] -= dec["weight"] * fee * 2
                elif j == 0:
                    dec["inv_ret"] += dec["weight"] * fee * 2

            kept[j] = dec
            row[j] = float(dec["inv_ret"].sum())

        sell, buy = kept[0].copy(), kept[cut - 1].copy()
        both = pd.concat([kept[0], kept[cut - 1]], axis=0)[["up_prob", "inv_ret"]]
        sell[["weight", "inv_ret"]] *= -1
        to_df = pd.concat([sell, buy], axis=0)
        return row, to_df, both


class MWSpreadStrategy(PortfolioStrategy):
    """Joint Markowitz on the union of the low and high deciles."""

    # Initialize the strategy and record whether shrinkage is enabled.
    def __init__(self, ctx, shrink: bool):
        super().__init__(ctx)
        self.shrink = bool(shrink)

    # Build the joint low-high spread portfolio for one rebalance date.
    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        ctx = self.ctx
        reb = df.loc[date].copy()
        row = np.zeros(cut, dtype=float)

        q10, q90 = self._require_tail_thresholds(
            reb,
            date=pd.Timestamp(date),
        )

        low_mask = reb["up_prob"] <= q10
        high_mask = reb["up_prob"] >= q90
        union = reb[low_mask | high_mask].copy()
        if union.empty:
            to_df = pd.DataFrame(columns=["weight", ret_name, "inv_ret"])
            return row, to_df, None

        z = (union["up_prob"] - union["up_prob"].min()) / (union["up_prob"].max() - union["up_prob"].min() + 1e-12)
        alpha = pd.Series(0.0, index=union.index)
        alpha.loc[high_mask] = z.loc[high_mask]
        alpha.loc[low_mask] = -(1 - z.loc[low_mask])

        _, Sigma, cols = self._eligible_cov_block(
            df,
            date,
            ret_name,
            union,
            leg_name="low/high union",
        )

        sub = union.loc[cols]
        a = alpha.loc[cols].to_numpy(dtype=float)
        if self.shrink:
            lam = ctx.mw_shrinkage
            if lam is None and ctx.mw_auto_shrinkage:
                lam = ctx._auto_lambda(ctx.mw_window, len(cols))
            lam = 0.5 if lam is None else lam
            lam = min(max(lam, 0.0), 1.0)
            Sigma_use = ctx._apply_shrinkage(Sigma, lam)
        else:
            Sigma_use = ctx._apply_shrinkage(Sigma, None)

        try:
            ones = np.ones(len(cols))
            w_vec, z_vec, _ = ctx._solve_sigma(Sigma_use, a, ones=ones)
            den = float(ones @ z_vec)
            if abs(den) > 1e-12:
                w_vec = w_vec - (float(ones @ w_vec) / den) * z_vec
        except Exception as exc:
            raise RuntimeError(f"MWSpreadStrategy: linear solve failed for date={date}.") from exc

        w = pd.Series(w_vec, index=sub.index)
        w = ctx._safe_reindex(w, union.index, 0.0)
        w = ctx._scale_to_gross(w, ctx.mw_spread_gross)

        low_ids = union.index[low_mask.loc[union.index]]
        high_ids = union.index[high_mask.loc[union.index]]

        low_rep = pd.DataFrame(index=low_ids)
        high_rep = pd.DataFrame(index=high_ids)
        if len(low_ids):
            low_abs = pd.to_numeric(w.reindex(low_ids), errors="coerce").fillna(0.0).abs()
            low_tot = float(low_abs.sum())
            low_rep["weight"] = (low_abs / low_tot) if (low_tot > 0 and np.isfinite(low_tot)) else 0.0
            low_rep[ret_name] = union.loc[low_ids, ret_name]
            low_rep["up_prob"] = union.loc[low_ids, "up_prob"]
            low_rep["inv_ret"] = low_rep["weight"] * pd.to_numeric(low_rep[ret_name], errors="coerce").fillna(0.0)
        else:
            low_rep = pd.DataFrame(columns=["weight", ret_name, "up_prob", "inv_ret"])

        if len(high_ids):
            high_abs = pd.to_numeric(w.reindex(high_ids), errors="coerce").fillna(0.0).abs()
            high_tot = float(high_abs.sum())
            high_rep["weight"] = (high_abs / high_tot) if (high_tot > 0 and np.isfinite(high_tot)) else 0.0
            high_rep[ret_name] = union.loc[high_ids, ret_name]
            high_rep["up_prob"] = union.loc[high_ids, "up_prob"]
            high_rep["inv_ret"] = high_rep["weight"] * pd.to_numeric(high_rep[ret_name], errors="coerce").fillna(0.0)
        else:
            high_rep = pd.DataFrame(columns=["weight", ret_name, "up_prob", "inv_ret"])

        if ctx.transaction_cost:
            fee_low = union.loc[low_ids].get("transaction_fee", 0.0)
            fee_high = union.loc[high_ids].get("transaction_fee", 0.0)
            if len(high_rep) and np.ndim(fee_high) != 0:
                high_fee = np.sum(np.abs(w.reindex(high_ids).fillna(0.0))) * float(np.nanmean(fee_high)) * 2
                high_rep.loc[high_rep.index[0], "inv_ret"] = high_rep.loc[high_rep.index[0], "inv_ret"] - float(high_fee)
            if len(low_rep) and np.ndim(fee_low) != 0:
                low_fee = np.sum(np.abs(w.reindex(low_ids).fillna(0.0))) * float(np.nanmean(fee_low)) * 2
                low_rep.loc[low_rep.index[0], "inv_ret"] = low_rep.loc[low_rep.index[0], "inv_ret"] + float(low_fee)

        row[0] = float(low_rep["inv_ret"].sum()) if len(low_rep) else 0.0
        row[cut - 1] = float(high_rep["inv_ret"].sum()) if len(high_rep) else 0.0

        to_df = pd.DataFrame(index=union.index)
        to_df["weight"] = w
        to_df[ret_name] = union[ret_name]
        to_df["inv_ret"] = w * union[ret_name]
        diag = pd.concat([low_rep[["up_prob", "inv_ret"]], high_rep[["up_prob", "inv_ret"]]], axis=0)
        return row, to_df, diag


class MWSpreadTCStrategy(TCOptimizerMixin, PortfolioStrategy):
    """Joint low-high spread solved on the union block with the nonlinear cost model."""

    def __init__(self, ctx, shrink: bool):
        super().__init__(ctx)
        self.shrink = bool(shrink)

    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        ctx = self.ctx
        reb = df.loc[date].copy()
        reb.index = pd.Index(reb.index).astype(str)
        row = np.zeros(cut, dtype=float)

        q10, q90 = self._require_tail_thresholds(
            reb,
            date=pd.Timestamp(date),
        )

        low_mask = reb["up_prob"] <= q10
        high_mask = reb["up_prob"] >= q90
        desired_union = reb[low_mask | high_mask].copy()

        pretrade = self._tc_pretrade_state(
            ctx=ctx,
            prev_state=prev_state,
            state_key="mw_spread_union",
            date=date,
        )
        union = self._tc_stateful_frame(
            reb_frame=self._tc_execution_support_frame(ctx=ctx, date=date, selection_frame=reb),
            desired_frame=desired_union,
            pretrade_state=pretrade,
        )
        if union.empty:
            to_df = pd.DataFrame(columns=["weight", ret_name, "inv_ret"])
            return row, to_df, None

        desired_low_ids = pd.Index(reb.index[low_mask]).astype(str)
        desired_high_ids = pd.Index(reb.index[high_mask]).astype(str)
        desired_ids = desired_low_ids.union(desired_high_ids)

        desired_probs = pd.to_numeric(reb["up_prob"].reindex(desired_ids), errors="coerce")
        pmin = float(desired_probs.min())
        pmax = float(desired_probs.max())
        z = (desired_probs - pmin) / (pmax - pmin + 1e-12)
        desired_alpha = pd.Series(0.0, index=desired_ids, dtype=float)
        desired_alpha.loc[desired_high_ids.intersection(desired_alpha.index)] = z.reindex(desired_high_ids).fillna(0.0)
        desired_alpha.loc[desired_low_ids.intersection(desired_alpha.index)] = -(1.0 - z.reindex(desired_low_ids).fillna(0.0))
        alpha = self._tc_desired_alpha(frame=union, desired_alpha=desired_alpha)

        _, Sigma, cols = self._eligible_cov_block(
            df,
            date,
            ret_name,
            union,
            leg_name="stateful low/high union",
        )

        prev_w_union = pretrade.aligned_weights(union.index)
        w_signed, total_cost = self._solve_tc_block(
            ctx=ctx,
            frame=union,
            alpha=alpha,
            prev_w=prev_w_union,
            Sigma=Sigma,
            cols=cols,
            long_only=False,
            dollar_neutral=True,
            gross_target=ctx.mw_spread_gross,
            shrink=self.shrink,
        )
        w_signed = pd.to_numeric(w_signed.reindex(union.index), errors="coerce").fillna(0.0)

        result = self._tc_store_leg_state(
            prev_state=prev_state,
            state_key="mw_spread_union",
            pretrade_state=pretrade,
            posttrade_weights=w_signed,
            frame=union,
            ret_name=ret_name,
            transaction_cost=float(total_cost),
        )
        prev_state["tc_signed_rebalance_result"] = result

        # Realized costs are computed from the same authoritative pre/post transition.
        tradable_mask = self._tc_tradable_mask(ctx=ctx, frame=union)
        tradable_idx = union.index[tradable_mask]
        if len(tradable_idx) > 0:
            cost_by_name = self._tc_trade_cost_series(
                ctx=ctx,
                w=result.posttrade_weights.reindex(tradable_idx).fillna(0.0),
                prev_w=result.pretrade_weights.reindex(tradable_idx).fillna(0.0),
                frame=union.loc[tradable_idx],
            ).reindex(union.index).fillna(0.0)
        else:
            cost_by_name = pd.Series(0.0, index=union.index, dtype=float)

        post = result.posttrade_weights.reindex(union.index).fillna(0.0)
        pre = result.pretrade_weights.reindex(union.index).fillna(0.0)
        side_sign = np.sign(post)
        side_sign = side_sign.where(side_sign != 0.0, np.sign(pre))
        low_cost = float(cost_by_name.loc[side_sign < 0.0].sum())
        high_cost = float(cost_by_name.loc[side_sign > 0.0].sum())

        low_ids = post.index[post < -1e-12]
        high_ids = post.index[post > 1e-12]

        low_rep = pd.DataFrame(index=low_ids)
        if len(low_ids):
            low_abs = post.reindex(low_ids).abs()
            low_tot = float(low_abs.sum())
            low_rep["weight"] = (low_abs / low_tot) if low_tot > 0.0 else 0.0
            low_rep[ret_name] = union.loc[low_ids, ret_name]
            low_rep["up_prob"] = union.loc[low_ids, "up_prob"]
            low_rep["inv_ret"] = low_rep["weight"] * pd.to_numeric(low_rep[ret_name], errors="coerce").fillna(0.0)
        else:
            low_rep = pd.DataFrame(columns=["weight", ret_name, "up_prob", "inv_ret"])

        high_rep = pd.DataFrame(index=high_ids)
        if len(high_ids):
            high_abs = post.reindex(high_ids).abs()
            high_tot = float(high_abs.sum())
            high_rep["weight"] = (high_abs / high_tot) if high_tot > 0.0 else 0.0
            high_rep[ret_name] = union.loc[high_ids, ret_name]
            high_rep["up_prob"] = union.loc[high_ids, "up_prob"]
            high_rep["inv_ret"] = high_rep["weight"] * pd.to_numeric(high_rep[ret_name], errors="coerce").fillna(0.0)
        else:
            high_rep = pd.DataFrame(columns=["weight", ret_name, "up_prob", "inv_ret"])

        low_return = float(low_rep["inv_ret"].sum()) if len(low_rep) else 0.0
        high_return = float(high_rep["inv_ret"].sum()) if len(high_rep) else 0.0
        row[0] = low_return + low_cost
        row[cut - 1] = high_return - high_cost

        self._append_paired_tc_debug_row(
            ctx=ctx,
            prev_state=prev_state,
            date=date,
            common_target_sum=float("nan"),
            low_frame=union.reindex(low_ids).copy(),
            high_frame=union.reindex(high_ids).copy(),
            low_weights_abs=post.reindex(low_ids).abs(),
            high_weights=post.reindex(high_ids).abs(),
            low_cost=low_cost,
            high_cost=high_cost,
        )

        to_df = pd.DataFrame(index=union.index)
        to_df["weight"] = post
        to_df[ret_name] = union[ret_name]
        to_df["inv_ret"] = to_df["weight"] * pd.to_numeric(to_df[ret_name], errors="coerce").fillna(0.0)
        if len(to_df) > 0:
            to_df.loc[to_df.index[0], "inv_ret"] = to_df.loc[to_df.index[0], "inv_ret"] - float(cost_by_name.sum())

        diag = pd.concat([low_rep[["up_prob", "inv_ret"]], high_rep[["up_prob", "inv_ret"]]], axis=0)
        return row, to_df, diag


class MWHPlusLowNegStrategy(PortfolioStrategy):
    """Two independent Markowitz legs: long high decile and short low decile."""

    # Build the High-plus-LowNeg portfolio for one rebalance date.
    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        ctx = self.ctx
        reb = df.loc[date].copy()
        row = np.zeros(cut, dtype=float)

        q10, q90 = self._require_tail_thresholds(
            reb,
            date=pd.Timestamp(date),
        )

        low_mask = reb["up_prob"] <= q10
        high_mask = reb["up_prob"] >= q90
        low_ids = reb.index[low_mask]
        high_ids = reb.index[high_mask]

        _, SigmaH, colsH = self._eligible_cov_block(
            df,
            date,
            ret_name,
            reb.loc[high_ids].copy(),
            leg_name="high-decile",
        )
        subH = reb.loc[colsH]
        aH = ctx._normalize_alpha(subH["up_prob"]).to_numpy(dtype=float)
        try:
            wH_vec, _, _ = ctx._solve_sigma(ctx._apply_shrinkage(SigmaH, None), aH)
            wH = pd.Series(wH_vec, index=subH.index)
        except Exception as exc:
            raise RuntimeError(f"MWHPlusLowNegStrategy: high-decile solve failed for date={date}.") from exc
        wH = wH.clip(lower=0.0)
        sH = float(np.nansum(wH))
        wH = (wH / sH).reindex(high_ids).fillna(0.0) if (sH > 0 and np.isfinite(sH)) else pd.Series(0.0, index=high_ids)

        _, SigmaL, colsL = self._eligible_cov_block(
            df,
            date,
            ret_name,
            reb.loc[low_ids].copy(),
            leg_name="low-decile",
        )
        subL = reb.loc[colsL]
        aL0 = ctx._normalize_alpha(subL["up_prob"]).astype(float)
        rng = float(aL0.max() - aL0.min())
        z = (aL0 - aL0.min()) / (rng if rng > 1e-12 else 1.0)
        aL = -(1.0 - z).to_numpy(dtype=float)
        try:
            wL_raw_vec, _, _ = ctx._solve_sigma(ctx._apply_shrinkage(SigmaL, None), aL)
            wL_raw = pd.Series(wL_raw_vec, index=subL.index)
        except Exception as exc:
            raise RuntimeError(f"MWHPlusLowNegStrategy: low-decile solve failed for date={date}.") from exc

        gL = float(np.nansum(np.abs(wL_raw)))
        if gL <= 0 or not np.isfinite(gL):
            raise RuntimeError(
                f"MWHPlusLowNegStrategy: low-decile gross exposure is non-finite or zero for date={date}."
            )
        wL_short = -(np.abs(wL_raw) / gL).reindex(low_ids).fillna(0.0)

        wH_abs_sum = float(np.abs(wH).sum())
        wL_abs_sum = float(np.abs(wL_raw).sum())
        if wH_abs_sum <= 0 or not np.isfinite(wH_abs_sum):
            raise RuntimeError(
                f"MWHPlusLowNegStrategy: high-decile gross exposure is non-finite or zero for date={date}."
            )
        if wL_abs_sum <= 0 or not np.isfinite(wL_abs_sum):
            raise RuntimeError(
                f"MWHPlusLowNegStrategy: low-decile gross exposure is non-finite or zero for date={date}."
            )

        low_rep = pd.DataFrame(index=low_ids)
        low_rep["weight"] = pd.to_numeric(wL_raw.reindex(low_ids), errors="coerce").fillna(0.0).abs() / wL_abs_sum
        low_rep[ret_name] = reb.loc[low_ids, ret_name]
        low_rep["up_prob"] = reb.loc[low_ids, "up_prob"]
        low_rep["inv_ret"] = low_rep["weight"] * pd.to_numeric(low_rep[ret_name], errors="coerce").fillna(0.0)

        high_rep = pd.DataFrame(index=high_ids)
        high_rep["weight"] = pd.to_numeric(wH.reindex(high_ids), errors="coerce").fillna(0.0).abs() / wH_abs_sum
        high_rep[ret_name] = reb.loc[high_ids, ret_name]
        high_rep["up_prob"] = reb.loc[high_ids, "up_prob"]
        high_rep["inv_ret"] = high_rep["weight"] * pd.to_numeric(high_rep[ret_name], errors="coerce").fillna(0.0)

        row[0] = float(low_rep["inv_ret"].sum())
        row[cut - 1] = float(high_rep["inv_ret"].sum())
        to_df = pd.concat(
            [
                pd.DataFrame({"weight": wL_short, ret_name: reb.loc[low_ids, ret_name]}),
                pd.DataFrame({"weight": wH, ret_name: reb.loc[high_ids, ret_name]}),
            ],
            axis=0,
        )
        to_df["inv_ret"] = to_df["weight"] * to_df[ret_name]
        diag = pd.concat([low_rep[["up_prob", "inv_ret"]], high_rep[["up_prob", "inv_ret"]]], axis=0)
        return row, to_df, diag

class MWHPlusLowNegTCStrategy(TCOptimizerMixin, PortfolioStrategy):
    """High-plus-LowNeg strategy with drifted pre-trade state for each sleeve."""

    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        ctx = self.ctx
        reb = df.loc[date].copy()
        reb.index = pd.Index(reb.index).astype(str)
        row = np.zeros(cut, dtype=float)

        q10, q90 = self._require_tail_thresholds(
            reb,
            date=pd.Timestamp(date),
        )

        low_desired = reb.loc[reb["up_prob"] <= q10].copy()
        high_desired = reb.loc[reb["up_prob"] >= q90].copy()

        pre_high = self._tc_pretrade_state(
            ctx=ctx, prev_state=prev_state, state_key="hplus_high", date=date
        )
        pre_low = self._tc_pretrade_state(
            ctx=ctx, prev_state=prev_state, state_key="hplus_low_abs", date=date
        )
        high_df = self._tc_stateful_frame(
            reb_frame=self._tc_execution_support_frame(ctx=ctx, date=date, selection_frame=reb), desired_frame=high_desired, pretrade_state=pre_high
        )
        low_df = self._tc_stateful_frame(
            reb_frame=self._tc_execution_support_frame(ctx=ctx, date=date, selection_frame=reb), desired_frame=low_desired, pretrade_state=pre_low
        )

        prev_w_high = pre_high.aligned_weights(high_df.index)
        prev_w_low_abs = pre_low.aligned_weights(low_df.index).abs()
        common_target_sum = self._paired_common_long_only_sum(
            ctx=ctx,
            low_frame=low_df,
            high_frame=high_df,
            prev_w_low=prev_w_low_abs,
            prev_w_high=prev_w_high,
            use_abs_low_prev=True,
        )
        prev_state["paired_target_sum"] = common_target_sum

        high_w = pd.Series(0.0, index=high_df.index, dtype=float)
        low_w_abs = pd.Series(0.0, index=low_df.index, dtype=float)
        high_cost = 0.0
        low_cost = 0.0

        if len(high_df) > 0:
            _, SigmaH, colsH = self._eligible_cov_block(
                df, date, ret_name, high_df, leg_name="stateful high-decile"
            )
            desired_alpha_high = ctx._normalize_alpha(high_desired["up_prob"]) if len(high_desired) else pd.Series(dtype=float)
            alpha_high = self._tc_desired_alpha(frame=high_df, desired_alpha=desired_alpha_high)
            high_w, high_cost = self._solve_tc_block(
                ctx=ctx,
                frame=high_df,
                alpha=alpha_high,
                prev_w=prev_w_high,
                Sigma=SigmaH,
                cols=colsH,
                long_only=True,
                dollar_neutral=False,
                gross_target=1.0,
                desired_target_sum=common_target_sum,
            )

        if len(low_df) > 0:
            _, SigmaL, colsL = self._eligible_cov_block(
                df, date, ret_name, low_df, leg_name="stateful low-decile"
            )
            if len(low_desired):
                a0 = ctx._normalize_alpha(low_desired["up_prob"]).astype(float)
                rng = float(a0.max() - a0.min())
                z = (a0 - a0.min()) / (rng if rng > 1e-12 else 1.0)
                desired_alpha_low = (1.0 - z) + 1e-8
            else:
                desired_alpha_low = pd.Series(dtype=float)
            alpha_low = self._tc_desired_alpha(frame=low_df, desired_alpha=desired_alpha_low)
            low_w_abs, low_cost = self._solve_tc_block(
                ctx=ctx,
                frame=low_df,
                alpha=alpha_low,
                prev_w=prev_w_low_abs,
                Sigma=SigmaL,
                cols=colsL,
                long_only=True,
                dollar_neutral=False,
                gross_target=1.0,
                use_abs_prev=True,
                desired_target_sum=common_target_sum,
            )

        high_w = pd.to_numeric(high_w.reindex(high_df.index), errors="coerce").fillna(0.0).clip(lower=0.0)
        low_w_abs = pd.to_numeric(low_w_abs.reindex(low_df.index), errors="coerce").fillna(0.0).abs()

        high_result = self._tc_store_leg_state(
            prev_state=prev_state,
            state_key="hplus_high",
            pretrade_state=pre_high,
            posttrade_weights=high_w,
            frame=high_df,
            ret_name=ret_name,
            transaction_cost=float(high_cost),
        )
        low_result = self._tc_store_leg_state(
            prev_state=prev_state,
            state_key="hplus_low_abs",
            pretrade_state=pre_low,
            posttrade_weights=low_w_abs,
            frame=low_df,
            ret_name=ret_name,
            transaction_cost=float(low_cost),
        )
        prev_state["tc_leg_rebalance_results"] = {"low": low_result, "high": high_result}

        low_w_short = -low_w_abs
        high_to = pd.DataFrame(index=high_df.index)
        high_to["weight"] = high_w
        high_to[ret_name] = high_df[ret_name]
        high_to["inv_ret"] = high_to["weight"] * pd.to_numeric(high_to[ret_name], errors="coerce").fillna(0.0)
        if len(high_to) > 0:
            high_to.loc[high_to.index[0], "inv_ret"] -= float(high_cost)

        low_to = pd.DataFrame(index=low_df.index)
        low_to["weight"] = low_w_short
        low_to[ret_name] = low_df[ret_name]
        low_to["inv_ret"] = low_to["weight"] * pd.to_numeric(low_to[ret_name], errors="coerce").fillna(0.0)
        if len(low_to) > 0:
            low_to.loc[low_to.index[0], "inv_ret"] -= float(low_cost)

        low_rep = pd.DataFrame(index=low_df.index)
        low_rep["weight"] = low_w_abs
        low_rep[ret_name] = low_df[ret_name]
        low_rep["up_prob"] = low_df["up_prob"]
        low_rep["inv_ret"] = low_rep["weight"] * pd.to_numeric(low_rep[ret_name], errors="coerce").fillna(0.0)
        if len(low_rep) > 0:
            low_rep.loc[low_rep.index[0], "inv_ret"] += float(low_cost)

        high_rep = pd.DataFrame(index=high_df.index)
        high_rep["weight"] = high_w
        high_rep[ret_name] = high_df[ret_name]
        high_rep["up_prob"] = high_df["up_prob"]
        high_rep["inv_ret"] = high_rep["weight"] * pd.to_numeric(high_rep[ret_name], errors="coerce").fillna(0.0)
        if len(high_rep) > 0:
            high_rep.loc[high_rep.index[0], "inv_ret"] -= float(high_cost)

        self._append_paired_tc_debug_row(
            ctx=ctx,
            prev_state=prev_state,
            date=date,
            common_target_sum=common_target_sum,
            low_frame=low_df,
            high_frame=high_df,
            low_weights_abs=low_w_abs,
            high_weights=high_w,
            low_cost=low_cost,
            high_cost=high_cost,
        )

        row[0] = float(low_rep["inv_ret"].sum()) if len(low_rep) else 0.0
        row[cut - 1] = float(high_rep["inv_ret"].sum()) if len(high_rep) else 0.0

        to_df = pd.concat([low_to, high_to], axis=0)
        if to_df.index.has_duplicates:
            # Opposite-sleeve migration can create the same name in both stateful sleeves.
            # Net the signed holdings for reporting; a unified signed optimizer is a later refactor.
            agg = to_df.groupby(level=0, sort=False).agg({"weight": "sum", ret_name: "first"})
            agg["inv_ret"] = agg["weight"] * pd.to_numeric(agg[ret_name], errors="coerce").fillna(0.0)
            to_df = agg
        diag = pd.concat([low_rep[["up_prob", "inv_ret"]], high_rep[["up_prob", "inv_ret"]]], axis=0)
        return row, to_df, diag


class EWTCStrategy(TCOptimizerMixin, PortfolioStrategy):
    """Equal-weight deciles with canonical drifted pre-trade state."""

    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        ctx = self.ctx
        reb = df.loc[date].copy()
        reb.index = pd.Index(reb.index).astype(str)
        row = np.zeros(cut, dtype=float)
        kept: Dict[int, pd.DataFrame] = {}
        costs: Dict[int, float] = {}
        states: Dict[int, PreTradeState] = {}
        frames: Dict[int, pd.DataFrame] = {}
        desired_frames: Dict[int, pd.DataFrame] = {}

        up = reb["up_prob"].astype(float)
        for j in range(cut):
            lo, hi = ctx._decile_bounds(up, cut, j)
            mask = ctx._decile_mask(up, lo, hi, is_lowest=(j == 0))
            desired = reb[mask].copy()
            state = self._tc_pretrade_state(
                ctx=ctx, prev_state=prev_state, state_key=f"ew_decile_{j}", date=date
            )
            frame = self._tc_stateful_frame(
                reb_frame=self._tc_execution_support_frame(ctx=ctx, date=date, selection_frame=reb), desired_frame=desired, pretrade_state=state
            )
            desired_frames[j] = desired
            states[j] = state
            frames[j] = frame

        common_target_sum = self._paired_common_long_only_sum(
            ctx=ctx,
            low_frame=frames[0],
            high_frame=frames[cut - 1],
            prev_w_low=states[0].aligned_weights(frames[0].index),
            prev_w_high=states[cut - 1].aligned_weights(frames[cut - 1].index),
            use_abs_low_prev=False,
        )
        prev_state["paired_target_sum"] = common_target_sum

        for j in range(cut):
            desired = desired_frames[j]
            ddf = frames[j].copy()
            state = states[j]
            if len(ddf) == 0:
                ddf["weight"] = 0.0
                ddf["inv_ret"] = 0.0
                kept[j] = ddf
                costs[j] = 0.0
                continue

            target_w = pd.Series(0.0, index=ddf.index, dtype=float)
            if len(desired):
                target_w.loc[desired.index] = 1.0 / float(len(desired))
            prev_w = state.aligned_weights(ddf.index).clip(lower=0.0)
            desired_target_sum = common_target_sum if j in {0, cut - 1} else None
            w, tc_cost = self._solve_tc_target_tracking(
                ctx=ctx,
                frame=ddf,
                target_w=target_w,
                prev_w=prev_w,
                desired_target_sum=desired_target_sum,
            )
            w = pd.to_numeric(w.reindex(ddf.index), errors="coerce").fillna(0.0).clip(lower=0.0)
            self._tc_store_leg_state(
                prev_state=prev_state,
                state_key=f"ew_decile_{j}",
                pretrade_state=state,
                posttrade_weights=w,
                frame=ddf,
                ret_name=ret_name,
                transaction_cost=float(tc_cost),
            )

            ddf["weight"] = w
            ddf["inv_ret"] = ddf["weight"] * pd.to_numeric(ddf[ret_name], errors="coerce").fillna(0.0)
            if len(ddf) > 0:
                ddf.loc[ddf.index[0], "inv_ret"] -= float(tc_cost)
            kept[j] = ddf
            costs[j] = float(tc_cost)
            row[j] = float(ddf["inv_ret"].sum())

        low_long = kept[0].copy()
        high_long = kept[cut - 1].copy()
        low_cost = float(costs.get(0, 0.0))
        high_cost = float(costs.get(cut - 1, 0.0))

        low_rep = low_long.copy()
        low_rep["weight"] = pd.to_numeric(low_rep["weight"], errors="coerce").fillna(0.0).abs()
        low_rep["inv_ret"] = low_rep["weight"] * pd.to_numeric(low_rep[ret_name], errors="coerce").fillna(0.0)
        if len(low_rep) > 0:
            low_rep.loc[low_rep.index[0], "inv_ret"] += low_cost

        high_rep = high_long.copy()
        high_rep["weight"] = pd.to_numeric(high_rep["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        high_rep["inv_ret"] = high_rep["weight"] * pd.to_numeric(high_rep[ret_name], errors="coerce").fillna(0.0)
        if len(high_rep) > 0:
            high_rep.loc[high_rep.index[0], "inv_ret"] -= high_cost

        self._append_paired_tc_debug_row(
            ctx=ctx, prev_state=prev_state, date=date, common_target_sum=common_target_sum,
            low_frame=low_long, high_frame=high_long, low_weights_abs=low_long["weight"],
            high_weights=high_long["weight"], low_cost=low_cost, high_cost=high_cost,
        )

        row[0] = float(low_rep["inv_ret"].sum()) if len(low_rep) else 0.0
        row[cut - 1] = float(high_rep["inv_ret"].sum()) if len(high_rep) else 0.0

        low_to = pd.DataFrame(index=low_long.index)
        low_to["weight"] = -pd.to_numeric(low_long["weight"], errors="coerce").fillna(0.0).abs()
        low_to[ret_name] = low_long[ret_name]
        low_to["inv_ret"] = low_to["weight"] * pd.to_numeric(low_to[ret_name], errors="coerce").fillna(0.0)
        if len(low_to) > 0:
            low_to.loc[low_to.index[0], "inv_ret"] -= low_cost

        high_to = pd.DataFrame(index=high_long.index)
        high_to["weight"] = pd.to_numeric(high_long["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        high_to[ret_name] = high_long[ret_name]
        high_to["inv_ret"] = high_to["weight"] * pd.to_numeric(high_to[ret_name], errors="coerce").fillna(0.0)
        if len(high_to) > 0:
            high_to.loc[high_to.index[0], "inv_ret"] -= high_cost

        to_df = pd.concat([low_to, high_to], axis=0)
        if to_df.index.has_duplicates:
            agg = to_df.groupby(level=0, sort=False).agg({"weight": "sum", ret_name: "first"})
            agg["inv_ret"] = agg["weight"] * pd.to_numeric(agg[ret_name], errors="coerce").fillna(0.0)
            to_df = agg
        diag = pd.concat([low_rep[["up_prob", "inv_ret"]], high_rep[["up_prob", "inv_ret"]]], axis=0)
        return row, to_df, diag


class VWTCStrategy(TCOptimizerMixin, PortfolioStrategy):
    """Value-weight deciles with canonical drifted pre-trade state."""

    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        ctx = self.ctx
        reb = df.loc[date].copy()
        reb.index = pd.Index(reb.index).astype(str)
        row = np.zeros(cut, dtype=float)
        kept: Dict[int, pd.DataFrame] = {}
        costs: Dict[int, float] = {}
        states: Dict[int, PreTradeState] = {}
        frames: Dict[int, pd.DataFrame] = {}
        desired_frames: Dict[int, pd.DataFrame] = {}

        up = reb["up_prob"].astype(float)
        for j in range(cut):
            lo, hi = ctx._decile_bounds(up, cut, j)
            mask = ctx._decile_mask(up, lo, hi, is_lowest=(j == 0))
            desired = reb[mask].copy()
            state = self._tc_pretrade_state(
                ctx=ctx, prev_state=prev_state, state_key=f"vw_decile_{j}", date=date
            )
            frame = self._tc_stateful_frame(
                reb_frame=self._tc_execution_support_frame(ctx=ctx, date=date, selection_frame=reb), desired_frame=desired, pretrade_state=state
            )
            desired_frames[j] = desired
            states[j] = state
            frames[j] = frame

        common_target_sum = self._paired_common_long_only_sum(
            ctx=ctx,
            low_frame=frames[0],
            high_frame=frames[cut - 1],
            prev_w_low=states[0].aligned_weights(frames[0].index),
            prev_w_high=states[cut - 1].aligned_weights(frames[cut - 1].index),
            use_abs_low_prev=False,
        )
        prev_state["paired_target_sum"] = common_target_sum

        for j in range(cut):
            desired = desired_frames[j]
            ddf = frames[j].copy()
            state = states[j]
            if len(ddf) == 0:
                ddf["weight"] = 0.0
                ddf["inv_ret"] = 0.0
                kept[j] = ddf
                costs[j] = 0.0
                continue

            target_w = pd.Series(0.0, index=ddf.index, dtype=float)
            if len(desired):
                desired_target = self._strict_value_weights(
                    desired,
                    date=pd.Timestamp(date),
                    label=f"VWTC decile={j}",
                )
                target_w.loc[desired.index] = desired_target
            prev_w = state.aligned_weights(ddf.index).clip(lower=0.0)
            desired_target_sum = common_target_sum if j in {0, cut - 1} else None
            w, tc_cost = self._solve_tc_target_tracking(
                ctx=ctx,
                frame=ddf,
                target_w=target_w,
                prev_w=prev_w,
                desired_target_sum=desired_target_sum,
            )
            w = pd.to_numeric(w.reindex(ddf.index), errors="coerce").fillna(0.0).clip(lower=0.0)
            self._tc_store_leg_state(
                prev_state=prev_state,
                state_key=f"vw_decile_{j}",
                pretrade_state=state,
                posttrade_weights=w,
                frame=ddf,
                ret_name=ret_name,
                transaction_cost=float(tc_cost),
            )

            ddf["weight"] = w
            ddf["inv_ret"] = ddf["weight"] * pd.to_numeric(ddf[ret_name], errors="coerce").fillna(0.0)
            if len(ddf) > 0:
                ddf.loc[ddf.index[0], "inv_ret"] -= float(tc_cost)
            kept[j] = ddf
            costs[j] = float(tc_cost)
            row[j] = float(ddf["inv_ret"].sum())

        low_long = kept[0].copy()
        high_long = kept[cut - 1].copy()
        low_cost = float(costs.get(0, 0.0))
        high_cost = float(costs.get(cut - 1, 0.0))

        low_rep = low_long.copy()
        low_rep["weight"] = pd.to_numeric(low_rep["weight"], errors="coerce").fillna(0.0).abs()
        low_rep["inv_ret"] = low_rep["weight"] * pd.to_numeric(low_rep[ret_name], errors="coerce").fillna(0.0)
        if len(low_rep) > 0:
            low_rep.loc[low_rep.index[0], "inv_ret"] += low_cost

        high_rep = high_long.copy()
        high_rep["weight"] = pd.to_numeric(high_rep["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        high_rep["inv_ret"] = high_rep["weight"] * pd.to_numeric(high_rep[ret_name], errors="coerce").fillna(0.0)
        if len(high_rep) > 0:
            high_rep.loc[high_rep.index[0], "inv_ret"] -= high_cost

        self._append_paired_tc_debug_row(
            ctx=ctx, prev_state=prev_state, date=date, common_target_sum=common_target_sum,
            low_frame=low_long, high_frame=high_long, low_weights_abs=low_long["weight"],
            high_weights=high_long["weight"], low_cost=low_cost, high_cost=high_cost,
        )

        row[0] = float(low_rep["inv_ret"].sum()) if len(low_rep) else 0.0
        row[cut - 1] = float(high_rep["inv_ret"].sum()) if len(high_rep) else 0.0

        low_to = pd.DataFrame(index=low_long.index)
        low_to["weight"] = -pd.to_numeric(low_long["weight"], errors="coerce").fillna(0.0).abs()
        low_to[ret_name] = low_long[ret_name]
        low_to["inv_ret"] = low_to["weight"] * pd.to_numeric(low_to[ret_name], errors="coerce").fillna(0.0)
        if len(low_to) > 0:
            low_to.loc[low_to.index[0], "inv_ret"] -= low_cost

        high_to = pd.DataFrame(index=high_long.index)
        high_to["weight"] = pd.to_numeric(high_long["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        high_to[ret_name] = high_long[ret_name]
        high_to["inv_ret"] = high_to["weight"] * pd.to_numeric(high_to[ret_name], errors="coerce").fillna(0.0)
        if len(high_to) > 0:
            high_to.loc[high_to.index[0], "inv_ret"] -= high_cost

        to_df = pd.concat([low_to, high_to], axis=0)
        if to_df.index.has_duplicates:
            agg = to_df.groupby(level=0, sort=False).agg({"weight": "sum", ret_name: "first"})
            agg["inv_ret"] = agg["weight"] * pd.to_numeric(agg[ret_name], errors="coerce").fillna(0.0)
            to_df = agg
        diag = pd.concat([low_rep[["up_prob", "inv_ret"]], high_rep[["up_prob", "inv_ret"]]], axis=0)
        return row, to_df, diag


class MWFullTCStrategy(TCOptimizerMixin, PortfolioStrategy):
    """Per-decile Markowitz with canonical drifted pre-trade state."""

    def compute_for_date(self, df, date, cut, ret_name, prev_state):
        ctx = self.ctx
        reb = df.loc[date].copy()
        reb.index = pd.Index(reb.index).astype(str)
        row = np.zeros(cut, dtype=float)
        kept: Dict[int, pd.DataFrame] = {}
        costs: Dict[int, float] = {}
        states: Dict[int, PreTradeState] = {}
        frames: Dict[int, pd.DataFrame] = {}
        desired_frames: Dict[int, pd.DataFrame] = {}

        up = reb["up_prob"].astype(float)
        for j in range(cut):
            lo, hi = ctx._decile_bounds(up, cut, j)
            mask = ctx._decile_mask(up, lo, hi, is_lowest=(j == 0))
            desired = reb[mask].copy()
            state = self._tc_pretrade_state(
                ctx=ctx, prev_state=prev_state, state_key=f"mw_full_decile_{j}", date=date
            )
            frame = self._tc_stateful_frame(
                reb_frame=self._tc_execution_support_frame(ctx=ctx, date=date, selection_frame=reb), desired_frame=desired, pretrade_state=state
            )
            desired_frames[j] = desired
            states[j] = state
            frames[j] = frame

        common_target_sum = None
        if ctx.mw_long_only:
            common_target_sum = self._paired_common_long_only_sum(
                ctx=ctx,
                low_frame=frames[0],
                high_frame=frames[cut - 1],
                prev_w_low=states[0].aligned_weights(frames[0].index),
                prev_w_high=states[cut - 1].aligned_weights(frames[cut - 1].index),
                use_abs_low_prev=False,
            )
        prev_state["paired_target_sum"] = common_target_sum

        for j in range(cut):
            desired = desired_frames[j]
            ddf = frames[j].copy()
            state = states[j]
            if len(ddf) == 0:
                ddf["weight"] = 0.0
                ddf["inv_ret"] = 0.0
                kept[j] = ddf
                costs[j] = 0.0
                continue

            _, Sigma, cols = self._eligible_cov_block(
                df, date, ret_name, ddf, leg_name=f"stateful decile={j}"
            )
            desired_alpha = ctx._normalize_alpha(desired["up_prob"]) if len(desired) else pd.Series(dtype=float)
            alpha = self._tc_desired_alpha(frame=ddf, desired_alpha=desired_alpha)
            prev_w = state.aligned_weights(ddf.index)
            if ctx.mw_long_only:
                prev_w = prev_w.clip(lower=0.0)
            dollar_neutral = bool(ctx.tc_dollar_neutral) if (ctx.tc_dollar_neutral is not None) else (not ctx.mw_long_only)
            desired_target_sum = common_target_sum if (ctx.mw_long_only and j in {0, cut - 1}) else None
            w, tc_cost = self._solve_tc_block(
                ctx=ctx,
                frame=ddf,
                alpha=alpha,
                prev_w=prev_w,
                Sigma=Sigma,
                cols=cols,
                long_only=ctx.mw_long_only,
                dollar_neutral=dollar_neutral,
                gross_target=ctx.mw_spread_gross,
                desired_target_sum=desired_target_sum,
            )
            w = pd.to_numeric(w.reindex(ddf.index), errors="coerce").fillna(0.0)
            if ctx.mw_long_only:
                w = w.clip(lower=0.0)
            self._tc_store_leg_state(
                prev_state=prev_state,
                state_key=f"mw_full_decile_{j}",
                pretrade_state=state,
                posttrade_weights=w,
                frame=ddf,
                ret_name=ret_name,
                transaction_cost=float(tc_cost),
            )

            ddf["weight"] = w
            ddf["inv_ret"] = ddf["weight"] * pd.to_numeric(ddf[ret_name], errors="coerce").fillna(0.0)
            if len(ddf) > 0:
                ddf.loc[ddf.index[0], "inv_ret"] -= float(tc_cost)
            kept[j] = ddf
            costs[j] = float(tc_cost)
            row[j] = float(ddf["inv_ret"].sum())

        low_long = kept[0].copy()
        high_long = kept[cut - 1].copy()
        low_cost = float(costs.get(0, 0.0))
        high_cost = float(costs.get(cut - 1, 0.0))

        low_rep = low_long.copy()
        low_rep["weight"] = pd.to_numeric(low_rep["weight"], errors="coerce").fillna(0.0).abs()
        low_rep["inv_ret"] = low_rep["weight"] * pd.to_numeric(low_rep[ret_name], errors="coerce").fillna(0.0)
        if len(low_rep) > 0:
            low_rep.loc[low_rep.index[0], "inv_ret"] += low_cost

        high_rep = high_long.copy()
        high_rep["weight"] = pd.to_numeric(high_rep["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        high_rep["inv_ret"] = high_rep["weight"] * pd.to_numeric(high_rep[ret_name], errors="coerce").fillna(0.0)
        if len(high_rep) > 0:
            high_rep.loc[high_rep.index[0], "inv_ret"] -= high_cost

        self._append_paired_tc_debug_row(
            ctx=ctx, prev_state=prev_state, date=date, common_target_sum=common_target_sum,
            low_frame=low_long, high_frame=high_long, low_weights_abs=low_long["weight"],
            high_weights=high_long["weight"], low_cost=low_cost, high_cost=high_cost,
        )

        row[0] = float(low_rep["inv_ret"].sum()) if len(low_rep) else 0.0
        row[cut - 1] = float(high_rep["inv_ret"].sum()) if len(high_rep) else 0.0

        low_to = pd.DataFrame(index=low_long.index)
        low_to["weight"] = -pd.to_numeric(low_long["weight"], errors="coerce").fillna(0.0).abs()
        low_to[ret_name] = low_long[ret_name]
        low_to["inv_ret"] = low_to["weight"] * pd.to_numeric(low_to[ret_name], errors="coerce").fillna(0.0)
        if len(low_to) > 0:
            low_to.loc[low_to.index[0], "inv_ret"] -= low_cost

        high_to = pd.DataFrame(index=high_long.index)
        high_to["weight"] = pd.to_numeric(high_long["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        high_to[ret_name] = high_long[ret_name]
        high_to["inv_ret"] = high_to["weight"] * pd.to_numeric(high_to[ret_name], errors="coerce").fillna(0.0)
        if len(high_to) > 0:
            high_to.loc[high_to.index[0], "inv_ret"] -= high_cost

        to_df = pd.concat([low_to, high_to], axis=0)
        if to_df.index.has_duplicates:
            agg = to_df.groupby(level=0, sort=False).agg({"weight": "sum", ret_name: "first"})
            agg["inv_ret"] = agg["weight"] * pd.to_numeric(agg[ret_name], errors="coerce").fillna(0.0)
            to_df = agg
        diag = pd.concat([low_rep[["up_prob", "inv_ret"]], high_rep[["up_prob", "inv_ret"]]], axis=0)
        return row, to_df, diag

