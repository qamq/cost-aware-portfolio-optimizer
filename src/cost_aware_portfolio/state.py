"""Canonical portfolio-state transitions for weekly rebalancing.

This module defines the portfolio state that exists immediately before and after
one rebalance.  Its purpose is to make one trade vector authoritative across
portfolio construction, liquidity constraints, turnover, and transaction-cost
accounting.

The central sequence is:

    previous post-trade weights
        -> drift through the realized holding-period returns
        -> current pre-trade weights
        -> optimize / choose current post-trade weights
        -> delta_w = post-trade - pre-trade

Two design choices are intentional:

1. Previous holdings and current desired holdings are aligned on their UNION.
   A stock does not disappear merely because it leaves the current signal sleeve;
   if it was previously held, an explicit trade toward zero is required.

2. Drift is based on portfolio wealth accounting rather than re-scaling the old
   portfolio to the new target gross exposure.  Re-levering back to a desired
   gross exposure is therefore an actual trade and is visible in ``delta_w``.

The module is independent of the optimizer and transaction-cost model.  Those
layers should consume these state objects rather than reconstructing trades on
an ad hoc basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


_NUMERIC_TOL = 1e-12


def _canonical_index(values: Iterable[object]) -> pd.Index:
    """Return a deterministic string index."""
    return pd.Index(values).astype(str)


def _as_numeric_series(
    values: Optional[pd.Series],
    *,
    index: Optional[pd.Index] = None,
    fill_value: float = 0.0,
    name: Optional[str] = None,
) -> pd.Series:
    """Coerce a Series to finite float values on a canonical string index."""
    if values is None:
        out = pd.Series(dtype=float)
    else:
        out = pd.Series(values).copy()
        out.index = _canonical_index(out.index)
        out = pd.to_numeric(out, errors="coerce").astype(float)
        out = out.replace([np.inf, -np.inf], np.nan)
        if out.index.has_duplicates:
            out = out.groupby(level=0, sort=False).last()

    if index is not None:
        canonical = _canonical_index(index)
        out = out.reindex(canonical)

    out = out.fillna(float(fill_value)).astype(float)
    if name is not None:
        out.name = name
    return out


def union_holdings_index(*series: Optional[pd.Series]) -> pd.Index:
    """Return the union of names appearing in any supplied portfolio Series.

    The result is sorted for deterministic downstream behavior.  This is the
    universe that should be used whenever previous holdings are compared with a
    new desired or post-trade portfolio.
    """
    names = []
    for item in series:
        if item is None:
            continue
        names.extend(pd.Index(item.index).astype(str).tolist())
    if not names:
        return pd.Index([], dtype=object)
    return pd.Index(sorted(set(names)), dtype=object)


def build_rebalance_universe(
    pretrade_weights: pd.Series,
    desired_index: Iterable[object],
) -> pd.Index:
    """Return current desired names plus every nonzero legacy holding.

    This helper is intended for the optimizer-facing universe *before* new
    post-trade weights exist.  Previously held names remain present even when
    they leave the current signal sleeve, allowing the optimizer to liquidate
    them explicitly subject to trade-cap constraints.
    """
    pre = _as_numeric_series(pretrade_weights, name="w_pre")
    legacy = pre.index[pre.abs() > _NUMERIC_TOL].astype(str).tolist()
    desired = pd.Index(desired_index).astype(str).tolist()
    return pd.Index(sorted(set(legacy + desired)), dtype=object)


@dataclass(frozen=True)
class DriftResult:
    """Portfolio state after market moves but before the next rebalance.

    Attributes
    ----------
    weights:
        Risky-asset weights immediately before trading at the next rebalance.
    cash_weight:
        Cash / financing weight immediately before trading.  The accounting
        identity ``cash_weight + weights.sum() == 1`` is preserved, up to
        floating-point tolerance.
    portfolio_return:
        Holding-period return on NAV, including the optional cash return.
    growth_factor:
        ``1 + portfolio_return``.  For a dynamic-NAV simulation, the caller can
        update NAV as ``new_aum = old_aum * growth_factor``.
    """

    weights: pd.Series
    cash_weight: float
    portfolio_return: float
    growth_factor: float


@dataclass(frozen=True)
class PreTradeState:
    """Authoritative state immediately before a rebalance."""

    date: pd.Timestamp
    aum: float
    weights: pd.Series
    cash_weight: float
    holding_period_return: float = 0.0
    growth_factor: float = 1.0
    previous_date: Optional[pd.Timestamp] = None

    def aligned_weights(self, index: pd.Index) -> pd.Series:
        """Return pre-trade risky weights aligned to a requested universe."""
        return _as_numeric_series(
            self.weights,
            index=index,
            fill_value=0.0,
            name="w_pre",
        )

    @property
    def gross_exposure(self) -> float:
        return float(self.weights.abs().sum())

    @property
    def net_exposure(self) -> float:
        return float(self.weights.sum())


@dataclass(frozen=True)
class RebalanceResult:
    """Authoritative post-rebalance state and executed trade vector.

    ``delta_w`` and ``dollar_trades`` are computed once from the pre- and
    post-trade portfolios.  Downstream turnover and cost diagnostics should use
    these stored values rather than reconstructing trades independently.
    """

    date: pd.Timestamp
    aum: float
    pretrade_weights: pd.Series
    posttrade_weights: pd.Series
    delta_w: pd.Series
    dollar_trades: pd.Series
    pretrade_cash_weight: float
    posttrade_cash_weight: float
    turnover_oneway: float

    @property
    def gross_pretrade(self) -> float:
        return float(self.pretrade_weights.abs().sum())

    @property
    def gross_posttrade(self) -> float:
        return float(self.posttrade_weights.abs().sum())

    @property
    def net_pretrade(self) -> float:
        return float(self.pretrade_weights.sum())

    @property
    def net_posttrade(self) -> float:
        return float(self.posttrade_weights.sum())

    @property
    def total_absolute_dollar_trades(self) -> float:
        return float(self.dollar_trades.abs().sum())

    @property
    def one_way_dollar_turnover(self) -> float:
        """One-way traded notional in dollars under the project's 1/2 convention."""
        return float(self.aum * self.turnover_oneway)


def drift_posttrade_portfolio(
    posttrade_weights: pd.Series,
    realized_returns: pd.Series,
    *,
    cash_weight: Optional[float] = None,
    cash_return: float = 0.0,
) -> DriftResult:
    """Drift post-trade weights through one realized holding period.

    Parameters
    ----------
    posttrade_weights:
        Risky-asset weights immediately after the previous rebalance.  Signed
        weights are supported, so the function can represent long-short books.
    realized_returns:
        Asset returns earned between the previous and current rebalance dates.
        Names not present in ``posttrade_weights`` are irrelevant.  A missing
        return for a held asset is treated as zero only after alignment; callers
        should normally supply the exact realized return series from the
        portfolio panel.
    cash_weight:
        Previous post-trade cash / financing weight.  If omitted, it is inferred
        from the accounting identity ``1 - sum(posttrade_weights)``.  For a
        dollar-neutral long-short portfolio whose risky weights sum to zero,
        this therefore defaults to 1.0.
    cash_return:
        Holding-period return on cash / financing.  The default is zero, matching
        the current project convention.

    Returns
    -------
    DriftResult
        The actual risky weights that exist immediately before the next trade,
        together with the NAV growth factor over the holding period.

    Notes
    -----
    Drift is based on wealth accounting:

        risky_value_i' = w_i * (1 + r_i)
        cash_value'    = c   * (1 + r_cash)
        growth         = cash_value' + sum_i risky_value_i'
        w_i_pre        = risky_value_i' / growth

    No target-gross normalization is applied here.  If the portfolio needs to be
    re-levered to a target gross exposure, that change belongs in the next
    rebalance and therefore appears in ``delta_w``.
    """
    w_post = _as_numeric_series(posttrade_weights, name="w_post_previous")
    rets = _as_numeric_series(
        realized_returns,
        index=w_post.index,
        fill_value=0.0,
        name="realized_return",
    )

    if cash_weight is None:
        cash_weight_value = 1.0 - float(w_post.sum())
    else:
        cash_weight_value = float(cash_weight)

    if not np.isfinite(cash_weight_value):
        raise ValueError("cash_weight must be finite.")
    if not np.isfinite(float(cash_return)):
        raise ValueError("cash_return must be finite.")

    risky_values = w_post * (1.0 + rets)
    cash_value = cash_weight_value * (1.0 + float(cash_return))
    growth_factor = float(cash_value + risky_values.sum())

    if not np.isfinite(growth_factor) or growth_factor <= _NUMERIC_TOL:
        raise ValueError(
            "Portfolio NAV growth factor is non-positive or non-finite; "
            "cannot form pre-trade weights."
        )

    w_pre = (risky_values / growth_factor).astype(float)
    w_pre.name = "w_pre"
    cash_pre = float(cash_value / growth_factor)
    portfolio_return = float(growth_factor - 1.0)

    accounting_total = float(w_pre.sum()) + cash_pre
    if not np.isclose(accounting_total, 1.0, atol=1e-10, rtol=1e-10):
        raise RuntimeError(
            "Drifted portfolio failed the NAV accounting identity: "
            "cash + risky weights != 1."
        )

    return DriftResult(
        weights=w_pre,
        cash_weight=cash_pre,
        portfolio_return=portfolio_return,
        growth_factor=growth_factor,
    )


def build_pretrade_state(
    *,
    date: pd.Timestamp,
    aum: float,
    previous_posttrade_weights: Optional[pd.Series] = None,
    realized_returns: Optional[pd.Series] = None,
    previous_cash_weight: Optional[float] = None,
    cash_return: float = 0.0,
    previous_date: Optional[pd.Timestamp] = None,
) -> PreTradeState:
    """Construct the authoritative pre-trade state for one rebalance date.

    For the first rebalance, omit ``previous_posttrade_weights`` and an all-cash
    state is returned.  Otherwise the previous holdings are drifted through the
    supplied realized returns.

    ``aum`` is the capital base to use for the CURRENT rebalance.  A fixed-AUM
    capacity test can pass the same value every week.  A dynamic-NAV simulation
    can instead pass the prior AUM multiplied by the preceding ``growth_factor``.
    """
    date_value = pd.Timestamp(date)
    aum_value = float(aum)
    if not np.isfinite(aum_value) or aum_value <= 0.0:
        raise ValueError("aum must be a positive finite number.")

    if previous_posttrade_weights is None or len(previous_posttrade_weights) == 0:
        return PreTradeState(
            date=date_value,
            aum=aum_value,
            weights=pd.Series(dtype=float, name="w_pre"),
            cash_weight=1.0,
            holding_period_return=0.0,
            growth_factor=1.0,
            previous_date=None if previous_date is None else pd.Timestamp(previous_date),
        )

    if realized_returns is None:
        raise ValueError(
            "realized_returns are required when previous_posttrade_weights are supplied."
        )

    drift = drift_posttrade_portfolio(
        previous_posttrade_weights,
        realized_returns,
        cash_weight=previous_cash_weight,
        cash_return=cash_return,
    )

    return PreTradeState(
        date=date_value,
        aum=aum_value,
        weights=drift.weights,
        cash_weight=drift.cash_weight,
        holding_period_return=drift.portfolio_return,
        growth_factor=drift.growth_factor,
        previous_date=None if previous_date is None else pd.Timestamp(previous_date),
    )


def align_pretrade_and_posttrade(
    pretrade_weights: pd.Series,
    posttrade_weights: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Align pre- and post-trade portfolios on their union of holdings."""
    idx = union_holdings_index(pretrade_weights, posttrade_weights)
    pre = _as_numeric_series(
        pretrade_weights,
        index=idx,
        fill_value=0.0,
        name="w_pre",
    )
    post = _as_numeric_series(
        posttrade_weights,
        index=idx,
        fill_value=0.0,
        name="w_post",
    )
    return pre, post


def compute_trade_vector(
    pretrade_weights: pd.Series,
    posttrade_weights: pd.Series,
) -> pd.Series:
    """Return the single authoritative signed trade vector ``post - pre``."""
    pre, post = align_pretrade_and_posttrade(pretrade_weights, posttrade_weights)
    delta = (post - pre).astype(float)
    delta.name = "delta_w"
    return delta


def build_rebalance_result(
    *,
    pretrade_state: PreTradeState,
    posttrade_weights: pd.Series,
    posttrade_cash_weight: Optional[float] = None,
) -> RebalanceResult:
    """Create the canonical result for one completed rebalance.

    The portfolio is aligned on the union of old and new holdings.  Consequently,
    a legacy holding omitted from ``posttrade_weights`` receives a post-trade
    target of zero and creates an explicit exit trade.
    """
    pre, post = align_pretrade_and_posttrade(
        pretrade_state.weights,
        posttrade_weights,
    )
    delta = (post - pre).astype(float)
    delta.name = "delta_w"

    if posttrade_cash_weight is None:
        post_cash = 1.0 - float(post.sum())
    else:
        post_cash = float(posttrade_cash_weight)

    if not np.isfinite(post_cash):
        raise ValueError("posttrade_cash_weight must be finite.")

    post_accounting_total = float(post.sum()) + post_cash
    if not np.isclose(post_accounting_total, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(
            "Post-trade portfolio violates the NAV accounting identity: "
            "cash + risky weights must equal 1."
        )

    aum = float(pretrade_state.aum)
    dollar_trades = (aum * delta).astype(float)
    dollar_trades.name = "dollar_trade"

    turnover_oneway = 0.5 * float(delta.abs().sum())

    return RebalanceResult(
        date=pd.Timestamp(pretrade_state.date),
        aum=aum,
        pretrade_weights=pre,
        posttrade_weights=post,
        delta_w=delta,
        dollar_trades=dollar_trades,
        pretrade_cash_weight=float(pretrade_state.cash_weight),
        posttrade_cash_weight=post_cash,
        turnover_oneway=turnover_oneway,
    )




def combine_signed_rebalance_results(
    *,
    low_abs_result: RebalanceResult,
    high_result: RebalanceResult,
) -> RebalanceResult:
    """Combine positive low/high sleeve results into one signed H-L transition.

    The low sleeve is interpreted as a short position and therefore receives a
    negative sign; the high sleeve remains positive.  The returned
    ``RebalanceResult`` is the authoritative signed portfolio transition used
    for turnover and transaction-cost diagnostics.

    If a stock migrates directly from one sleeve to the other, the two sleeve
    states are netted before ``delta_w`` is computed.  For example, moving from
    +2% long to -2% short creates one -4% signed trade, rather than two
    independently reconstructed trades.
    """
    if pd.Timestamp(low_abs_result.date) != pd.Timestamp(high_result.date):
        raise ValueError("Low and high sleeve results must have the same rebalance date.")
    if not np.isclose(float(low_abs_result.aum), float(high_result.aum)):
        raise ValueError("Low and high sleeve results must use the same AUM.")

    idx = union_holdings_index(
        low_abs_result.pretrade_weights,
        low_abs_result.posttrade_weights,
        high_result.pretrade_weights,
        high_result.posttrade_weights,
    )
    low_pre = _as_numeric_series(low_abs_result.pretrade_weights, index=idx, fill_value=0.0)
    low_post = _as_numeric_series(low_abs_result.posttrade_weights, index=idx, fill_value=0.0)
    high_pre = _as_numeric_series(high_result.pretrade_weights, index=idx, fill_value=0.0)
    high_post = _as_numeric_series(high_result.posttrade_weights, index=idx, fill_value=0.0)

    signed_pre = (high_pre - low_pre).astype(float)
    signed_post = (high_post - low_post).astype(float)
    signed_pre.name = "w_pre"
    signed_post.name = "w_post"

    pre_state = PreTradeState(
        date=pd.Timestamp(high_result.date),
        aum=float(high_result.aum),
        weights=signed_pre,
        cash_weight=1.0 - float(signed_pre.sum()),
    )
    return build_rebalance_result(
        pretrade_state=pre_state,
        posttrade_weights=signed_post,
        posttrade_cash_weight=1.0 - float(signed_post.sum()),
    )


__all__ = [
    "DriftResult",
    "PreTradeState",
    "RebalanceResult",
    "align_pretrade_and_posttrade",
    "build_pretrade_state",
    "build_rebalance_universe",
    "build_rebalance_result",
    "combine_signed_rebalance_results",
    "compute_trade_vector",
    "drift_posttrade_portfolio",
    "union_holdings_index",
]
