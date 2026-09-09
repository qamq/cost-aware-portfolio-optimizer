"""Canonical transaction-cost inputs for portfolio execution.

The portfolio optimizer consumes three execution inputs with explicit units:

``tc_adv_daily``
    Forecast average DAILY dollar volume over the forward execution window.
    The frozen CatBoost artifact supplies this as ``pred_adv``.

``tc_sigma_daily``
    Forecast DAILY return volatility over the same forward window.  The frozen
    CatBoost artifact supplies this as ``pred_vol``.

``tc_spread_oneway``
    One-way spread cost as a fraction of traded notional.  The weekly portfolio
    panel supplies either ``spread_oneway`` directly or ``bid_ask_bps`` which is
    converted from basis points to a fraction.

The purpose of this module is to make the execution units explicit and prevent
silent substitution of incompatible historical proxies.  In particular, the
primary cost-aware path must not combine daily ADV with a weekly volatility
proxy such as ``sqrt(mw_var)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


TC_ADV_DAILY_COL = "tc_adv_daily"
TC_SIGMA_DAILY_COL = "tc_sigma_daily"
TC_SPREAD_ONEWAY_COL = "tc_spread_oneway"
TC_TRAIN_END_YEAR_COL = "tc_train_end_year"


@dataclass(frozen=True)
class TransactionCostInputs:
    """Optimizer-ready transaction-cost inputs on one aligned stock universe."""

    adv_daily: pd.Series
    sigma_daily: pd.Series
    spread_oneway: pd.Series


def _coerce_index(df: pd.DataFrame, *, frame_name: str) -> pd.DataFrame:
    """Return a unique ``(Date, StockID)`` MultiIndex with canonical key types."""
    out = df.copy()

    if isinstance(out.index, pd.MultiIndex):
        if list(out.index.names) != ["Date", "StockID"]:
            raise KeyError(
                "%s must use MultiIndex names ['Date', 'StockID']; got %s"
                % (frame_name, list(out.index.names))
            )
        out = out.reset_index()
    else:
        missing = {"Date", "StockID"} - set(out.columns)
        if missing:
            raise KeyError("%s is missing required key columns: %s" % (frame_name, sorted(missing)))

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["StockID"] = out["StockID"].astype(str)
    if out["Date"].isna().any():
        raise ValueError("%s contains invalid Date values." % frame_name)

    duplicate_mask = out.duplicated(["Date", "StockID"], keep=False)
    if duplicate_mask.any():
        sample = out.loc[duplicate_mask, ["Date", "StockID"]].head(5).to_dict("records")
        raise ValueError("%s contains duplicate (Date, StockID) keys; examples=%s" % (frame_name, sample))

    return out.set_index(["Date", "StockID"]).sort_index()


def _validate_finite_nonnegative(
    values: pd.Series,
    *,
    name: str,
    strictly_positive: bool,
    allow_missing: bool,
) -> pd.Series:
    """Coerce a numeric execution series and enforce its economic domain."""
    out = pd.to_numeric(values, errors="coerce").astype(float)
    finite = np.isfinite(out.to_numpy(dtype=float))
    missing = out.isna().to_numpy() | ~finite

    if not allow_missing and bool(np.any(missing)):
        raise RuntimeError("%s contains missing or non-finite values." % name)

    valid = ~missing
    arr = out.to_numpy(dtype=float)
    if strictly_positive:
        bad_domain = valid & (arr <= 0.0)
        if bool(np.any(bad_domain)):
            raise RuntimeError("%s must be strictly positive wherever present." % name)
    else:
        bad_domain = valid & (arr < 0.0)
        if bool(np.any(bad_domain)):
            raise RuntimeError("%s must be non-negative wherever present." % name)

    return out


def load_frozen_cost_forecasts(
    path: str,
    *,
    pred_adv_col: str = "pred_adv",
    pred_vol_col: str = "pred_vol",
    train_end_year_col: str = "train_end_year",
) -> pd.DataFrame:
    """Load only the columns required from the frozen OOS forecast artifact."""
    p = Path(path)
    suffix = p.suffix.lower()
    columns = ["Date", "StockID", pred_adv_col, pred_vol_col, train_end_year_col]

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(str(p), columns=columns)
    if suffix == ".csv":
        return pd.read_csv(str(p), usecols=columns)
    raise ValueError("Unsupported transaction-cost forecast file: %s" % p)


def standardize_frozen_cost_forecasts(
    pred_df: pd.DataFrame,
    *,
    pred_adv_col: str = "pred_adv",
    pred_vol_col: str = "pred_vol",
    train_end_year_col: str = "train_end_year",
    require_annual_oos_metadata: bool = True,
) -> pd.DataFrame:
    """Standardize frozen CatBoost forecasts into explicit daily execution units.

    The saved forecasting pipeline produces one OOS block per calendar year and
    stores ``train_end_year = prediction_year - 1``.  That metadata is checked
    here so an in-sample or incorrectly stitched forecast panel cannot silently
    enter the portfolio layer.
    """
    df = _coerce_index(pred_df, frame_name="transaction-cost forecast panel")

    required = [pred_adv_col, pred_vol_col]
    if require_annual_oos_metadata:
        required.append(train_end_year_col)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError("Forecast panel is missing required columns: %s" % missing_cols)

    out = pd.DataFrame(index=df.index)
    out[TC_ADV_DAILY_COL] = _validate_finite_nonnegative(
        df[pred_adv_col],
        name=TC_ADV_DAILY_COL,
        strictly_positive=True,
        allow_missing=False,
    )
    out[TC_SIGMA_DAILY_COL] = _validate_finite_nonnegative(
        df[pred_vol_col],
        name=TC_SIGMA_DAILY_COL,
        strictly_positive=False,
        allow_missing=False,
    )

    if train_end_year_col in df.columns:
        train_end = pd.to_numeric(df[train_end_year_col], errors="coerce")
        if train_end.isna().any() or not np.isfinite(train_end.to_numpy(dtype=float)).all():
            raise RuntimeError("train_end_year contains missing or non-finite values.")
        train_end = train_end.astype(int)
        prediction_year = pd.Series(
            out.index.get_level_values("Date").year,
            index=out.index,
            dtype=int,
        )
        if require_annual_oos_metadata:
            bad = train_end != (prediction_year - 1)
            if bad.any():
                examples = pd.DataFrame(
                    {
                        "train_end_year": train_end[bad],
                        "prediction_year": prediction_year[bad],
                    }
                ).head(5).to_dict("records")
                raise RuntimeError(
                    "Forecast panel violates annual OOS metadata: expected "
                    "train_end_year == prediction_year - 1; examples=%s" % examples
                )
        out[TC_TRAIN_END_YEAR_COL] = train_end

    return out.sort_index()


def attach_frozen_cost_forecasts(
    portfolio_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    pred_adv_col: str = "pred_adv",
    pred_vol_col: str = "pred_vol",
    train_end_year_col: str = "train_end_year",
    require_complete: bool = True,
) -> pd.DataFrame:
    """Attach canonical daily ADV/volatility forecasts to a portfolio panel."""
    panel = _coerce_index(portfolio_df, frame_name="portfolio panel")
    preds = standardize_frozen_cost_forecasts(
        pred_df,
        pred_adv_col=pred_adv_col,
        pred_vol_col=pred_vol_col,
        train_end_year_col=train_end_year_col,
        require_annual_oos_metadata=True,
    )

    cols = [TC_ADV_DAILY_COL, TC_SIGMA_DAILY_COL]
    if TC_TRAIN_END_YEAR_COL in preds.columns:
        cols.append(TC_TRAIN_END_YEAR_COL)
    out = panel.join(preds[cols], how="left")

    if require_complete:
        miss_adv = int(out[TC_ADV_DAILY_COL].isna().sum())
        miss_sigma = int(out[TC_SIGMA_DAILY_COL].isna().sum())
        if miss_adv or miss_sigma:
            raise RuntimeError(
                "Frozen transaction-cost forecast merge is incomplete: "
                "%s missing=%d, %s missing=%d."
                % (TC_ADV_DAILY_COL, miss_adv, TC_SIGMA_DAILY_COL, miss_sigma)
            )

    return out.sort_index()


def resolve_spread_oneway(frame: pd.DataFrame, index: Sequence[object]) -> pd.Series:
    """Return one-way spread cost as a fraction of traded notional."""
    idx = pd.Index(index).astype(str)
    use = frame.copy()
    use.index = pd.Index(use.index).astype(str)
    if use.index.has_duplicates:
        raise RuntimeError("Transaction-cost frame has duplicate asset identifiers after string canonicalization.")

    if "spread_oneway" in use.columns:
        raw = pd.to_numeric(use["spread_oneway"], errors="coerce").reindex(idx)
    elif "bid_ask_bps" in use.columns:
        raw = pd.to_numeric(use["bid_ask_bps"], errors="coerce").reindex(idx) * 1e-4
    else:
        raise RuntimeError(
            "Transaction-cost execution requires spread_oneway or bid_ask_bps; "
            "no default spread is used in the primary path."
        )

    spread = _validate_finite_nonnegative(
        raw,
        name=TC_SPREAD_ONEWAY_COL,
        strictly_positive=False,
        allow_missing=False,
    )
    spread.name = TC_SPREAD_ONEWAY_COL
    return spread


def extract_transaction_cost_inputs(
    frame: pd.DataFrame,
    index: Sequence[object],
) -> TransactionCostInputs:
    """Extract and validate all optimizer execution inputs on one aligned universe."""
    idx = pd.Index(index).astype(str)
    use = frame.copy()
    use.index = pd.Index(use.index).astype(str)
    if use.index.has_duplicates:
        raise RuntimeError("Transaction-cost frame has duplicate asset identifiers after string canonicalization.")

    missing_cols = [c for c in [TC_ADV_DAILY_COL, TC_SIGMA_DAILY_COL] if c not in use.columns]
    if missing_cols:
        raise RuntimeError(
            "Canonical transaction-cost forecast columns are missing: %s. "
            "Enable frozen forecast inputs; historical ADV/weekly-volatility fallbacks are disabled."
            % missing_cols
        )

    adv = _validate_finite_nonnegative(
        pd.to_numeric(use[TC_ADV_DAILY_COL], errors="coerce").reindex(idx),
        name=TC_ADV_DAILY_COL,
        strictly_positive=True,
        allow_missing=False,
    )
    sigma = _validate_finite_nonnegative(
        pd.to_numeric(use[TC_SIGMA_DAILY_COL], errors="coerce").reindex(idx),
        name=TC_SIGMA_DAILY_COL,
        strictly_positive=False,
        allow_missing=False,
    )
    spread = resolve_spread_oneway(use, idx)

    adv.name = TC_ADV_DAILY_COL
    sigma.name = TC_SIGMA_DAILY_COL
    return TransactionCostInputs(
        adv_daily=adv,
        sigma_daily=sigma,
        spread_oneway=spread,
    )


__all__ = [
    "TC_ADV_DAILY_COL",
    "TC_SIGMA_DAILY_COL",
    "TC_SPREAD_ONEWAY_COL",
    "TC_TRAIN_END_YEAR_COL",
    "TransactionCostInputs",
    "attach_frozen_cost_forecasts",
    "extract_transaction_cost_inputs",
    "load_frozen_cost_forecasts",
    "resolve_spread_oneway",
    "standardize_frozen_cost_forecasts",
]
