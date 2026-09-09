import numpy as np
import pandas as pd
import pytest

from cost_aware_portfolio import CostAwareParams, compute_trade_cost_breakdown, solve_cost_aware
from cost_aware_portfolio.transaction_costs import extract_transaction_cost_inputs


def test_optimizer_accepts_numeric_asset_identifiers_and_canonicalizes_output():
    idx = pd.Index([101, 202])
    params = CostAwareParams(
        lambda_risk=5.0, pos_adv_cap=10.0, trade_adv_cap=10.0, box_abs=1.0, max_iters=500
    )
    out = solve_cost_aware(
        alpha=pd.Series([0.02, 0.01], index=idx),
        Sigma=np.diag([0.01, 0.02]),
        w_prev=pd.Series([0.5, 0.5], index=idx),
        adv=pd.Series([1e9, 1e9], index=idx),
        sigma=pd.Series([0.02, 0.02], index=idx),
        spreads_oneway=pd.Series([0.0005, 0.0005], index=idx),
        aum=10e6,
        params=params,
    )
    assert list(out.index) == ["101", "202"]
    assert np.isfinite(out.to_numpy()).all()


def test_trade_cost_breakdown_accepts_numeric_asset_identifiers():
    idx = pd.Index([101, 202])
    out = compute_trade_cost_breakdown(
        w=pd.Series([0.55, 0.45], index=idx),
        w_prev=pd.Series([0.50, 0.50], index=idx),
        spreads_oneway=pd.Series([0.001, 0.002], index=idx),
        sigma=pd.Series([0.02, 0.03], index=idx),
        adv=pd.Series([50e6, 100e6], index=idx),
        aum=100e6,
    )
    assert out.total_cost > 0.0


def test_transaction_cost_input_extraction_accepts_numeric_index():
    frame = pd.DataFrame(
        {
            "tc_adv_daily": [50e6, 100e6],
            "tc_sigma_daily": [0.02, 0.03],
            "spread_oneway": [0.001, 0.002],
        },
        index=[101, 202],
    )
    out = extract_transaction_cost_inputs(frame, [101, 202])
    assert list(out.adv_daily.index) == ["101", "202"]
    assert np.allclose(out.adv_daily.to_numpy(), [50e6, 100e6])


def test_duplicate_identifiers_after_string_canonicalization_fail_closed():
    alpha = pd.Series([0.02, 0.01], index=pd.Index([1, "1"], dtype=object))
    with pytest.raises(ValueError, match="duplicate asset identifiers"):
        solve_cost_aware(
            alpha=alpha,
            Sigma=np.eye(2),
            w_prev=pd.Series([0.5, 0.5], index=alpha.index),
            adv=pd.Series([1e9, 1e9], index=alpha.index),
            sigma=pd.Series([0.02, 0.02], index=alpha.index),
            aum=10e6,
        )
