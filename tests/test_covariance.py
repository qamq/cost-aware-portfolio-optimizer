import numpy as np
import pandas as pd

from cost_aware_portfolio.covariance import CovarianceEngine


def test_covariance_is_symmetric_and_psd_after_stabilization():
    rng = np.random.RandomState(7)
    dates = pd.date_range("2025-01-01", periods=40, freq="W")
    rows = []
    for d in dates:
        for stock in ["A", "B", "C"]:
            rows.append({"Date": d, "StockID": stock, "ret": rng.normal(0, 0.02)})
    frame = pd.DataFrame(rows)

    eng = CovarianceEngine(
        mw_window=20,
        mw_min_periods=8,
        mw_ridge=1e-8,
        cov_mode="rolling",
        ewma_alpha=0.06,
        ewma_center=False,
    )
    eng.attach_returns_panel(frame, "ret")
    sigma, cols = eng.build_cov(dates[-1], pd.Index(["A", "B", "C"]))
    assert cols == ["A", "B", "C"]
    assert np.allclose(sigma, sigma.T, atol=1e-12)
    assert np.linalg.eigvalsh(sigma).min() >= -1e-10
