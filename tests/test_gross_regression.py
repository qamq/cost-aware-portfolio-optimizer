import numpy as np
import pandas as pd
import pytest

from cost_aware_portfolio.regression import (
    assert_gross_return_series_unchanged,
    compare_gross_return_frames,
    compare_gross_return_series,
)


def _series(values):
    idx = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"])
    return pd.Series(values, index=idx, dtype=float)


def test_exact_gross_stream_passes():
    ref = _series([0.01, -0.02, 0.03, 0.00])
    cur = ref.copy()
    result = assert_gross_return_series_unchanged(ref, cur)
    assert result.passed
    assert result.max_abs_diff == 0.0
    assert result.missing_from_current == 0
    assert result.extra_in_current == 0


def test_material_gross_change_fails_closed():
    ref = _series([0.01, -0.02, 0.03, 0.00])
    cur = ref.copy()
    cur.iloc[2] += 1e-4
    result = compare_gross_return_series(ref, cur, atol=1e-12, rtol=1e-10)
    assert not result.passed
    assert np.isclose(result.max_abs_diff, 1e-4)
    with pytest.raises(AssertionError, match="Gross regression failed"):
        assert_gross_return_series_unchanged(ref, cur, atol=1e-12, rtol=1e-10)


def test_date_mismatch_fails_when_required():
    ref = _series([0.01, -0.02, 0.03, 0.00])
    cur = ref.iloc[1:].copy()
    result = compare_gross_return_series(ref, cur, require_same_dates=True)
    assert not result.passed
    assert result.missing_from_current == 1


def test_frame_comparison_returns_one_row_per_requested_column():
    ref = pd.DataFrame({"Low": _series([0.0, 0.1, 0.0, 0.1]), "H-L": _series([0.1, 0.2, 0.3, 0.4])})
    cur = ref.copy()
    out = compare_gross_return_frames(ref, cur, columns=["Low", "H-L"])
    assert list(out["label"]) == ["Low", "H-L"]
    assert bool(out["passed"].all())
