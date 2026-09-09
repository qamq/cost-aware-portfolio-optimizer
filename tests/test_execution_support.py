"""Execution-support behavior for screened-out legacy holdings."""

import numpy as np
import pandas as pd

from cost_aware_portfolio.strategies import EWTCStrategy

class _SupportContext:
    tc_aum_dollars = 100_000_000.0

    def __init__(self, support_panel):
        self.execution_support_df = support_panel

    def _tc_execution_frame(self, date):
        out = self.execution_support_df.loc[pd.Timestamp(date)].copy()
        out.index = pd.Index(out.index).astype(str)
        return out

    @staticmethod
    def _decile_bounds(up, cut, decile_idx):
        low = float(np.percentile(up, decile_idx * 100.0 / cut))
        high = float(np.percentile(up, (decile_idx + 1) * 100.0 / cut))
        return low, high

    @staticmethod
    def _decile_mask(up, low, high, is_lowest):
        return ((up >= low) & (up <= high)) if is_lowest else ((up > low) & (up <= high))


def test_ew_tc_uses_support_panel_to_exit_name_removed_by_screen(monkeypatch):
    ret_name = "next_week_ret"
    d0 = pd.Timestamp("2024-01-05")
    d1 = pd.Timestamp("2024-01-12")
    ids = [str(i) for i in range(20)]

    p0 = np.linspace(0.01, 0.99, 20)
    p1 = p0.copy()
    p1[16], p1[17], p1[18], p1[19] = 0.98, 0.99, 0.40, 0.41

    support_rows = []
    for date, probs in [(d0, p0), (d1, p1)]:
        for sid, prob in zip(ids, probs):
            r = 0.10 if (date == d0 and sid == "18") else 0.0
            support_rows.append((date, sid, prob, r))
    support = pd.DataFrame(
        support_rows, columns=["Date", "StockID", "up_prob", ret_name]
    ).set_index(["Date", "StockID"]).sort_index()

    # Screened selection panel deliberately drops old top-decile names 18/19 on d1.
    screened = support.copy()
    screened = screened.drop(index=[(d1, "18"), (d1, "19")])

    ctx = _SupportContext(support)
    strategy = EWTCStrategy(ctx)
    prev_state = {}

    monkeypatch.setattr(strategy, "_paired_common_long_only_sum", lambda **kwargs: 1.0)
    calls = []

    def _fake_solve(*, ctx, frame, target_w, prev_w, desired_target_sum=None):
        calls.append({"frame": frame.copy(), "target": target_w.copy(), "prev": prev_w.copy()})
        return target_w.copy(), 0.0

    monkeypatch.setattr(strategy, "_solve_tc_target_tracking", _fake_solve)

    strategy.compute_for_date(screened, d0, 10, ret_name, prev_state)
    first_count = len(calls)
    strategy.compute_for_date(screened, d1, 10, ret_name, prev_state)

    high_call = calls[first_count:][9]
    assert "18" in high_call["frame"].index
    assert "19" in high_call["frame"].index
    assert np.isclose(high_call["target"].get("18", 0.0), 0.0)
    assert np.isclose(high_call["target"].get("19", 0.0), 0.0)
    assert high_call["prev"]["18"] > high_call["prev"]["19"]

    result = prev_state["tc_rebalance_results"]["ew_decile_9"]
    assert result.delta_w["18"] < 0.0
    assert result.delta_w["19"] < 0.0
