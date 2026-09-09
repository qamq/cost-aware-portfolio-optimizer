"""Standalone two-rebalance demonstration using synthetic execution inputs."""

import numpy as np
import pandas as pd

from cost_aware_portfolio import (
    CostAwareParams,
    build_pretrade_state,
    build_rebalance_result,
    compute_trade_cost_breakdown,
    solve_cost_aware_nonlinear_pg,
)


def main() -> None:
    assets = pd.Index(["A", "B", "C", "D", "E", "F"])
    alpha = pd.Series([0.030, 0.022, 0.015, 0.006, -0.004, -0.012], index=assets)
    adv = pd.Series([250e6, 180e6, 140e6, 100e6, 80e6, 60e6], index=assets)
    sigma = pd.Series([0.016, 0.018, 0.019, 0.021, 0.024, 0.027], index=assets)
    spread = pd.Series([3, 4, 4, 5, 7, 9], index=assets) / 10_000.0

    vols = np.array([0.016, 0.018, 0.019, 0.021, 0.024, 0.027])
    corr = 0.20 * np.ones((len(assets), len(assets))) + 0.80 * np.eye(len(assets))
    Sigma = np.outer(vols, vols) * corr

    params = CostAwareParams(
        lambda_risk=8.0,
        eta_linear=1.0,
        eta_impact=1.0,
        pos_adv_cap=0.10,
        trade_adv_cap=0.20,
        box_abs=0.40,
        execution_days=1.0,
        step=1.0,
    )
    aum = 50_000_000.0

    # First rebalance begins all-cash.
    pre = build_pretrade_state(date=pd.Timestamp("2026-01-09"), aum=aum)
    w_prev = pd.Series(0.0, index=assets)

    w_post = solve_cost_aware_nonlinear_pg(
        alpha=alpha,
        Sigma=Sigma,
        w_prev=w_prev,
        adv=adv,
        sigma=sigma,
        aum=aum,
        spreads_oneway=spread,
        params=params,
    )
    result = build_rebalance_result(pretrade_state=pre, posttrade_weights=w_post)
    cost = compute_trade_cost_breakdown(
        w=result.posttrade_weights,
        w_prev=result.pretrade_weights,
        spreads_oneway=spread.reindex(result.posttrade_weights.index).fillna(0.0),
        sigma=sigma.reindex(result.posttrade_weights.index).fillna(0.0),
        adv=adv.reindex(result.posttrade_weights.index).fillna(1.0),
        aum=aum,
        impact_kappa=params.impact_kappa,
        impact_exponent=params.impact_exponent,
        execution_days=params.execution_days,
    )

    print("First post-trade weights")
    print(w_post.round(4))
    print(f"one-way turnover: {result.turnover_oneway:.4f}")
    print(f"linear cost:       {1e4 * cost.linear_cost:.2f} bp")
    print(f"impact cost:       {1e4 * cost.impact_cost:.2f} bp")
    print(f"total cost:        {1e4 * cost.total_cost:.2f} bp")
    print(f"solver:            {w_post.attrs.get('solver_method')}")


if __name__ == "__main__":
    main()
