"""Re-solve one synthetic target-tracking rebalance across an AUM grid."""

import numpy as np
import pandas as pd

from cost_aware_portfolio import CostAwareParams, compute_trade_cost_breakdown, solve_rebalance_to_target


def main() -> None:
    assets = pd.Index([f"Stock{i}" for i in range(8)])
    adv = pd.Series([500e6, 400e6, 300e6, 250e6, 200e6, 150e6, 120e6, 100e6], index=assets)
    sigma = pd.Series(np.linspace(0.015, 0.028, len(assets)), index=assets)
    spread = pd.Series(np.linspace(3, 10, len(assets)), index=assets) / 10_000.0
    w_prev = pd.Series(1.0 / len(assets), index=assets)
    w_target = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10, 0.0, 0.0, 0.0], index=assets)

    params = CostAwareParams(
        eta_linear=1.0,
        eta_impact=1.0,
        pos_adv_cap=0.50,
        trade_adv_cap=0.20,
        box_abs=0.30,
        target_penalty=100.0,
        execution_days=1.0,
    )

    rows = []
    for aum in [10e6, 25e6, 50e6, 100e6, 250e6, 500e6, 1e9]:
        w = solve_rebalance_to_target(
            w_target=w_target,
            w_prev=w_prev,
            adv=adv,
            sigma=sigma,
            aum=aum,
            spreads_oneway=spread,
            params=params,
        )
        cost = compute_trade_cost_breakdown(
            w=w,
            w_prev=w_prev,
            spreads_oneway=spread,
            sigma=sigma,
            adv=adv,
            aum=aum,
            impact_kappa=params.impact_kappa,
            impact_exponent=params.impact_exponent,
            execution_days=params.execution_days,
        )
        rows.append(
            {
                "AUM ($m)": aum / 1e6,
                "one-way turnover": 0.5 * float((w - w_prev).abs().sum()),
                "cost (bp)": 1e4 * cost.total_cost,
                "max target gap": float((w - w_target).abs().max()),
                "converged": bool(w.attrs.get("solver_converged")),
            }
        )

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()
