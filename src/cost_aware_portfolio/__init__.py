"""Cost-aware portfolio construction with stateful rebalancing and liquidity constraints."""

from .optimizer import (
    CostAwareParams,
    TradeCostBreakdown,
    compute_long_only_feasible_interval,
    compute_trade_cost_breakdown,
    solve_cost_aware_nonlinear_pg,
    solve_rebalance_to_target_nonlinear_pg,
)
from .covariance import CovarianceEngine
from .state import (
    DriftResult,
    PreTradeState,
    RebalanceResult,
    build_pretrade_state,
    build_rebalance_result,
    build_rebalance_universe,
    compute_trade_vector,
    drift_posttrade_portfolio,
)
from .transaction_costs import TransactionCostInputs

__all__ = [
    "CostAwareParams",
    "TradeCostBreakdown",
    "CovarianceEngine",
    "TransactionCostInputs",
    "DriftResult",
    "PreTradeState",
    "RebalanceResult",
    "compute_long_only_feasible_interval",
    "compute_trade_cost_breakdown",
    "solve_cost_aware_nonlinear_pg",
    "solve_rebalance_to_target_nonlinear_pg",
    "solve_cost_aware",
    "solve_rebalance_to_target",
    "build_pretrade_state",
    "build_rebalance_result",
    "build_rebalance_universe",
    "compute_trade_vector",
    "drift_posttrade_portfolio",
]

# Public convenience aliases; original function names remain available for backward compatibility.
solve_cost_aware = solve_cost_aware_nonlinear_pg
solve_rebalance_to_target = solve_rebalance_to_target_nonlinear_pg
