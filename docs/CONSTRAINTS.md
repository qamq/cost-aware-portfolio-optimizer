# Liquidity and Portfolio Constraints

The optimizer combines soft trading-cost penalties with hard feasibility constraints.

## Static box constraint

`box_abs` limits absolute portfolio weight per security.

## Position ADV cap

The maximum position size is constrained by daily dollar ADV:

$$
|w_i| \leq \text{pos\_adv\_cap}\frac{ADV_i}{AUM}.
$$

The position cap is an inventory/capacity constraint and intentionally uses daily ADV.

## Trade ADV cap

The maximum one-rebalance change is

$$
|w_i-w_{prev,i}|
\leq
\text{trade\_adv\_cap}\frac{D\,ADV_i}{AUM}.
$$

This is an execution-speed constraint. Increasing `execution_days` expands the amount that may be traded and also changes the market-impact coefficient through execution-window volume.

## Long-only and dollar-neutral modes

The solver supports long-only and signed portfolios, explicit target sums, and a gross-exposure target for long-short applications.

## Partial fills

For long-only sleeves, the requested total exposure may be unreachable because of current position and trade caps. When `allow_partial_long_only=True`, the solver uses the nearest feasible total exposure rather than failing or pretending the target was executed in full.

## Legacy holdings

A previous holding already outside a new static cap is grandfathered sufficiently to permit an orderly move back toward feasibility within the current trade cap. It is not forced to jump immediately to the new cap.
