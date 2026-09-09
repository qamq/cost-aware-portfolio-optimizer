# Objective Function

The alpha-driven optimizer trades off signal reward, covariance risk, linear spread cost, and nonlinear market impact.

For pre-trade weights \(w_{prev}\), proposed weights \(w\), and \(\Delta w=w-w_{prev}\):

\[
\max_w \; \alpha^T w - \lambda w^T\Sigma w
- \eta_L\sum_i s_i|\Delta w_i|
- \eta_I\sum_i c_i|\Delta w_i|^p.
\]

The implementation minimizes the smooth equivalent

\[
J(w)=\frac{1}{2}w^TGw-\alpha^Tw
+\eta_L\sum_i s_i\sqrt{\Delta w_i^2+\varepsilon^2}
+\eta_I\sum_i c_i(\Delta w_i^2+\varepsilon^2)^{p/2},
\]

with \(G=2\lambda\Sigma\).

## Market impact

The implementation constructs

\[
c_i=\kappa\sigma_i\sqrt{\frac{AUM}{D\,ADV_i}},
\]

where `ADV_i` is forecast daily dollar volume, `sigma_i` is forecast daily return volatility, and `D` is the execution horizon in trading days.

The realized diagnostic cost is reported separately as linear spread cost, nonlinear impact cost, and total cost.

## Two optimization modes

### Alpha-driven

`solve_cost_aware_nonlinear_pg(...)` chooses the portfolio endogenously from signal strength, risk, and implementation costs.

### Target tracking

`solve_rebalance_to_target_nonlinear_pg(...)` replaces the free alpha objective with a quadratic penalty around a desired portfolio. It asks for the cheapest feasible move toward a fixed target after accounting for spread, impact, and liquidity constraints.
