# Cost-Aware Portfolio Optimizer

**Stateful portfolio rebalancing under spread costs, nonlinear market impact, and liquidity constraints**

This repository contains a standalone implementation of the portfolio-construction and execution layer developed for a broader cross-sectional equity research project.

The central problem is not simply to find the portfolio with the strongest signal exposure. It is to decide **how far to trade from the portfolio that is actually held** once risk, spread crossing, nonlinear market impact, limited liquidity, and implementation capacity are taken seriously.

The package therefore treats portfolio construction as a stateful constrained optimization problem. Previous holdings drift through realized returns, the next rebalance begins from those pre-trade weights, and the optimizer chooses a feasible post-trade portfolio subject to explicit position and trade-participation limits.

The core implementation includes:

- alpha-driven cost-aware optimization;
- target-tracking rebalancing for equal- or value-weight targets;
- linear one-way spread costs;
- nonlinear volatility/liquidity market impact;
- ADV-based position and trade caps;
- partial fills when a desired long-only portfolio cannot be reached in one rebalance;
- stateful holdings and explicit exit trades;
- rolling and EWMA covariance estimation with numerical stabilization;
- projected L-BFGS and projected-gradient solvers with Armijo safeguards;
- transaction-cost decomposition and implementation diagnostics; and
- fixed-AUM and dynamic-NAV research adapters for capacity analysis.

---

## Economic Objective

For the alpha-driven portfolio, the optimizer chooses new weights $w$ relative to pre-trade weights $w_{t^-}$. Define the trade vector

```math
\Delta w = w - w_{t^-}.
```

The economic objective is

```math
\max_w
\quad
\alpha^\top w
- \lambda_{\mathrm{risk}} w^\top \Sigma w
- \eta_{\mathrm{linear}}\sum_i s_i |\Delta w_i|
- \eta_{\mathrm{impact}}\sum_i c_i |\Delta w_i|^p,
```

where:

- $\alpha$ is the expected-return or ranking signal;
- $\Sigma$ is the covariance matrix;
- $s_i$ is the one-way spread cost;
- $c_i$ is a stock-specific market-impact coefficient;
- $p$ is the nonlinear impact exponent; and
- $\lambda_{\mathrm{risk}}$, $\eta_{\mathrm{linear}}$, and $\eta_{\mathrm{impact}}$ control the trade-off among signal, risk, and implementation cost.

The implementation uses the smooth minimization equivalent

```math
J(w)
=
\frac{1}{2}w^\top G w
- \alpha^\top w
+ \eta_{\mathrm{linear}}\sum_i s_i\sqrt{\Delta w_i^2+\varepsilon^2}
+ \eta_{\mathrm{impact}}\sum_i c_i(\Delta w_i^2+\varepsilon^2)^{p/2},
```

with $G=2\lambda_{\mathrm{risk}}\Sigma$. The smoothing parameter $\varepsilon$ is numerical only; it does not change the intended economic interpretation of the cost terms.

The market-impact coefficient is constructed from forecast daily volatility and forecast execution liquidity:

```math
c_i
=
\kappa\,\sigma_i
\sqrt{\frac{AUM}{D\cdot ADV_i}},
```

where $D$ is the number of execution days. Higher volatility, lower liquidity, or larger AUM therefore makes a given weight change more expensive.

---

## Liquidity Constraints

Transaction costs are not treated only as soft penalties. Liquidity also restricts the feasible portfolio.

### Position capacity

A stock's total position is capped as a fraction of daily ADV:

```math
|w_i|
\leq
\min\left(
\text{box cap},
\frac{\text{position ADV cap}\times ADV_i}{AUM}
\right).
```

### Trade participation

A single rebalance cannot trade more than a chosen fraction of the execution-window volume:

```math
|w_i-w_{t^-,i}|
\leq
\frac{\text{trade ADV cap}\times D\times ADV_i}{AUM}.
```

These are distinct constraints: the first limits **inventory**, while the second limits **how quickly the portfolio can move**.

When a long-only target is infeasible under the current trade caps, the optimizer can perform a partial fill and carry the remaining position adjustment into future rebalances rather than silently assuming immediate execution.

---

## Stateful Rebalancing

The repository uses one authoritative portfolio transition:

```text
previous post-trade weights
        ↓
realized holding-period returns
        ↓
current pre-trade weights
        ↓
optimization / target tracking
        ↓
current post-trade weights
        ↓
Δw = post-trade − pre-trade
```

Previous and current holdings are aligned on their **union**. A security that leaves the signal universe does not disappear from the book; it remains a holding until an explicit trade reduces it toward zero.

This matters for turnover and transaction-cost accounting. The same executed trade vector is used for liquidity constraints, dollar trades, turnover, and realized cost diagnostics.

---

## Numerical Methods

The alpha-driven problem uses **projected L-BFGS with Armijo backtracking** when the configured problem is in the intended strongly convex regime.

Each candidate step is projected back onto the same feasible set, so the numerical acceleration does not alter the economic constraints. A safeguarded projected-gradient path remains available when the quasi-Newton update is unsuitable.

The target-tracking problem has a simpler quadratic block and uses **projected gradient descent with a Barzilai-Borwein step initialization and Armijo backtracking**.

The package also includes covariance stabilization through:

- symmetrization;
- repair of invalid diagonal entries;
- cleanup of undefined off-diagonal entries from sparse overlap;
- spectral positive-semidefinite repair; and
- a small configurable diagonal ridge.

---

## Capacity Analysis

Capacity is evaluated by **re-optimizing the portfolio at each AUM level** rather than scaling the transaction costs from one reference portfolio.

Changing AUM changes:

- nonlinear impact coefficients;
- position-cap feasibility;
- trade-cap feasibility;
- realized turnover;
- holdings; and
- the resulting return stream.

The research adapters support both:

1. **Fixed-AUM capacity analysis** — repeatedly solve the same strategy at a grid of permanent fund sizes.
2. **Dynamic-NAV simulation** — allow NAV to evolve through time and feed the current capital base into each subsequent rebalance.

The standalone public package requires the user to inject a compatible research portfolio manager into these higher-level adapters. The core optimizer itself has no dependency on the original private data pipeline.

---

## Repository Structure

```text
cost-aware-portfolio-optimizer/
│
├── src/
│   └── cost_aware_portfolio/
│       ├── optimizer.py           # Cost-aware objective, constraints, and solvers
│       ├── covariance.py          # Rolling/EWMA covariance estimation and PSD repair
│       ├── state.py               # Pre-trade/post-trade portfolio state transitions
│       ├── transaction_costs.py   # Canonical execution inputs and unit validation
│       ├── strategies.py          # Portfolio strategy integration layer
│       ├── diagnostics.py         # Implementation and cost diagnostics
│       ├── calculations.py        # Shared portfolio calculations
│       ├── capacity.py            # Fixed-AUM research adapter
│       ├── dynamic_nav.py         # Dynamic-NAV research adapter
│       ├── validation.py          # HAC, block-bootstrap, SPA, and Sharpe inference
│       ├── regression.py          # Gross-return regression checks
│       └── fingerprint.py         # Deterministic run/checkpoint fingerprints
│
├── examples/
│   ├── synthetic_rebalance.py
│   └── capacity_snapshot.py
│
├── tests/
├── docs/
└── data/
```

---

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

Activate the environment.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the package and test dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Run the standalone synthetic examples:

```bash
python examples/synthetic_rebalance.py
python examples/capacity_snapshot.py
```

---

## Minimal Example

```python
import numpy as np
import pandas as pd

from cost_aware_portfolio import (
    CostAwareParams,
    compute_trade_cost_breakdown,
    solve_cost_aware,
)

assets = pd.Index(["A", "B", "C", "D", "E", "F"])
alpha = pd.Series([0.030, 0.022, 0.015, 0.006, -0.004, -0.012], index=assets)
w_prev = pd.Series(0.0, index=assets)
adv = pd.Series([250e6, 180e6, 140e6, 100e6, 80e6, 60e6], index=assets)
sigma = pd.Series([0.016, 0.018, 0.019, 0.021, 0.024, 0.027], index=assets)
spread = pd.Series([3, 4, 4, 5, 7, 9], index=assets) / 10_000

vols = sigma.to_numpy()
corr = 0.20 * np.ones((len(assets), len(assets))) + 0.80 * np.eye(len(assets))
Sigma = np.outer(vols, vols) * corr

params = CostAwareParams(
    lambda_risk=8.0,
    eta_linear=1.0,
    eta_impact=1.0,
    pos_adv_cap=0.10,
    trade_adv_cap=0.20,
    box_abs=0.40,
    step=1.0,
)

w = solve_cost_aware(
    alpha=alpha,
    Sigma=Sigma,
    w_prev=w_prev,
    adv=adv,
    sigma=sigma,
    aum=50_000_000,
    spreads_oneway=spread,
    params=params,
)

cost = compute_trade_cost_breakdown(
    w=w,
    w_prev=w_prev,
    spreads_oneway=spread,
    sigma=sigma,
    adv=adv,
    aum=50_000_000,
)

print(w)
print(cost)
```

---

## Data and Units

The primary transaction-cost path expects explicit execution units:

| Input | Unit | Interpretation |
|---|---|---|
| `adv` | dollars/day | Forecast average daily dollar volume |
| `sigma` | daily return decimal | Forecast daily volatility |
| `spreads_oneway` | fraction of notional | One-way spread cost |
| `aum` | dollars | Portfolio assets under management |
| `w`, `w_prev` | portfolio weights | Post- and pre-trade risky weights |

The original research used out-of-sample forecasts of forward ADV and volatility as execution inputs. This public repository does not redistribute the underlying licensed financial data or forecast artifacts.

See [`docs/DATA.md`](docs/DATA.md) for details.

---

## Documentation

- [`docs/OBJECTIVE.md`](docs/OBJECTIVE.md) — objective function and economic interpretation
- [`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md) — box, position-ADV, trade-ADV, and partial-fill logic
- [`docs/STATEFUL_REBALANCING.md`](docs/STATEFUL_REBALANCING.md) — drift, portfolio state, and turnover accounting
- [`docs/CAPACITY.md`](docs/CAPACITY.md) — fixed-AUM and dynamic-NAV methodology
- [`docs/DATA.md`](docs/DATA.md) — input units and data boundaries
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — public/private boundary and reproducibility scope
- [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — boundary between the standalone package and private research infrastructure

---

## Research Scope and Limitations

This repository is an empirical research implementation, not a live execution system.

Important limitations include:

- market impact is modeled rather than estimated from proprietary realized executions;
- spread and liquidity inputs depend on the quality of the supplied forecasts or estimates;
- borrow availability, stock-loan fees, queue position, intraday execution scheduling, and venue selection are outside the standalone model;
- the public package does not include the licensed market/accounting datasets used in the original research; and
- historical backtest performance is not evidence that future performance will persist.

---

## Use and Rights

Copyright © 2026 Quinn McMurtry. All rights reserved.

This repository is made publicly available for research review and portfolio demonstration. **No open-source license is granted.** Unless a license is added later, reuse, redistribution, or incorporation of the source code into other projects requires permission except where otherwise permitted by applicable law.

---

## Disclaimer

This repository is provided for research and educational purposes only. It does not constitute investment advice, a recommendation, or an offer to buy or sell any security.
