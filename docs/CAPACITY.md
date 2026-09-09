# Capacity Methodology

Capacity is not estimated by taking one reference portfolio and multiplying its transaction costs by an AUM scaling factor.

The research workflow **re-optimizes at each AUM** because fund size changes both the objective and the feasible set:

- market-impact coefficients increase with AUM;
- position ADV caps tighten in portfolio-weight terms;
- trade ADV caps tighten in portfolio-weight terms;
- holdings and turnover change endogenously; and
- after-cost returns can therefore change for reasons beyond a simple cost multiplier.

## Fixed-AUM analysis

The fixed-AUM adapter evaluates a strategy at a grid such as USD 10m, USD 25m, USD 50m, USD 100m, USD 250m, USD 500m, and USD 1bn. Each point is a separate portfolio run.

## Dynamic NAV

The dynamic-NAV adapter compounds

$$
NAV_{t+1}=NAV_t(1+R_{net,t})
$$

and passes the current beginning-of-period NAV into the next rebalance. AUM-sensitive costs and constraints therefore evolve with the realized portfolio path.

The higher-level adapters require an injected `PortfolioManager`-compatible research engine. They are retained to document the capacity protocol without redistributing the original private data infrastructure.
