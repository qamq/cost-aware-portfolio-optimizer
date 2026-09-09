# Stateful Rebalancing

Transaction-cost accounting is based on the portfolio that actually exists immediately before trading.

The canonical sequence is:

1. Start from the previous post-trade portfolio.
2. Drift risky positions through realized holding-period returns.
3. Account for cash/financing and compute current pre-trade weights.
4. Build the optimization universe from current desired names plus nonzero legacy holdings.
5. Solve for current post-trade weights.
6. Compute one authoritative trade vector: `delta_w = posttrade - pretrade`.
7. Use that same vector for turnover, dollar trades, liquidity diagnostics, and trading costs.

The old and new portfolios are aligned on their union. A security therefore cannot disappear simply because it leaves the current signal universe; an exit must appear as an explicit trade.

One-way turnover follows the project convention

$$
\text{turnover}_{1w}=\frac{1}{2}\sum_i|\Delta w_i|.
$$

For a combined long-short portfolio, the low sleeve is signed negative and the high sleeve positive before the authoritative trade vector is constructed. This correctly nets names that migrate directly between sleeves.
