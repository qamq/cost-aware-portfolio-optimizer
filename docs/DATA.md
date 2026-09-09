# Data and Units

The standalone optimizer does not require a specific vendor or security identifier system. It expects aligned pandas Series and a covariance matrix.

Asset identifiers are canonicalized to strings by the state/execution layer. Use identifiers whose string representations are unique (for example, do not mix integer `1` and string `"1"` as distinct assets).

## Core inputs

| Field | Unit | Requirement |
|---|---|---|
| `alpha` | arbitrary comparable score / expected return | finite and aligned |
| `Sigma` | return covariance | square, finite, aligned to assets |
| `w_prev` | portfolio weight | finite |
| `adv` | dollars per day | strictly positive |
| `sigma` | daily return decimal | non-negative |
| `spreads_oneway` | fraction of traded notional | non-negative |
| `aum` | dollars | positive |

A 5 bp one-way spread should be supplied as `0.0005`, not `5.0`.

## Original research boundary

The broader research project used licensed market and accounting data and frozen out-of-sample forecasts of forward daily ADV and volatility. Those datasets and forecast artifacts are not distributed here.

The public examples use synthetic data only.
