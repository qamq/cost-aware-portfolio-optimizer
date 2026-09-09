# Reproducibility

The public repository is designed to make the **optimization mechanics** independently inspectable and executable without access to the original licensed equity database.

Publicly reproducible components include:

- nonlinear cost-aware objective and gradients;
- feasibility projection;
- projected L-BFGS / projected-gradient numerical routines;
- position and trade ADV constraints;
- stateful rebalance accounting;
- covariance stabilization;
- transaction-cost decomposition;
- statistical validation utilities; and
- synthetic examples and tests.

Not redistributed:

- CRSP, Compustat, WRDS, or other licensed data;
- saved private model forecasts;
- private data-access helpers;
- credentials or local filesystem configuration; and
- the original full research portfolio manager/data-assembly stack.

The `capacity.py` and `dynamic_nav.py` modules expose the original high-level research protocol but require a compatible manager class to be injected by the user.
