# Research Integration

The public package deliberately separates the standalone optimization layer from the original private market-data and portfolio-orchestration stack.

## Standalone components

The following modules can be inspected and tested independently of the original research database:

- `optimizer.py`
- `covariance.py`
- `state.py`
- `transaction_costs.py`
- `calculations.py`
- `validation.py`
- `regression.py`

`strategies.py` and `diagnostics.py` operate against a portfolio-context interface and are retained as the integration layer around those primitives.

## Capacity adapters

`capacity.py` and `dynamic_nav.py` preserve the research protocols for fixed-AUM re-optimization and evolving-NAV simulation. They intentionally require a `PortfolioManager`-compatible class to be supplied through `manager_cls`.

The original project-specific `PortfolioManager`, data assembler, annual runner, and process-parallel wrappers are not redistributed in this standalone repository because they depend on private data-access and project infrastructure rather than defining the cost-aware optimization method itself.

This boundary keeps the public repository executable without creating fake substitutes for licensed data or private research services.
