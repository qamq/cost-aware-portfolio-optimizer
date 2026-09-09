import importlib


def test_all_public_modules_import():
    modules = [
        "cost_aware_portfolio",
        "cost_aware_portfolio.calculations",
        "cost_aware_portfolio.capacity",
        "cost_aware_portfolio.covariance",
        "cost_aware_portfolio.diagnostics",
        "cost_aware_portfolio.dynamic_nav",
        "cost_aware_portfolio.fingerprint",
        "cost_aware_portfolio.optimizer",
        "cost_aware_portfolio.regression",
        "cost_aware_portfolio.state",
        "cost_aware_portfolio.strategies",
        "cost_aware_portfolio.transaction_costs",
        "cost_aware_portfolio.validation",
    ]
    for name in modules:
        importlib.import_module(name)
