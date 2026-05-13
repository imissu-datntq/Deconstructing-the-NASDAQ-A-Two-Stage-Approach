import numpy as np
import pandas as pd

from nasdaq_svar.stage2_svar import run_stage2_recursive_svar


def test_stage2_irf_contains_confidence_bands():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-01-31", periods=180, freq="ME")
    eps = rng.normal(size=(180, 4))
    x = np.zeros((180, 4))
    for t in range(1, 180):
        x[t] = 0.35 * x[t - 1] + eps[t]
    df = pd.DataFrame(x, index=idx, columns=["INDPRO", "CPIAUCNS", "FEDFUNDS", "NASDAQ_SA"])

    cfg = {
        "maxlags": 8,
        "ic": "aic",
        "fallback_lag": 2,
        "irf_horizon": 12,
        "fevd_horizon": 12,
        "compute_irf_bands": True,
        "irf_confidence_level": 0.9,
        "irf_errband_repl": 50,
        "irf_errband_burn": 50,
        "irf_errband_seed": 123,
        "policy_shock_variable": "FEDFUNDS",
        "equity_response_variable": "NASDAQ_SA",
    }

    result = run_stage2_recursive_svar(df, cfg)
    assert result.selected_lag >= 1
    assert {"orth_irf", "lower", "upper"}.issubset(result.irf_table.columns)
    assert result.irf_table.shape[0] == 13
    assert result.irf_confidence_level == 0.9
    assert not np.allclose(result.irf_table["lower"], result.irf_table["upper"])
