import numpy as np
import pandas as pd

from nasdaq_svar.stage1_stl import run_stage1_stl


def test_stage1_stl_outputs_expected_artifacts():
    idx = pd.bdate_range("2020-01-01", periods=260)
    t = np.arange(len(idx))
    trend = 100 + 0.05 * t
    seasonal = 0.03 * np.sin(2 * np.pi * t / 21)
    noise = 0.005 * np.cos(2 * np.pi * t / 7)
    synthetic = trend * np.exp(seasonal + noise)
    series = pd.Series(synthetic, index=idx, name="NASDAQCOM")

    cfg = {
        "stl_period": 21,
        "seasonal_window": 13,
        "trend_window": None,
        "robust": True,
        "log_input": True,
        "fill_method": "ffill",
    }
    result = run_stage1_stl(series, cfg)

    assert not result.daily_components.empty
    assert not result.monthly_deseasonalized.empty
    assert "seasonal" in result.daily_components.columns
    assert "deseasonalized_level" in result.daily_components.columns

