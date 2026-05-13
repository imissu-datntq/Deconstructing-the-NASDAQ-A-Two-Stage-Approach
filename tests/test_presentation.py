import numpy as np
import pandas as pd

from nasdaq_svar.presentation import (
    build_chapter1_conclusion,
    build_chapter2_conclusion,
    build_chapter3_conclusion,
    build_chapter4_conclusion,
)


def test_chapter_conclusion_builders_return_markdown():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    raw = {
        "INDPRO": pd.Series(np.linspace(100, 120, 24), index=idx),
        "CPIAUCNS": pd.Series(np.linspace(250, 270, 24), index=idx),
        "FEDFUNDS": pd.Series(np.linspace(1.0, 2.5, 24), index=idx),
        "NASDAQCOM": pd.Series(np.linspace(8000, 12000, 24), index=idx),
    }
    compare = pd.DataFrame(
        {
            "NASDAQ_SA_LAST": np.linspace(8000, 9000, 24),
            "NASDAQ_SA_MEAN": np.linspace(7990, 8990, 24),
        },
        index=idx,
    )
    adf_table = pd.DataFrame(
        {
            "adf_stat": [-3.1, -2.9, -3.8, -4.2],
            "pvalue": [0.02, 0.04, 0.001, 0.0001],
        },
        index=["INDPRO", "CPIAUCNS", "FEDFUNDS", "NASDAQ_SA"],
    )
    transforms = {"INDPRO": "log_diff", "CPIAUCNS": "log_diff", "FEDFUNDS": "level", "NASDAQ_SA": "log_diff"}

    chapter1 = build_chapter1_conclusion(raw, compare, adf_table, transforms)
    assert "Chapter 1" in chapter1

    inv_roots = np.array([0.7 + 0.1j, 0.7 - 0.1j, 0.5 + 0j])
    lag_table = pd.DataFrame([{"aic": 2, "bic": 1, "hqic": 2, "fpe": 2}])
    chapter2 = build_chapter2_conclusion(2, lag_table, inv_roots, 0.12, "aic")
    assert "stable" in chapter2

    irf_table = pd.DataFrame(
        {
            "horizon": list(range(0, 5)),
            "orth_irf": [-0.1, -0.2, -0.3, -0.15, -0.05],
            "lower": [-0.2, -0.3, -0.4, -0.25, -0.1],
            "upper": [0.01, -0.1, -0.2, -0.05, 0.02],
        }
    )
    fevd_table = pd.DataFrame(
        {
            "horizon": [1, 12, 24],
            "shock_INDPRO": [0.2, 0.2, 0.2],
            "shock_CPIAUCNS": [0.2, 0.2, 0.2],
            "shock_FEDFUNDS": [0.2, 0.08, 0.09],
            "shock_NASDAQ_SA": [0.4, 0.52, 0.51],
        }
    )
    chapter3 = build_chapter3_conclusion(irf_table, fevd_table, "NASDAQ_SA", "FEDFUNDS", 0.9)
    assert "Chapter 3" in chapter3
    assert "8.00%" in chapter3

    summary = pd.DataFrame(
        {
            "scenario": ["a", "b", "c"],
            "irf_trough_value": [-0.2, -0.3, -0.1],
            "fevd_policy_share_h12": [0.02, 0.03, 0.01],
        }
    )
    chapter4 = build_chapter4_conclusion(summary)
    assert "Chapter 4" in chapter4

