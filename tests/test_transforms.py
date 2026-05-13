import numpy as np
import pandas as pd

from nasdaq_svar.transforms import apply_transform


def test_log_diff_transform():
    idx = pd.date_range("2020-01-01", periods=4, freq="ME")
    s = pd.Series([100.0, 110.0, 121.0, 133.1], index=idx, name="x")
    out = apply_transform(s, "log_diff")
    assert np.isclose(out.dropna().iloc[0], 100.0 * np.log(1.1))
    assert out.isna().sum() == 1


def test_level_transform():
    idx = pd.date_range("2020-01-01", periods=3, freq="ME")
    s = pd.Series([1.0, 2.0, 3.0], index=idx, name="x")
    out = apply_transform(s, "level")
    pd.testing.assert_series_equal(out, s)
