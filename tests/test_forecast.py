import numpy as np
import pandas as pd

from nasdaq_svar.forecast import _metrics, _residual_diagnostics, _split_series


def test_split_series_keeps_non_empty_valid_and_test():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    s = pd.Series(np.linspace(100.0, 220.0, 120), index=idx, name="NASDAQ_SA")

    split = _split_series(s, train_ratio=0.7, valid_ratio=0.15, min_train_size=60)

    assert len(split.train) >= 60
    assert len(split.valid) > 0
    assert len(split.test) > 0
    assert len(split.train) + len(split.valid) + len(split.test) == 120


def test_metrics_returns_expected_keys():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    y_true = pd.Series([1, 2, 3, 4, 5, 6], index=idx, dtype=float)
    y_pred = pd.Series([1.1, 1.9, 3.2, 4.1, 4.8, 6.2], index=idx, dtype=float)

    out = _metrics(y_true, y_pred)
    assert {"rmse", "mae", "mape", "smape"}.issubset(out.keys())
    assert out["rmse"] >= 0
    assert out["mae"] >= 0


def test_residual_diagnostics_has_required_tests():
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(123)
    resid = pd.Series(rng.normal(0.0, 1.0, size=120), index=idx)

    diag = _residual_diagnostics(resid, lb_lags=12)

    assert {"test", "statistic", "pvalue", "decision_5pct"}.issubset(diag.columns)
    assert {"Ljung-Box", "Jarque-Bera", "Residual moments"}.issubset(set(diag["test"]))
