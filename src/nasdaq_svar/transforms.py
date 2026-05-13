from __future__ import annotations

import numpy as np
import pandas as pd


def to_month_end(series: pd.Series) -> pd.Series:
    s = series.dropna().copy()
    s.index = pd.PeriodIndex(s.index, freq="M").to_timestamp("M")
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def apply_transform(series: pd.Series, method: str) -> pd.Series:
    m = method.lower()
    if m == "level":
        out = series.copy()
    elif m == "diff":
        out = series.diff()
    elif m == "log":
        out = np.log(series)
    elif m == "log_diff":
        out = 100.0 * np.log(series).diff()
    elif m == "pct_change":
        out = 100.0 * series.pct_change()
    elif m == "yoy_log_diff":
        out = 100.0 * (np.log(series) - np.log(series.shift(12)))
    else:
        raise ValueError(f"Unsupported transform method: {method}")
    out.name = series.name
    return out


def build_stage2_dataset(
    monthly_frame: pd.DataFrame,
    transforms: dict[str, str],
    ordering: list[str],
) -> pd.DataFrame:
    transformed = {}
    for col in ordering:
        if col not in monthly_frame.columns:
            raise KeyError(f"Missing column in monthly input frame: {col}")
        method = transforms.get(col, "level")
        transformed[col] = apply_transform(monthly_frame[col], method=method)
    out = pd.DataFrame(transformed).dropna().sort_index()
    return out

