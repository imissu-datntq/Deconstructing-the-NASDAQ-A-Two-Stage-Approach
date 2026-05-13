from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


@dataclass
class Stage1Result:
    daily_components: pd.DataFrame
    monthly_deseasonalized: pd.Series


def _as_odd(value: int, minimum: int) -> int:
    val = max(int(value), minimum)
    if val % 2 == 0:
        val += 1
    return val


def _to_business_daily(series: pd.Series, fill_method: str) -> pd.Series:
    idx = pd.date_range(series.index.min(), series.index.max(), freq="B")
    out = series.reindex(idx)
    if fill_method == "interpolate":
        out = out.interpolate(method="time").ffill().bfill()
    else:
        out = out.ffill().bfill()
    return out


def run_stage1_stl(nasdaq_daily: pd.Series, stage1_cfg: dict[str, Any]) -> Stage1Result:
    series = nasdaq_daily.sort_index().dropna().astype(float)
    if (series <= 0).any():
        raise ValueError("NASDAQ series must be strictly positive for log-based STL.")

    fill_method = str(stage1_cfg.get("fill_method", "ffill"))
    series = _to_business_daily(series, fill_method=fill_method)

    log_input = bool(stage1_cfg.get("log_input", True))
    observed = np.log(series) if log_input else series.copy()

    period = int(stage1_cfg.get("stl_period", 21))
    seasonal = _as_odd(int(stage1_cfg.get("seasonal_window", 13)), minimum=7)
    trend_window = stage1_cfg.get("trend_window")
    if trend_window is not None:
        trend_window = _as_odd(int(trend_window), minimum=period + 1)

    stl = STL(
        observed,
        period=period,
        seasonal=seasonal,
        trend=trend_window,
        robust=bool(stage1_cfg.get("robust", True)),
    )

    fit_kwargs: dict[str, int] = {}
    if stage1_cfg.get("inner_iter") is not None:
        fit_kwargs["inner_iter"] = int(stage1_cfg["inner_iter"])
    if stage1_cfg.get("outer_iter") is not None:
        fit_kwargs["outer_iter"] = int(stage1_cfg["outer_iter"])
    result = stl.fit(**fit_kwargs)

    deseasonalized = observed - result.seasonal
    deseasonalized_level = np.exp(deseasonalized) if log_input else deseasonalized

    components = pd.DataFrame(
        {
            "observed_level": series,
            "observed_transformed": observed,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
            "deseasonalized_transformed": deseasonalized,
            "deseasonalized_level": deseasonalized_level,
            "robust_weight": result.weights,
        },
        index=series.index,
    )

    monthly_sa = components["deseasonalized_level"].resample("ME").last().dropna()
    monthly_sa.name = "NASDAQ_SA"
    return Stage1Result(daily_components=components, monthly_deseasonalized=monthly_sa)
