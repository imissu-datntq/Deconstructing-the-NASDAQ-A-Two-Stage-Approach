from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import ConfigDict, load_config, resolve_paths
from .reporting import write_dataframe, write_text


@dataclass
class ForecastSplit:
    train: pd.Series
    valid: pd.Series
    test: pd.Series


@dataclass
class CandidateSpec:
    family: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]


def _read_target_series(csv_path: Path, target_column: str | None) -> pd.Series:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    date_col = df.columns[0]
    if target_column and target_column in df.columns:
        value_col = target_column
    elif len(df.columns) >= 2:
        value_col = df.columns[1]
    else:
        raise ValueError("Could not infer target column for forecasting.")

    parsed_index = pd.DatetimeIndex(pd.to_datetime(df[date_col], errors="coerce"))
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy()
    series = pd.Series(values, index=parsed_index, name=value_col).dropna()
    series = series.sort_index().astype(float)
    series = series[~series.index.duplicated(keep="last")]

    inferred = None
    if len(series.index) >= 3:
        try:
            inferred = pd.infer_freq(series.index)
        except ValueError:
            inferred = None
    freq = inferred or "ME"
    series = series.asfreq(freq).ffill()
    return series


def _split_series(series: pd.Series, train_ratio: float, valid_ratio: float, min_train_size: int) -> ForecastSplit:
    n_obs = len(series)
    if n_obs < max(min_train_size + 12, 48):
        raise ValueError(
            f"Not enough observations for robust forecasting split. Found {n_obs}, required >= {max(min_train_size + 12, 48)}."
        )

    train_end = int(n_obs * train_ratio)
    valid_end = train_end + int(n_obs * valid_ratio)

    train_end = max(train_end, min_train_size)
    valid_end = min(valid_end, n_obs - 1)

    train = series.iloc[:train_end]
    valid = series.iloc[train_end:valid_end]
    test = series.iloc[valid_end:]

    if len(valid) == 0 or len(test) == 0:
        raise ValueError("Validation and test splits must be non-empty.")

    return ForecastSplit(train=train, valid=valid, test=test)


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
    err = aligned_true - aligned_pred
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))

    denom_mape = aligned_true.abs().replace(0.0, np.nan)
    mape = float((np.abs(err) / denom_mape).dropna().mean() * 100.0)

    denom_smape = (aligned_true.abs() + aligned_pred.abs()).replace(0.0, np.nan)
    smape = float((2.0 * np.abs(err) / denom_smape).dropna().mean() * 100.0)

    return {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape}


def _build_candidates(model_space: ConfigDict) -> list[CandidateSpec]:
    p_values = [int(x) for x in model_space.get("p_values", [0, 1, 2])]
    d_values = [int(x) for x in model_space.get("d_values", [0, 1])]
    q_values = [int(x) for x in model_space.get("q_values", [0, 1, 2])]

    P_values = [int(x) for x in model_space.get("P_values", [0, 1])]
    D_values = [int(x) for x in model_space.get("D_values", [0, 1])]
    Q_values = [int(x) for x in model_space.get("Q_values", [0, 1])]
    seasonal_period = int(model_space.get("seasonal_period", 12))

    include_arima = bool(model_space.get("include_arima", True))
    include_sarima = bool(model_space.get("include_sarima", True))
    max_candidates = int(model_space.get("max_candidates", 80))

    out: list[CandidateSpec] = []

    if include_arima:
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    out.append(CandidateSpec("ARIMA", (p, d, q), (0, 0, 0, 0)))

    if include_sarima:
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                if P == 0 and D == 0 and Q == 0:
                                    continue
                                out.append(CandidateSpec("SARIMA", (p, d, q), (P, D, Q, seasonal_period)))

    return out[:max_candidates]


def _fit_one(
    train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> Any:
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )
    return model.fit(disp=False)


def _score_candidates(
    split: ForecastSplit,
    candidates: list[CandidateSpec],
    cfg: ConfigDict,
) -> pd.DataFrame:
    runtime_cfg = cfg.get("runtime", {})
    enforce_stationarity = bool(runtime_cfg.get("enforce_stationarity", False))
    enforce_invertibility = bool(runtime_cfg.get("enforce_invertibility", False))

    rows: list[dict[str, Any]] = []

    for spec in candidates:
        try:
            fitted = _fit_one(
                split.train,
                order=spec.order,
                seasonal_order=spec.seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            )
            pred = fitted.get_forecast(steps=len(split.valid)).predicted_mean
            pred.name = split.valid.name
            score = _metrics(split.valid, pred)
            rows.append(
                {
                    "family": spec.family,
                    "order": str(spec.order),
                    "seasonal_order": str(spec.seasonal_order),
                    "aic": float(fitted.aic),
                    "bic": float(fitted.bic),
                    "valid_rmse": score["rmse"],
                    "valid_mae": score["mae"],
                    "valid_mape": score["mape"],
                    "valid_smape": score["smape"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "family": spec.family,
                    "order": str(spec.order),
                    "seasonal_order": str(spec.seasonal_order),
                    "aic": np.nan,
                    "bic": np.nan,
                    "valid_rmse": np.nan,
                    "valid_mae": np.nan,
                    "valid_mape": np.nan,
                    "valid_smape": np.nan,
                    "error": str(exc),
                }
            )

    scored = pd.DataFrame(rows)
    if scored.empty:
        raise ValueError("No candidate models were evaluated.")

    return scored.sort_values(["valid_rmse", "aic"], na_position="last").reset_index(drop=True)


def _parse_tuple(text: str) -> tuple[int, int, int] | tuple[int, int, int, int]:
    values = text.strip().strip("()").split(",")
    values = [v.strip() for v in values if v.strip()]
    return tuple(int(v) for v in values)  # type: ignore[return-value]


def _residual_diagnostics(residuals: pd.Series, lb_lags: int = 12) -> pd.DataFrame:
    clean_resid = residuals.dropna().astype(float)
    if clean_resid.empty:
        raise ValueError("Residual series is empty; cannot run diagnostics.")

    max_lb_lags = max(1, min(lb_lags, max(len(clean_resid) // 3, 1)))
    lb_res = acorr_ljungbox(clean_resid, lags=[max_lb_lags], return_df=True)
    lb_stat = float(lb_res["lb_stat"].iloc[0])
    lb_pvalue = float(lb_res["lb_pvalue"].iloc[0])

    jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(clean_resid)

    return pd.DataFrame(
        [
            {
                "test": "Ljung-Box",
                "statistic": lb_stat,
                "pvalue": lb_pvalue,
                "lags": int(max_lb_lags),
                "null_hypothesis": "No residual autocorrelation",
                "decision_5pct": "pass" if lb_pvalue > 0.05 else "fail",
            },
            {
                "test": "Jarque-Bera",
                "statistic": float(jb_stat),
                "pvalue": float(jb_pvalue),
                "lags": np.nan,
                "null_hypothesis": "Residuals are normally distributed",
                "decision_5pct": "pass" if float(jb_pvalue) > 0.05 else "fail",
            },
            {
                "test": "Residual moments",
                "statistic": np.nan,
                "pvalue": np.nan,
                "lags": np.nan,
                "null_hypothesis": "Summary only",
                "decision_5pct": f"skew={float(skewness):.4f}, kurtosis={float(kurtosis):.4f}",
            },
        ]
    )


def _plot_forecast(
    series: pd.Series,
    split: ForecastSplit,
    test_pred: pd.Series,
    future_df: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, color="#4c78a8", lw=1.8, label="Observed")
    ax.axvline(split.valid.index[0], color="#999999", lw=1, linestyle="--", label="Validation start")
    ax.axvline(split.test.index[0], color="#666666", lw=1, linestyle="--", label="Test start")

    ax.plot(test_pred.index, test_pred.values, color="#f58518", lw=1.8, label="Test forecast")
    ax.plot(
        future_df.index,
        future_df["forecast"].values,
        color="#54a24b",
        lw=2.0,
        label="Future forecast",
    )
    ax.fill_between(
        future_df.index,
        future_df["lower"],
        future_df["upper"],
        color="#54a24b",
        alpha=0.2,
        label="Prediction interval",
    )

    ax.set_title("NASDAQ Forecast: ARIMA/SARIMA")
    ax.set_xlabel("Date")
    ax.set_ylabel(series.name)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_forecast_from_config(
    cfg: ConfigDict,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    paths = resolve_paths(root, cfg)

    forecast_cfg = cfg.get("forecast", {})
    input_csv = Path(forecast_cfg.get("input_csv", "data/interim/nasdaq_deseasonalized_monthly.csv"))
    if not input_csv.is_absolute():
        input_csv = (root / input_csv).resolve()

    target_column = forecast_cfg.get("target_column", "NASDAQ_SA")
    horizon = int(forecast_cfg.get("horizon", 12))
    confidence_level = float(forecast_cfg.get("confidence_level", 0.9))
    output_tag = str(forecast_cfg.get("output_tag", "baseline"))

    split_cfg = forecast_cfg.get("evaluation", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.7))
    valid_ratio = float(split_cfg.get("valid_ratio", 0.15))
    min_train_size = int(split_cfg.get("min_train_size", 72))

    if not (0.0 < train_ratio < 1.0) or not (0.0 < valid_ratio < 1.0):
        raise ValueError("train_ratio and valid_ratio must be in (0, 1).")
    if train_ratio + valid_ratio >= 0.95:
        raise ValueError("train_ratio + valid_ratio must leave non-empty test set.")

    series = _read_target_series(input_csv, target_column)
    split = _split_series(series, train_ratio=train_ratio, valid_ratio=valid_ratio, min_train_size=min_train_size)

    candidates = _build_candidates(forecast_cfg.get("model_space", {}))
    scores = _score_candidates(split, candidates, cfg=forecast_cfg)

    best_row = scores.dropna(subset=["valid_rmse"]).iloc[0]
    best_order = _parse_tuple(str(best_row["order"]))
    best_seasonal = _parse_tuple(str(best_row["seasonal_order"]))

    runtime_cfg = forecast_cfg.get("runtime", {})
    fitted_train_valid = _fit_one(
        pd.concat([split.train, split.valid]),
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=bool(runtime_cfg.get("enforce_stationarity", False)),
        enforce_invertibility=bool(runtime_cfg.get("enforce_invertibility", False)),
    )

    test_pred = fitted_train_valid.get_forecast(steps=len(split.test)).predicted_mean
    test_metrics = _metrics(split.test, test_pred)

    fitted_full = _fit_one(
        series,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=bool(runtime_cfg.get("enforce_stationarity", False)),
        enforce_invertibility=bool(runtime_cfg.get("enforce_invertibility", False)),
    )
    alpha = max(1.0 - confidence_level, 1e-4)
    future_forecast = fitted_full.get_forecast(steps=horizon)
    future_mean = future_forecast.predicted_mean
    future_ci = future_forecast.conf_int(alpha=alpha)

    lower_col = future_ci.columns[0]
    upper_col = future_ci.columns[1]
    future_df = pd.DataFrame(
        {
            "forecast": future_mean,
            "lower": future_ci[lower_col],
            "upper": future_ci[upper_col],
        },
        index=future_mean.index,
    )

    backtest_df = pd.DataFrame(
        [
            {
                "split": "validation",
                "rmse": float(best_row["valid_rmse"]),
                "mae": float(best_row["valid_mae"]),
                "mape": float(best_row["valid_mape"]),
                "smape": float(best_row["valid_smape"]),
            },
            {
                "split": "test",
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"],
                "mape": test_metrics["mape"],
                "smape": test_metrics["smape"],
            },
        ]
    )

    diagnostics_cfg = forecast_cfg.get("diagnostics", {})
    lb_lags = int(diagnostics_cfg.get("ljung_box_lags", 12))
    diagnostics_df = _residual_diagnostics(pd.Series(fitted_full.resid), lb_lags=lb_lags)

    tables_dir = paths["tables_dir"] / "forecasts" / output_tag
    figures_dir = paths["figures_dir"] / "forecasts" / output_tag
    models_dir = paths["models_dir"] / "forecasts" / output_tag

    write_dataframe(scores, tables_dir / "candidate_scores.csv")
    write_dataframe(future_df, tables_dir / "forecast_points.csv")
    write_dataframe(backtest_df, tables_dir / "backtest_metrics.csv")
    write_dataframe(diagnostics_df, tables_dir / "model_diagnostics.csv")
    write_text(str(fitted_full.summary()), models_dir / "best_model_summary.txt")

    _plot_forecast(
        series=series,
        split=split,
        test_pred=test_pred,
        future_df=future_df,
        path=figures_dir / "forecast_plot.png",
    )

    return {
        "target": str(series.name),
        "n_obs": int(len(series)),
        "train_obs": int(len(split.train)),
        "valid_obs": int(len(split.valid)),
        "test_obs": int(len(split.test)),
        "best_family": str(best_row["family"]),
        "best_order": str(best_row["order"]),
        "best_seasonal_order": str(best_row["seasonal_order"]),
        "valid_rmse": float(best_row["valid_rmse"]),
        "test_rmse": float(test_metrics["rmse"]),
        "forecast_horizon": int(horizon),
        "forecast_table": str(tables_dir / "forecast_points.csv"),
        "scores_table": str(tables_dir / "candidate_scores.csv"),
        "backtest_table": str(tables_dir / "backtest_metrics.csv"),
        "diagnostics_table": str(tables_dir / "model_diagnostics.csv"),
        "forecast_figure": str(figures_dir / "forecast_plot.png"),
    }


def run_forecast(config_path: str | Path | None = None, project_root: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    summary = run_forecast_from_config(cfg, project_root=project_root)
    summary["config_path"] = str(config_path) if config_path else "default"
    return summary
