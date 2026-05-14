from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from nasdaq_svar.forecast import run_forecast


st.set_page_config(page_title="NASDAQ Forecast Dashboard", layout="wide")

st.title("NASDAQ ARIMA/SARIMA Forecast Dashboard")
st.caption("Run forecast pipeline, inspect backtest and model diagnostics, and review prediction intervals.")

project_root = Path(__file__).resolve().parents[1]
default_cfg = project_root / "configs" / "forecast.yaml"

with st.sidebar:
    st.header("Run Settings")
    cfg_path_text = st.text_input("Forecast config path", value=str(default_cfg))
    run_button = st.button("Run Forecast", type="primary")

summary = None
error_msg = None

if run_button:
    cfg_path = Path(cfg_path_text)
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()

    try:
        with st.spinner("Running forecast pipeline..."):
            summary = run_forecast(config_path=cfg_path, project_root=project_root)
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)

if error_msg:
    st.error(error_msg)

if summary:
    st.success("Forecast pipeline completed.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best family", summary["best_family"])
    col2.metric("Best order", summary["best_order"])
    col3.metric("Validation RMSE", f"{summary['valid_rmse']:.6f}")
    col4.metric("Test RMSE", f"{summary['test_rmse']:.6f}")

    st.subheader("Forecast Summary")
    st.json(
        {
            "target": summary["target"],
            "n_obs": summary["n_obs"],
            "train_obs": summary["train_obs"],
            "valid_obs": summary["valid_obs"],
            "test_obs": summary["test_obs"],
            "forecast_horizon": summary["forecast_horizon"],
            "forecast_table": summary["forecast_table"],
            "diagnostics_table": summary["diagnostics_table"],
        }
    )

    st.subheader("Backtest Metrics")
    backtest_df = pd.read_csv(summary["backtest_table"])
    st.dataframe(backtest_df, use_container_width=True)

    st.subheader("Model Diagnostics")
    diagnostics_df = pd.read_csv(summary["diagnostics_table"])
    st.dataframe(diagnostics_df, use_container_width=True)

    st.subheader("Top Candidate Models")
    scores_df = pd.read_csv(summary["scores_table"]).head(15)
    st.dataframe(scores_df, use_container_width=True)

    st.subheader("Forecast Points")
    forecast_df = pd.read_csv(summary["forecast_table"])
    st.dataframe(forecast_df, use_container_width=True)

    st.subheader("Forecast Figure")
    st.image(summary["forecast_figure"], use_container_width=True)
else:
    st.info("Click 'Run Forecast' in the sidebar to start.")
