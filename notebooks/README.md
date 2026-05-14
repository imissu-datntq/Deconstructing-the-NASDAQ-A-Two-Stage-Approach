# Notebook Guide

This folder provides a narrative, grading-friendly workflow on top of the package pipeline.

## Suggested order

1. `01_EDA_and_Data_Prep.ipynb`
2. `02_VAR_Modeling_and_Diagnostics.ipynb`
3. `03_SVAR_and_Inference.ipynb`
4. `04_Robustness_Checks.ipynb`
5. `05_Forecast_Data_and_Splits.ipynb`
6. `06_ARIMA_SARIMA_Tuning.ipynb`
7. `07_Backtest_and_Final_Forecast.ipynb`

## Notes

- Notebooks import functions from `src/nasdaq_svar` instead of duplicating implementation logic.
- If `data/raw` is missing, notebooks can trigger `run_pipeline()` to pull data and generate baseline artifacts.
- Sensitivity scenarios are configured in `configs/sensitivity.yaml`.
- Forecasting scenarios are configured in `configs/forecast.yaml`.
- Each notebook now ends with a **Slide-Ready Conclusion** section generated automatically from computed outputs.
- Conclusion text templates are centralized in `src/nasdaq_svar/presentation.py` to keep chapter messaging consistent.
