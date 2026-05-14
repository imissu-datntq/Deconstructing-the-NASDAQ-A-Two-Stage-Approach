from __future__ import annotations

import argparse
from pathlib import Path

from .forecast import run_forecast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ARIMA/SARIMA forecasting pipeline for monthly NASDAQ series."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecast.yaml"),
        help="Path to YAML forecast config file.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve input/output folders.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_forecast(config_path=args.config, project_root=args.project_root)
    print("Forecast pipeline finished.")
    print(f"Target: {summary['target']}")
    print(f"Observations: {summary['n_obs']} (train={summary['train_obs']}, valid={summary['valid_obs']}, test={summary['test_obs']})")
    print(f"Best model: {summary['best_family']} order={summary['best_order']} seasonal={summary['best_seasonal_order']}")
    print(f"Validation RMSE: {summary['valid_rmse']:.6f}")
    print(f"Test RMSE: {summary['test_rmse']:.6f}")
    print(f"Forecast table: {summary['forecast_table']}")
    print(f"Forecast figure: {summary['forecast_figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
