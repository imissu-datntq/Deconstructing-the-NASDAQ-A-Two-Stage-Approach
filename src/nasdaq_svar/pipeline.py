from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConfigDict, load_config, resolve_paths
from .data import download_fred_series_map
from .reporting import (
    plot_fevd,
    plot_irf,
    plot_stage1_components,
    write_dataframe,
    write_text,
)
from .stage1_stl import run_stage1_stl
from .stage2_svar import run_stage2_recursive_svar
from .transforms import build_stage2_dataset, to_month_end


def _build_monthly_input(raw_series: dict[str, pd.Series], monthly_nasdaq_sa: pd.Series) -> pd.DataFrame:
    macro_df = pd.DataFrame(
        {
            "INDPRO": to_month_end(raw_series["INDPRO"]),
            "CPIAUCNS": to_month_end(raw_series["CPIAUCNS"]),
            "FEDFUNDS": to_month_end(raw_series["FEDFUNDS"]),
        }
    )
    nasdaq_df = monthly_nasdaq_sa.to_frame(name="NASDAQ_SA")
    monthly = macro_df.join(nasdaq_df, how="inner").sort_index()
    return monthly


def _with_run_label(paths: dict[str, Path], run_label: str | None) -> dict[str, Path]:
    if not run_label:
        return paths
    labeled = dict(paths)
    for key in ("interim_dir", "processed_dir", "figures_dir", "tables_dir", "models_dir"):
        labeled[key] = (paths[key] / run_label).resolve()
        labeled[key].mkdir(parents=True, exist_ok=True)
    return labeled


def _summarize_stage2_outputs(irf_table: pd.DataFrame, fevd_table: pd.DataFrame) -> dict[str, float | int]:
    trough_idx = int(irf_table["orth_irf"].idxmin())
    trough_horizon = int(irf_table.loc[trough_idx, "horizon"])
    trough_value = float(irf_table.loc[trough_idx, "orth_irf"])
    impact_value = float(irf_table.loc[irf_table["horizon"] == 0, "orth_irf"].iloc[0])

    policy_cols = [c for c in fevd_table.columns if c.startswith("shock_FEDFUNDS")]
    policy_col = policy_cols[0] if policy_cols else None
    fevd_12 = float("nan")
    fevd_24 = float("nan")
    if policy_col is not None:
        row12 = fevd_table.loc[fevd_table["horizon"] == 12, policy_col]
        row24 = fevd_table.loc[fevd_table["horizon"] == 24, policy_col]
        if not row12.empty:
            fevd_12 = float(row12.iloc[0])
        if not row24.empty:
            fevd_24 = float(row24.iloc[0])

    return {
        "irf_impact_h0": impact_value,
        "irf_trough_value": trough_value,
        "irf_trough_horizon": trough_horizon,
        "fevd_policy_share_h12": fevd_12,
        "fevd_policy_share_h24": fevd_24,
    }


def run_pipeline_from_config(
    cfg: ConfigDict,
    project_root: str | Path | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    paths = resolve_paths(root, cfg)
    out_paths = _with_run_label(paths, run_label)

    sample = cfg["sample"]
    raw_series = download_fred_series_map(
        cfg["series"],
        start=str(sample["start"]),
        end=str(sample["end"]),
        raw_dir=paths["raw_dir"],
    )

    stage1 = run_stage1_stl(raw_series["NASDAQCOM"], cfg["stage1"])
    monthly_input = _build_monthly_input(raw_series, stage1.monthly_deseasonalized)
    stage2_data = build_stage2_dataset(
        monthly_input,
        transforms=cfg["stage2"]["transforms"],
        ordering=cfg["stage2"]["ordering"],
    )
    stage2 = run_stage2_recursive_svar(stage2_data, cfg["stage2"])

    write_dataframe(stage1.daily_components, out_paths["interim_dir"] / "nasdaq_stl_daily_components.csv")
    write_dataframe(
        stage1.monthly_deseasonalized.to_frame(),
        out_paths["interim_dir"] / "nasdaq_deseasonalized_monthly.csv",
    )
    write_dataframe(monthly_input, out_paths["processed_dir"] / "stage2_monthly_aligned_levels.csv")
    write_dataframe(stage2_data, out_paths["processed_dir"] / "stage2_transformed_dataset.csv")

    write_dataframe(stage2.adf_table, out_paths["tables_dir"] / "adf_stationarity_tests.csv")
    write_dataframe(stage2.lag_selection, out_paths["tables_dir"] / "var_lag_selection.csv")
    write_dataframe(stage2.irf_table, out_paths["tables_dir"] / "irf_nasdaq_to_fedfunds_shock.csv")
    write_dataframe(stage2.fevd_table, out_paths["tables_dir"] / "fevd_nasdaq.csv")
    write_dataframe(stage2.structural_shocks, out_paths["models_dir"] / "structural_shocks.csv")
    write_text(str(stage2.model.summary()), out_paths["models_dir"] / "var_summary.txt")

    plot_stage1_components(stage1.daily_components, out_paths["figures_dir"] / "stage1_stl_components.png")
    plot_irf(
        stage2.irf_table,
        out_paths["figures_dir"] / "irf_nasdaq_response.png",
        title="Orthogonalized IRF: NASDAQ_SA response to FEDFUNDS shock",
        confidence_level=stage2.irf_confidence_level,
    )
    plot_fevd(
        stage2.fevd_table,
        out_paths["figures_dir"] / "fevd_nasdaq.png",
        title="FEVD of NASDAQ_SA",
    )

    metrics = _summarize_stage2_outputs(stage2.irf_table, stage2.fevd_table)
    return {
        "run_label": run_label or "baseline",
        "sample_start": sample["start"],
        "sample_end": sample["end"],
        "n_obs_stage2": int(stage2_data.shape[0]),
        "selected_lag": int(stage2.selected_lag),
        "output_root": str(root),
        "interim_dir": str(out_paths["interim_dir"]),
        "processed_dir": str(out_paths["processed_dir"]),
        "tables_dir": str(out_paths["tables_dir"]),
        "figures_dir": str(out_paths["figures_dir"]),
        "models_dir": str(out_paths["models_dir"]),
        **metrics,
    }


def run_pipeline(config_path: str | Path | None = None, project_root: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    summary = run_pipeline_from_config(cfg, project_root=project_root, run_label=None)
    summary["config_path"] = str(config_path) if config_path else "default"
    return summary
