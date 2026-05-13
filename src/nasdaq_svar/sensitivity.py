from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .config import load_config, merge_config
from .pipeline import run_pipeline_from_config
from .reporting import write_dataframe


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "scenario"


def _plot_irf_comparison(irf_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for scenario, subset in irf_df.groupby("scenario"):
        ax.plot(subset["horizon"], subset["orth_irf"], lw=1.8, label=scenario)
    ax.axhline(0.0, color="black", lw=1.0, linestyle="--")
    ax.set_title("IRF Sensitivity: NASDAQ_SA response to FEDFUNDS shock")
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Orthogonalized response")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_sensitivity(plan_path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    plan_file = Path(plan_path)
    if not plan_file.is_absolute():
        plan_file = (root / plan_file).resolve()

    with plan_file.open("r", encoding="utf-8") as f:
        plan = yaml.safe_load(f) or {}

    base_config_path = Path(plan.get("base_config", "configs/default.yaml"))
    if not base_config_path.is_absolute():
        candidate = (plan_file.parent / base_config_path).resolve()
        if candidate.exists():
            base_config_path = candidate
        else:
            base_config_path = (root / base_config_path).resolve()
    base_cfg = load_config(base_config_path)

    scenario_output_tag = str(plan.get("scenario_output_tag", "sensitivity"))
    scenarios = plan.get("scenarios", [])
    if not scenarios:
        raise ValueError("No scenarios found in sensitivity config.")

    summary_rows: list[dict[str, Any]] = []
    irf_frames: list[pd.DataFrame] = []

    for scenario in scenarios:
        scenario_name = str(scenario["name"])
        overrides = scenario.get("overrides", {})
        scenario_cfg = merge_config(base_cfg, overrides)
        scenario_label = f"{scenario_output_tag}/{_slugify(scenario_name)}"

        result = run_pipeline_from_config(
            scenario_cfg,
            project_root=root,
            run_label=scenario_label,
        )
        result["scenario"] = scenario_name
        summary_rows.append(result)

        irf_path = Path(result["tables_dir"]) / "irf_nasdaq_to_fedfunds_shock.csv"
        irf_df = pd.read_csv(irf_path)
        irf_df["scenario"] = scenario_name
        irf_frames.append(irf_df[["horizon", "orth_irf", "scenario"]])

    summary_df = pd.DataFrame(summary_rows)
    summary_cols = [
        "scenario",
        "sample_start",
        "sample_end",
        "n_obs_stage2",
        "selected_lag",
        "irf_impact_h0",
        "irf_trough_value",
        "irf_trough_horizon",
        "fevd_policy_share_h12",
        "fevd_policy_share_h24",
        "tables_dir",
    ]
    summary_df = summary_df[summary_cols]
    summary_df = summary_df.sort_values("scenario").reset_index(drop=True)

    irf_all = pd.concat(irf_frames, axis=0, ignore_index=True)

    summary_dir = root / "outputs" / "tables" / scenario_output_tag
    figures_dir = root / "outputs" / "figures" / scenario_output_tag
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_dataframe(summary_df, summary_dir / "sensitivity_summary.csv")
    write_dataframe(irf_all, summary_dir / "sensitivity_irf_paths.csv")
    _plot_irf_comparison(irf_all, figures_dir / "sensitivity_irf_comparison.png")

    return {
        "scenarios_run": int(summary_df.shape[0]),
        "summary_csv": str(summary_dir / "sensitivity_summary.csv"),
        "irf_comparison_figure": str(figures_dir / "sensitivity_irf_comparison.png"),
    }
