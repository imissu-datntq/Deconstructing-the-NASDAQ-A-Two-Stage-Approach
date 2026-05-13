from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NASDAQ two-stage STL + recursive SVAR research pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve output folders.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_pipeline(config_path=args.config, project_root=args.project_root)
    print("Pipeline finished.")
    print(f"Sample window: {summary['sample_start']} -> {summary['sample_end']}")
    print(f"Stage-2 observations: {summary['n_obs_stage2']}")
    print(f"Selected VAR lag: {summary['selected_lag']}")
    print(f"IRF trough: {summary['irf_trough_value']:.6f} at h={summary['irf_trough_horizon']}")
    print(f"FEVD policy share at 12m: {summary['fevd_policy_share_h12']:.4f}")
    print(f"Outputs written under: {summary['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
