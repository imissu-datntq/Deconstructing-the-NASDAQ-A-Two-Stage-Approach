from __future__ import annotations

import argparse
from pathlib import Path

from .sensitivity import run_sensitivity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for NASDAQ two-stage STL + recursive SVAR pipeline."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/sensitivity.yaml"),
        help="Path to sensitivity scenario YAML plan.",
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
    result = run_sensitivity(plan_path=args.plan, project_root=args.project_root)
    print("Sensitivity run finished.")
    print(f"Scenarios: {result['scenarios_run']}")
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"IRF comparison figure: {result['irf_comparison_figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

