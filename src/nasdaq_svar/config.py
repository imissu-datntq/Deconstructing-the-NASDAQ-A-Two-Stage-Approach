from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ConfigDict = dict[str, Any]


DEFAULT_CONFIG: ConfigDict = {
    "sample": {
        "start": "1990-01-01",
        "end": None,
    },
    "series": {
        "INDPRO": "INDPRO",
        "CPIAUCNS": "CPIAUCNS",
        "FEDFUNDS": "FEDFUNDS",
        "NASDAQCOM": "NASDAQCOM",
    },
    "stage1": {
        "stl_period": 21,
        "seasonal_window": 13,
        "trend_window": None,
        "robust": True,
        "inner_iter": None,
        "outer_iter": None,
        "log_input": True,
        "fill_method": "ffill",
    },
    "stage2": {
        "ordering": ["INDPRO", "CPIAUCNS", "FEDFUNDS", "NASDAQ_SA"],
        "transforms": {
            "INDPRO": "log_diff",
            "CPIAUCNS": "log_diff",
            "FEDFUNDS": "level",
            "NASDAQ_SA": "log_diff",
        },
        "maxlags": 18,
        "ic": "aic",
        "fallback_lag": 2,
        "irf_horizon": 24,
        "fevd_horizon": 24,
        "irf_confidence_level": 0.90,
        "irf_errband_repl": 400,
        "irf_errband_burn": 100,
        "irf_errband_seed": 2026,
        "compute_irf_bands": True,
        "policy_shock_variable": "FEDFUNDS",
        "equity_response_variable": "NASDAQ_SA",
    },
    "paths": {
        "raw_dir": "data/raw",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
        "figures_dir": "outputs/figures",
        "tables_dir": "outputs/tables",
        "models_dir": "outputs/models",
    },
}


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> ConfigDict:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    if cfg["sample"]["end"] is None:
        cfg["sample"]["end"] = date.today().isoformat()
    return cfg


def merge_config(base: ConfigDict, override: ConfigDict | None = None) -> ConfigDict:
    if not override:
        return deepcopy(base)
    return _deep_merge(base, override)


def resolve_paths(project_root: Path, config: ConfigDict) -> dict[str, Path]:
    path_cfg = config["paths"]
    resolved = {name: (project_root / rel).resolve() for name, rel in path_cfg.items()}
    for directory in resolved.values():
        directory.mkdir(parents=True, exist_ok=True)
    return resolved
