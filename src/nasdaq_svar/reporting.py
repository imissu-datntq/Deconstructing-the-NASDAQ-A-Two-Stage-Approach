from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_stage1_components(components: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    components["observed_level"].plot(ax=axes[0], color="#1f77b4", lw=1.0, title="NASDAQ Observed (Daily)")
    components["trend"].plot(ax=axes[1], color="#2ca02c", lw=1.0, title="STL Trend (Transformed Scale)")
    components["seasonal"].plot(ax=axes[2], color="#ff7f0e", lw=1.0, title="STL Seasonal (Transformed Scale)")
    components["resid"].plot(ax=axes[3], color="#d62728", lw=0.8, title="STL Residual (Transformed Scale)")
    for ax in axes:
        ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_irf(
    irf_table: pd.DataFrame,
    path: Path,
    title: str,
    confidence_level: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(irf_table["horizon"], irf_table["orth_irf"], color="#1f77b4", lw=2)
    if {"lower", "upper"}.issubset(irf_table.columns):
        label = "MC band"
        if confidence_level is not None:
            label = f"{int(confidence_level * 100)}% MC band"
        ax.fill_between(
            irf_table["horizon"],
            irf_table["lower"],
            irf_table["upper"],
            color="#1f77b4",
            alpha=0.2,
            label=label,
        )
    ax.axhline(0.0, color="black", lw=1, linestyle="--")
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Response")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle="--")
    if {"lower", "upper"}.issubset(irf_table.columns):
        ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fevd(fevd_table: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = fevd_table["horizon"]
    shock_cols = [c for c in fevd_table.columns if c.startswith("shock_")]
    for col in shock_cols:
        ax.plot(horizons, fevd_table[col], lw=1.8, label=col)
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Variance Share")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
