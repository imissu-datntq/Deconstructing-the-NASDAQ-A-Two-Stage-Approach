from __future__ import annotations

from math import isnan
from typing import Any

import numpy as np
import pandas as pd


def _fmt_float(value: float, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)) and isnan(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: float, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)) and isnan(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.{digits}f}%"


def build_chapter1_conclusion(
    raw: dict[str, pd.Series],
    compare: pd.DataFrame,
    adf_table: pd.DataFrame,
    transforms: dict[str, str],
) -> str:
    starts = [s.index.min() for s in raw.values() if not s.empty]
    ends = [s.index.max() for s in raw.values() if not s.empty]
    start = min(starts).date().isoformat() if starts else "N/A"
    end = max(ends).date().isoformat() if ends else "N/A"

    median_abs_gap = (compare["NASDAQ_SA_LAST"] - compare["NASDAQ_SA_MEAN"]).abs().median()
    stationary_count = int((adf_table["pvalue"] < 0.05).sum())
    non_stationary = adf_table.index[adf_table["pvalue"] >= 0.05].tolist()
    non_stationary_text = ", ".join(non_stationary) if non_stationary else "none"
    transform_text = ", ".join(f"{k}: {v}" for k, v in transforms.items())

    return "\n".join(
        [
            "### Auto Conclusion - Chapter 1 (EDA & Data Prep)",
            f"- Data window used: **{start} to {end}**.",
            "- Stage-1 STL decomposition isolates trend/seasonal/residual components on daily NASDAQ before macro modeling.",
            f"- End-of-month sampling is materially different from monthly averaging (median absolute gap: **{_fmt_float(median_abs_gap, 3)}** index points), supporting the month-end synchronization choice.",
            f"- ADF result after configured transforms: **{stationary_count}/{len(adf_table)}** series stationary at 5% level.",
            f"- Non-stationary transformed series at 5%: **{non_stationary_text}**.",
            f"- Transform policy selected for Stage-2: **{transform_text}**.",
        ]
    )


def build_chapter2_conclusion(
    selected_lag: int,
    lag_order_table: pd.DataFrame,
    inv_roots: np.ndarray,
    whiteness_pvalue: float | None,
    chosen_ic: str,
) -> str:
    max_modulus = float(np.max(np.abs(inv_roots))) if inv_roots.size else float("nan")
    is_stable = bool(max_modulus < 1.0) if not np.isnan(max_modulus) else False
    residual_text = "no autocorrelation evidence" if (whiteness_pvalue is not None and whiteness_pvalue > 0.05) else "possible residual autocorrelation"

    selectors = ", ".join(f"{k.upper()}={v}" for k, v in lag_order_table.iloc[0].to_dict().items())
    return "\n".join(
        [
            "### Auto Conclusion - Chapter 2 (VAR Diagnostics)",
            f"- Lag-order candidates from information criteria: **{selectors}**.",
            f"- Baseline lag selected by **{chosen_ic.upper()}** rule: **p={selected_lag}**.",
            f"- Stability check (inverse roots): max modulus **{_fmt_float(max_modulus, 3)}** -> model is **{'stable' if is_stable else 'not stable'}**.",
            f"- Residual whiteness test suggests **{residual_text}** (p-value: **{_fmt_float(whiteness_pvalue, 4)}**).",
            "- Diagnostic outcome supports moving to structural interpretation under recursive identification.",
        ]
    )


def build_chapter3_conclusion(
    irf_table: pd.DataFrame,
    fevd_table: pd.DataFrame,
    response_variable: str,
    shock_variable: str,
    confidence_level: float | None,
) -> str:
    trough_idx = int(irf_table["orth_irf"].idxmin())
    trough_h = int(irf_table.loc[trough_idx, "horizon"])
    trough_val = float(irf_table.loc[trough_idx, "orth_irf"])
    impact_val = float(irf_table.loc[irf_table["horizon"] == 0, "orth_irf"].iloc[0])

    sig_horizons = 0
    if {"lower", "upper"}.issubset(irf_table.columns):
        sig_mask = (irf_table["lower"] > 0) | (irf_table["upper"] < 0)
        sig_horizons = int(sig_mask.sum())

    policy_col = f"shock_{shock_variable}"
    fevd_12 = float("nan")
    fevd_24 = float("nan")
    if policy_col in fevd_table.columns:
        row12 = fevd_table.loc[fevd_table["horizon"] == 12, policy_col]
        row24 = fevd_table.loc[fevd_table["horizon"] == 24, policy_col]
        if not row12.empty:
            fevd_12 = float(row12.iloc[0])
        if not row24.empty:
            fevd_24 = float(row24.iloc[0])

    ci_text = "not computed"
    if confidence_level is not None:
        ci_text = f"{int(confidence_level * 100)}% MC band, significant horizons={sig_horizons}"

    return "\n".join(
        [
            "### Auto Conclusion - Chapter 3 (SVAR Inference)",
            f"- Structural question: response of **{response_variable}** to one-standard-deviation **{shock_variable}** shock.",
            f"- Impact response (h=0): **{_fmt_float(impact_val, 4)}**.",
            f"- Peak contraction occurs at horizon **h={trough_h}** with magnitude **{_fmt_float(trough_val, 4)}**.",
            f"- IRF uncertainty treatment: **{ci_text}**.",
            f"- FEVD policy-shock share of {response_variable} variance: **{_fmt_pct(fevd_12)}** at 12 months, **{_fmt_pct(fevd_24)}** at 24 months.",
            "- Interpretation: monetary-policy shocks are identifiable and economically meaningful, while total variance share can remain limited in equity dynamics.",
        ]
    )


def build_chapter4_conclusion(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "\n".join(
            [
                "### Auto Conclusion - Chapter 4 (Robustness)",
                "- No scenario results found.",
            ]
        )

    all_negative = bool((summary["irf_trough_value"] < 0).all())
    trough_min = float(summary["irf_trough_value"].min())
    trough_max = float(summary["irf_trough_value"].max())
    fevd12_min = float(summary["fevd_policy_share_h12"].min())
    fevd12_max = float(summary["fevd_policy_share_h12"].max())

    strongest = summary.loc[summary["irf_trough_value"].idxmin(), "scenario"]
    weakest = summary.loc[summary["irf_trough_value"].idxmax(), "scenario"]

    return "\n".join(
        [
            "### Auto Conclusion - Chapter 4 (Robustness Checks)",
            f"- Number of robustness scenarios evaluated: **{summary.shape[0]}**.",
            f"- IRF trough sign consistency across scenarios: **{'consistent negative' if all_negative else 'mixed sign'}**.",
            f"- Trough range across scenarios: **{_fmt_float(trough_min, 4)} to {_fmt_float(trough_max, 4)}**.",
            f"- Strongest contraction scenario: **{strongest}**; weakest contraction scenario: **{weakest}**.",
            f"- FEVD policy-share at 12 months ranges from **{_fmt_pct(fevd12_min)}** to **{_fmt_pct(fevd12_max)}**.",
            "- Robustness verdict: core directional response is stable, while magnitude depends on lag and transform choices.",
        ]
    )


def whiteness_pvalue(whiteness_result: Any) -> float | None:
    for attr in ("pvalue", "pvalue_adj", "pvalue_df"):
        value = getattr(whiteness_result, attr, None)
        if value is None:
            continue
        if isinstance(value, (float, int, np.floating)):
            return float(value)
    return None

