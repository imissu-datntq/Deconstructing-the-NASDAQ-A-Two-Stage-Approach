from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


@dataclass
class Stage2Result:
    model: Any
    selected_lag: int
    lag_selection: pd.DataFrame
    adf_table: pd.DataFrame
    structural_shocks: pd.DataFrame
    irf_table: pd.DataFrame
    fevd_table: pd.DataFrame
    irf_confidence_level: float | None


def _run_adf_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in data.columns:
        series = data[col].dropna()
        stat, pvalue, usedlag, nobs, crit, _ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "variable": col,
                "adf_stat": stat,
                "pvalue": pvalue,
                "used_lag": usedlag,
                "nobs": nobs,
                "critical_1pct": crit["1%"],
                "critical_5pct": crit["5%"],
                "critical_10pct": crit["10%"],
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def _select_lag_order(model: VAR, maxlags: int, ic: str, fallback_lag: int) -> tuple[int, pd.DataFrame]:
    selection = model.select_order(maxlags=maxlags)
    selected_orders = selection.selected_orders
    lag = selected_orders.get(ic)
    if lag is None:
        lag = fallback_lag
    lag = max(int(lag), 1)
    selection_table = pd.DataFrame([selected_orders])
    return lag, selection_table


def _recover_recursive_structural_shocks(model_fit: Any) -> pd.DataFrame:
    resid = model_fit.resid
    sigma_u = model_fit.sigma_u.to_numpy()
    cholesky = np.linalg.cholesky(sigma_u)
    eps = np.linalg.solve(cholesky, resid.to_numpy().T).T
    columns = [f"shock_{name}" for name in model_fit.names]
    return pd.DataFrame(eps, index=resid.index, columns=columns)


def _build_irf_table(
    model_fit: Any,
    response_variable: str,
    shock_variable: str,
    horizon: int,
    compute_bands: bool = True,
    confidence_level: float = 0.90,
    repl: int = 400,
    burn: int = 100,
    seed: int | None = None,
) -> pd.DataFrame:
    irf = model_fit.irf(horizon)
    names = model_fit.names
    response_idx = names.index(response_variable)
    shock_idx = names.index(shock_variable)
    response = irf.orth_irfs[:, response_idx, shock_idx]

    table = pd.DataFrame({"horizon": np.arange(response.size), "orth_irf": response})
    if compute_bands:
        conf = float(confidence_level)
        if not (0.5 < conf < 1.0):
            raise ValueError("irf_confidence_level must be between 0.5 and 1.0.")
        signif = 1.0 - conf
        lower, upper = _compute_irf_bands(
            model_fit=model_fit,
            horizon=horizon,
            orth=True,
            repl=int(repl),
            signif=signif,
            burn=int(burn),
            seed=seed,
        )
        table["lower"] = lower[:, response_idx, shock_idx]
        table["upper"] = upper[:, response_idx, shock_idx]
    return table


def _compute_irf_bands(
    model_fit: Any,
    horizon: int,
    orth: bool,
    repl: int,
    signif: float,
    burn: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if seed is None:
        return model_fit.irf_errband_mc(
            orth=orth,
            repl=repl,
            steps=horizon,
            signif=signif,
            seed=None,
            burn=burn,
            cum=False,
        )

    # Statsmodels seeds each replication internally when seed is passed, which can
    # collapse the interval. We draw a new seed per replication for stable, reproducible bands.
    rng = np.random.default_rng(seed)
    neqs = int(model_fit.neqs)
    sims = np.zeros((repl, horizon + 1, neqs, neqs))
    for i in range(repl):
        draw_seed = int(rng.integers(0, 2_147_483_647))
        sim = model_fit.irf_resim(
            orth=orth,
            repl=1,
            steps=horizon,
            seed=draw_seed,
            burn=burn,
            cum=False,
        )
        sims[i] = sim[0]

    sims = np.sort(sims, axis=0)
    low_idx = int(round(signif / 2 * repl) - 1)
    upp_idx = int(round((1 - signif / 2) * repl) - 1)
    low_idx = min(max(low_idx, 0), repl - 1)
    upp_idx = min(max(upp_idx, 0), repl - 1)
    return sims[low_idx], sims[upp_idx]


def _build_fevd_table(model_fit: Any, response_variable: str, horizon: int) -> pd.DataFrame:
    fevd = model_fit.fevd(horizon)
    decomp = np.asarray(fevd.decomp)
    names = list(model_fit.names)
    k = len(names)
    response_idx = names.index(response_variable)

    if decomp.shape[0] == k and decomp.shape[-1] == k:
        contrib = decomp[response_idx]
    elif decomp.shape[1] == k and decomp.shape[2] == k:
        contrib = decomp[:, response_idx, :]
    else:
        raise ValueError(f"Unexpected FEVD shape: {decomp.shape}")

    fevd_df = pd.DataFrame(contrib, columns=[f"shock_{name}" for name in names])
    fevd_df.insert(0, "horizon", np.arange(1, fevd_df.shape[0] + 1))
    return fevd_df


def run_stage2_recursive_svar(stage2_data: pd.DataFrame, stage2_cfg: dict[str, Any]) -> Stage2Result:
    prepared = stage2_data.copy()
    if isinstance(prepared.index, pd.DatetimeIndex):
        prepared.index = pd.PeriodIndex(prepared.index, freq="M")

    maxlags = int(stage2_cfg.get("maxlags", 18))
    ic = str(stage2_cfg.get("ic", "aic")).lower()
    fallback_lag = int(stage2_cfg.get("fallback_lag", 2))
    irf_horizon = int(stage2_cfg.get("irf_horizon", 24))
    fevd_horizon = int(stage2_cfg.get("fevd_horizon", 24))
    compute_irf_bands = bool(stage2_cfg.get("compute_irf_bands", True))
    irf_confidence_level = float(stage2_cfg.get("irf_confidence_level", 0.90))
    irf_errband_repl = int(stage2_cfg.get("irf_errband_repl", 400))
    irf_errband_burn = int(stage2_cfg.get("irf_errband_burn", 100))
    irf_errband_seed = stage2_cfg.get("irf_errband_seed")
    if irf_errband_seed is not None:
        irf_errband_seed = int(irf_errband_seed)
    response_var = str(stage2_cfg.get("equity_response_variable", "NASDAQ_SA"))
    shock_var = str(stage2_cfg.get("policy_shock_variable", "FEDFUNDS"))

    model = VAR(prepared)
    selected_lag, lag_selection = _select_lag_order(model, maxlags=maxlags, ic=ic, fallback_lag=fallback_lag)
    model_fit = model.fit(selected_lag)

    adf_table = _run_adf_tests(prepared)
    structural_shocks = _recover_recursive_structural_shocks(model_fit)
    irf_table = _build_irf_table(
        model_fit,
        response_variable=response_var,
        shock_variable=shock_var,
        horizon=irf_horizon,
        compute_bands=compute_irf_bands,
        confidence_level=irf_confidence_level,
        repl=irf_errband_repl,
        burn=irf_errband_burn,
        seed=irf_errband_seed,
    )
    fevd_table = _build_fevd_table(
        model_fit,
        response_variable=response_var,
        horizon=fevd_horizon,
    )

    return Stage2Result(
        model=model_fit,
        selected_lag=selected_lag,
        lag_selection=lag_selection,
        adf_table=adf_table,
        structural_shocks=structural_shocks,
        irf_table=irf_table,
        fevd_table=fevd_table,
        irf_confidence_level=irf_confidence_level if compute_irf_bands else None,
    )
