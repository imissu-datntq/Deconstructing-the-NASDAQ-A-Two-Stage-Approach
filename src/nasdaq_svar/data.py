from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id, "cosd": start, "coed": end},
        timeout=30,
    )
    response.raise_for_status()

    csv_df = pd.read_csv(StringIO(response.text))
    date_col = "DATE" if "DATE" in csv_df.columns else "observation_date"
    if date_col not in csv_df.columns or series_id not in csv_df.columns:
        raise ValueError(f"Unexpected CSV format while fetching {series_id}")

    series = pd.Series(
        pd.to_numeric(csv_df[series_id], errors="coerce").to_numpy(),
        index=pd.to_datetime(csv_df[date_col]),
        name=series_id,
    )
    series = series.sort_index().dropna().astype(float)
    return series


def download_fred_series_map(
    series_map: dict[str, str],
    start: str,
    end: str,
    raw_dir: Path,
) -> dict[str, pd.Series]:
    downloaded: dict[str, pd.Series] = {}
    for alias, fred_id in series_map.items():
        series = fetch_fred_series(fred_id, start=start, end=end)
        downloaded[alias] = series
        output_path = raw_dir / f"{alias}.csv"
        series.to_frame(name=fred_id).to_csv(output_path, index=True)
    return downloaded
