"""Compound CRSP monthly returns into 12-month fiscal-year returns and join
them onto the firm-year panel.

Run:  python code/03_compute_returns.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import DATA_CLEAN, DATA_RAW


def main() -> None:
    funda = pd.read_parquet(DATA_CLEAN / "funda_linked.parquet")
    msf = pd.read_parquet(DATA_RAW / "msf.parquet")

    msf["ret"] = pd.to_numeric(msf["ret"], errors="coerce")
    msf = msf.dropna(subset=["ret"]).copy()
    msf["date"] = pd.to_datetime(msf["date"])
    msf["ym"] = msf["date"].dt.to_period("M")
    msf = msf[["permno", "ym", "ret"]]

    funda["fy_end"] = pd.to_datetime(funda["datadate"])
    funda["fy_end_period"] = funda["fy_end"].dt.to_period("M")
    funda["fy_start_period"] = (
        (funda["fy_end"] - pd.DateOffset(months=11)).dt.to_period("M")
    )

    # Build long table: one row per firm-year-month inside the fiscal window
    rows = []
    for idx, row in funda.iterrows():
        months = pd.period_range(row["fy_start_period"], row["fy_end_period"], freq="M")
        for ym in months:
            rows.append((idx, row["permno"], ym))
    long = pd.DataFrame(rows, columns=["idx", "permno", "ym"])

    long = long.merge(msf, on=["permno", "ym"], how="left")

    grouped = long.groupby("idx")["ret"]

    def fy_return(s: pd.Series) -> float:
        clean = s.dropna()
        if len(clean) < 12:
            return np.nan
        return float(np.prod(1.0 + clean.to_numpy()) - 1.0)

    funda["R"] = grouped.apply(fy_return)
    funda = funda.dropna(subset=["R"]).reset_index(drop=True)

    funda.to_parquet(DATA_CLEAN / "funda_with_R.parquet", index=False)
    print(f"With 12-month returns: {len(funda):,} firm-years")


if __name__ == "__main__":
    main()
