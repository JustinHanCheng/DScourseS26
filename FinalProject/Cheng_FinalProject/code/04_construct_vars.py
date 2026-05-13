"""Construct Basu (1997) variables: scaled earnings (EP), bad-news dummy (D)
and the bad-news interaction (DR). Winsorize EP and R at 1% / 99%.

Run:  python code/04_construct_vars.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import DATA_CLEAN, SAMPLE_END, SAMPLE_START


def winz(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def main() -> None:
    df = pd.read_parquet(DATA_CLEAN / "funda_with_R.parquet")

    df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)

    df["prcc_lag"] = df.groupby("gvkey")["prcc_f"].shift(1)
    df["ajex_lag"] = df.groupby("gvkey")["ajex"].shift(1)
    df["fyear_lag"] = df.groupby("gvkey")["fyear"].shift(1)

    df = df[df["fyear_lag"] == df["fyear"] - 1]
    df = df.dropna(subset=["prcc_lag", "ajex_lag"])
    df = df[df["ajex_lag"] != 0]

    # EPS scaled by split-adjusted beginning-of-period price (Basu 1997)
    df["EP"] = df["epspx"] / (df["prcc_lag"] * df["ajex"] / df["ajex_lag"])

    df = df[df["fyear"].between(SAMPLE_START, SAMPLE_END)]

    df["EP"] = winz(df["EP"])
    df["R"] = winz(df["R"])

    df["D"] = (df["R"] < 0).astype(int)
    df["DR"] = df["D"] * df["R"]

    keep = [
        "gvkey", "permno", "fyear", "datadate", "sich",
        "EP", "R", "D", "DR", "epspx", "prcc_f", "prcc_lag", "at", "ceq",
    ]
    final = df[keep].dropna(subset=["EP", "R"]).reset_index(drop=True)

    final.to_parquet(DATA_CLEAN / "panel.parquet", index=False)
    print(
        f"Final panel: {len(final):,} firm-years across "
        f"{final['fyear'].nunique()} fiscal years "
        f"({final['fyear'].min()}–{final['fyear'].max()})"
    )


if __name__ == "__main__":
    main()
