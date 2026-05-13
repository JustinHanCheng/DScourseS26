"""Merge Compustat funda with CRSP via the CCM linking table and apply
sample filters (US common stock, NYSE/AMEX/NASDAQ, December fiscal year-end,
non-financial, price > $1, no missing EPS / price).

Run:  python code/02_clean_merge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import DATA_CLEAN, DATA_RAW


def main() -> None:
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)

    funda = pd.read_parquet(DATA_RAW / "funda.parquet")
    ccm = pd.read_parquet(DATA_RAW / "ccm.parquet")
    stocknames = pd.read_parquet(DATA_RAW / "stocknames.parquet")

    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

    m = funda.merge(ccm, on="gvkey", how="inner")
    m = m[(m["datadate"] >= m["linkdt"]) & (m["datadate"] <= m["linkenddt"])]

    m = m.merge(stocknames, on="permno", how="inner")
    m = m[(m["datadate"] >= m["namedt"]) & (m["datadate"] <= m["nameenddt"])]

    # US common stock on NYSE / AMEX / NASDAQ
    m = m[m["shrcd"].isin([10, 11])]
    m = m[m["exchcd"].isin([1, 2, 3])]

    # Fiscal year-end December (Basu 1997 sample restriction; aligns return window)
    m = m[m["fyr"] == 12]

    # Drop financials (SIC 6000-6999); keep utilities for now
    sich = pd.to_numeric(m["sich"], errors="coerce")
    m = m[~sich.between(6000, 6999)]

    m = m.dropna(subset=["epspx", "prcc_f", "ajex"])
    m = m[m["prcc_f"] > 1.0]

    # One row per gvkey-fyear
    m = (
        m.sort_values(["gvkey", "fyear", "datadate"])
        .drop_duplicates(["gvkey", "fyear"], keep="last")
        .reset_index(drop=True)
    )

    m.to_parquet(DATA_CLEAN / "funda_linked.parquet", index=False)
    print(f"Merged sample: {len(m):,} firm-years; {m['gvkey'].nunique():,} unique firms")


if __name__ == "__main__":
    main()
