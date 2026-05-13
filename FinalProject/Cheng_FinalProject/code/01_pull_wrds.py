"""Pull Compustat funda, CRSP monthly stock file, CCM linking table, and
CRSP stocknames from WRDS into local parquet files.

Run:  python code/01_pull_wrds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wrds  # type: ignore

from config import (
    DATA_RAW,
    SAMPLE_END,
    SAMPLE_START,
    ensure_pgpass,
    get_wrds_credentials,
)


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    ensure_pgpass()
    username, _ = get_wrds_credentials()
    db = wrds.Connection(wrds_username=username)

    pull_start = SAMPLE_START - 1  # need fyear t-1 for lag price

    print("Pulling comp.funda ...")
    funda = db.raw_sql(
        f"""
        SELECT gvkey, datadate, fyear, fyr, indfmt, consol, popsrc, datafmt,
               epspx, prcc_f, ajex, csho, at, ceq, sich
        FROM comp.funda
        WHERE datadate BETWEEN '{pull_start}-01-01' AND '{SAMPLE_END}-12-31'
          AND indfmt  = 'INDL'
          AND consol  = 'C'
          AND popsrc  = 'D'
          AND datafmt = 'STD'
        """,
        date_cols=["datadate"],
    )
    funda.to_parquet(DATA_RAW / "funda.parquet", index=False)
    print(f"  funda: {len(funda):,} rows")

    print("Pulling crsp.msf ...")
    msf = db.raw_sql(
        f"""
        SELECT permno, date, ret, prc, shrout
        FROM crsp.msf
        WHERE date BETWEEN '{pull_start}-01-01' AND '{SAMPLE_END}-12-31'
        """,
        date_cols=["date"],
    )
    msf.to_parquet(DATA_RAW / "msf.parquet", index=False)
    print(f"  msf:   {len(msf):,} rows")

    print("Pulling crsp.stocknames ...")
    stocknames = db.raw_sql(
        """
        SELECT permno, namedt, nameenddt, shrcd, exchcd, ticker, comnam
        FROM crsp.stocknames
        """,
        date_cols=["namedt", "nameenddt"],
    )
    stocknames.to_parquet(DATA_RAW / "stocknames.parquet", index=False)
    print(f"  stocknames: {len(stocknames):,} rows")

    print("Pulling crsp.ccmxpf_lnkhist ...")
    ccm = db.raw_sql(
        """
        SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
        """,
        date_cols=["linkdt", "linkenddt"],
    )
    ccm.to_parquet(DATA_RAW / "ccm.parquet", index=False)
    print(f"  ccm: {len(ccm):,} rows")

    db.close()
    print("Done.")


if __name__ == "__main__":
    main()
