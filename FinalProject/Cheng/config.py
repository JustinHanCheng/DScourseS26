"""Project paths and WRDS credential helper.

Reads WRDS username / password from WRDS.txt at the project root and (on first
use) writes a `.pgpass` line so the `wrds` Python package can authenticate
non-interactively.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
OUTPUT_TABLES = ROOT / "output" / "tables"
OUTPUT_FIGURES = ROOT / "output" / "figures"

# Sample window. We pull one extra prior fiscal year (SAMPLE_START - 1) so we
# can construct a beginning-of-period (lagged) price for every observation in
# the analysis window.
SAMPLE_START = 2015
SAMPLE_END = 2025

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"


def get_wrds_credentials() -> tuple[str, str]:
    creds_file = ROOT / "WRDS.txt"
    creds: dict[str, str] = {}
    for line in creds_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        creds[key.strip().lower()] = value.strip()
    return creds["username"], creds["password"]


def ensure_pgpass() -> None:
    """Append a `.pgpass` line for WRDS so `wrds.Connection()` can log in."""
    username, password = get_wrds_credentials()
    line = f"{WRDS_HOST}:{WRDS_PORT}:{WRDS_DB}:{username}:{password}"

    pgpass = Path.home() / ".pgpass"
    existing = pgpass.read_text(encoding="utf-8") if pgpass.exists() else ""
    if line in existing:
        return
    with pgpass.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(line + "\n")
    try:
        os.chmod(pgpass, 0o600)
    except OSError:
        pass
