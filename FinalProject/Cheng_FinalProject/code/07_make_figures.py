"""Plot the year-by-year asymmetric-timeliness coefficient (beta_3) with 95%
confidence bars.

Run:  python code/07_make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_CLEAN, OUTPUT_FIGURES


def main() -> None:
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    yearly = pd.read_csv(DATA_CLEAN / "yearly_betas.csv")
    yearly = yearly.sort_values("year").reset_index(drop=True)
    yearly["ci"] = 1.96 * yearly["se_DR"]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(
        yearly["year"], yearly["beta3"],
        yerr=yearly["ci"], capsize=4,
        color="steelblue", edgecolor="black", linewidth=0.7,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel(r"$\beta_3$ (asymmetric timeliness)")
    ax.set_title("Year-by-year Basu coefficient with 95% CI, 2015--2024")
    ax.set_xticks(yearly["year"])
    plt.tight_layout()

    for ext in ("pdf", "eps", "png"):
        fig.savefig(OUTPUT_FIGURES / f"fig1_beta3_yearly.{ext}", dpi=200)
    plt.close(fig)
    print(f"Wrote {OUTPUT_FIGURES / 'fig1_beta3_yearly.pdf'}")


if __name__ == "__main__":
    main()
