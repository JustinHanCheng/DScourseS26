"""Descriptive statistics for the firm-year panel; writes Table 1 (LaTeX).

Run:  python code/05_descriptive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import DATA_CLEAN, OUTPUT_TABLES


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_CLEAN / "panel.parquet")

    stats = (
        df[["EP", "R", "D", "DR"]]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T[["count", "mean", "std", "25%", "50%", "75%"]]
    )
    stats.columns = ["N", "Mean", "SD", "P25", "Median", "P75"]
    stats["N"] = stats["N"].astype(int)

    rename = {
        "EP": r"$EP$ (Earnings/Price)",
        "R": r"$R$ (Annual return)",
        "D": r"$D$ (Bad-news dummy)",
        "DR": r"$D \times R$",
    }
    stats.index = [rename[i] for i in stats.index]

    (OUTPUT_TABLES / "table1_descriptive.tex").write_text(
        stats.to_latex(
            float_format="%.4f",
            escape=False,
            caption=(
                "Pooled firm-year statistics. Sample: U.S. listed firms with "
                "December fiscal year-ends, 2015--2024. $EP$ is EPS scaled by "
                "the split-adjusted beginning-of-period price; $R$ is the "
                "firm's compounded 12-month return for the fiscal year; "
                "$D=\\mathbf{1}\\{R<0\\}$. I winsorize $EP$ and $R$ at the "
                "pooled 1\\%/99\\% tails before computing what follows."
            ),
            label="tab:desc",
        ),
        encoding="utf-8",
    )

    yearly = df.groupby("fyear").agg(
        N=("EP", "size"),
        bad_pct=("D", "mean"),
    )
    yearly["bad_pct"] = yearly["bad_pct"].mul(100)
    yearly.index = yearly.index.astype(int)
    yearly.index.name = "Year"

    years = list(yearly.index)
    n_vals = [int(row["N"]) for _, row in yearly.iterrows()]
    pct_vals = [row["bad_pct"] for _, row in yearly.iterrows()]

    col_spec = "l" + "r" * len(years)
    lines_y = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Annual sample size and share of bad-news firm-years.}",
        r"\label{tab:yearly_n}",
        r"\small",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        " & ".join([""] + [str(y) for y in years]) + r" \\",
        r"\midrule",
        " & ".join(["$N$"] + [f"{n:,}" for n in n_vals]) + r" \\",
        " & ".join([r"\% Bad news"] + [f"{p:.1f}" for p in pct_vals]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUTPUT_TABLES / "table_yearly_n.tex").write_text(
        "\n".join(lines_y), encoding="utf-8"
    )

    print("Descriptive statistics:\n", stats)
    print("\nObservations by fiscal year:\n", yearly)


if __name__ == "__main__":
    main()
