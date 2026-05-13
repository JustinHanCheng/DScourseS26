"""Estimate Basu (1997) asymmetric-timeliness regressions.

Specifications:
  (1) Full sample 2015--2024
  (2) Pre-COVID sub-sample (2015--2019)
  (3) COVID era sub-sample  (2020--2024)

Standard errors are clustered by firm (gvkey).  Year-by-year coefficients are
also written for the time-series figure produced in 07_make_figures.py.

Run:  python code/06_basu_regression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import statsmodels.formula.api as smf

from config import DATA_CLEAN, OUTPUT_TABLES


def fit_basu(df: pd.DataFrame):
    return smf.ols("EP ~ D + R + DR", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["gvkey"]}
    )


def _star(t: float) -> str:
    if abs(t) > 2.58:
        return "$^{***}$"
    if abs(t) > 1.96:
        return "$^{**}$"
    if abs(t) > 1.645:
        return "$^{*}$"
    return ""


def coef_row(fits: dict, var: str, label: str) -> list[str]:
    """Return two LaTeX lines: coefficients then standard errors."""
    coefs, ses = [], []
    for m in fits.values():
        c = m.params.get(var, float("nan"))
        s = m.bse.get(var, float("nan"))
        t = m.tvalues.get(var, float("nan"))
        coefs.append(f"{c:.4f}{_star(t)}")
        ses.append(f"({s:.4f})")
    return [
        f"{label} & " + " & ".join(coefs) + r" \\",
        " & " + " & ".join(ses) + r" \\",
    ]


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_CLEAN / "panel.parquet")

    specs = {
        "Full Sample": df,
        "Pre-COVID (2015--2019)": df[df["fyear"].between(2015, 2019)],
        "COVID Era (2020--2024)": df[df["fyear"].between(2020, 2025)],
    }
    fits = {name: fit_basu(sub) for name, sub in specs.items()}

    var_labels = [
        ("Intercept", "Intercept"),
        ("D", r"$D$ (Bad-news dummy)"),
        ("R", r"$R$ (Good-news slope, $\beta_2$)"),
        ("DR", r"$D \times R$ (Asym.\ timeliness, $\beta_3$)"),
    ]

    cols = list(fits.keys())
    n_cols = len(cols)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Basu (1997) asymmetric-timeliness regressions, 2015--2024."
        r" Dependent variable is $EP_{it} = \mathit{EPS}_{it} / P_{i,t-1}$."
        r" Heteroskedasticity-robust standard errors clustered by firm in"
        r" parentheses. $^{*}$, $^{**}$, $^{***}$ denote significance at the"
        r" 10\%, 5\%, and 1\% levels (two-sided).}",
        r"\label{tab:basu}",
        r"\begin{tabular}{l" + "c" * n_cols + r"}",
        r"\toprule",
        " & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for var, label in var_labels:
        lines.extend(coef_row(fits, var, label))
    lines += [
        r"\midrule",
        r"$N$ & " + " & ".join(f"{int(fits[c].nobs):,}" for c in cols) + r" \\",
        r"Adj. $R^2$ & " + " & ".join(f"{fits[c].rsquared_adj:.3f}" for c in cols) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (OUTPUT_TABLES / "table2_basu.tex").write_text("\n".join(lines), encoding="utf-8")

    yearly_rows = []
    for y, sub in df.groupby("fyear"):
        if len(sub) < 50:
            continue
        m = smf.ols("EP ~ D + R + DR", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["gvkey"]}
        )
        yearly_rows.append(
            {
                "year": int(y),
                "beta3": float(m.params["DR"]),
                "se_DR": float(m.bse["DR"]),
                "t_DR": float(m.tvalues["DR"]),
                "n": int(m.nobs),
                "r2": float(m.rsquared),
            }
        )
    pd.DataFrame(yearly_rows).to_csv(DATA_CLEAN / "yearly_betas.csv", index=False)

    for name, m in fits.items():
        print(f"\n=== {name} ===")
        print(m.summary().tables[1])


if __name__ == "__main__":
    main()
