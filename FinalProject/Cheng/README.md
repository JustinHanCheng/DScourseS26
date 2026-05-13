# Replicating Basu (1997) on U.S. firms, 2015–2024

Final project for ECON 5253 (Data Science for Economists).

This repository replicates the asymmetric-timeliness regression of
Basu (1997, *JAE*) on a fresh ten-year panel of U.S. listed firms drawn from
the WRDS Compustat–CRSP merged universe.

## Repository layout

```
.
├── README.md                 # this file
├── requirements.txt
├── config.py                 # paths, sample window, WRDS auth helper
├── WRDS.txt                  # local WRDS credentials (NOT to be redistributed)
├── code/
│   ├── run_all.py            # one-command reproduction of the data pipeline
│   ├── 01_pull_wrds.py       # download Compustat funda + CRSP msf + CCM link
│   ├── 02_clean_merge.py     # CCM link, sample filters
│   ├── 03_compute_returns.py # compound 12-month fiscal-year returns
│   ├── 04_construct_vars.py  # EP, R, D, DR; winsorise
│   ├── 05_descriptive.py     # Table 1 (descriptive statistics)
│   ├── 06_basu_regression.py # Table 2 (Basu regressions, full + sub-samples)
│   └── 07_make_figures.py    # Figure 1 (year-by-year β₃)
├── data/
│   ├── raw/                  # parquet files written by 01_pull_wrds.py
│   └── clean/                # intermediate and final firm-year panel
├── output/
│   ├── tables/               # LaTeX tables included by paper/main.tex
│   └── figures/              # PDF / EPS / PNG figures
└── paper/
    ├── main.tex              # 10-page written report (source)
    ├── main.pdf              # compiled report
    ├── References.bib        # bibliography
    ├── slides.tex            # 5-slide Beamer presentation (source)
    └── slides.pdf            # compiled slide deck
```

## Prerequisites

* Python ≥ 3.10
* A working WRDS account with access to `comp.funda`, `crsp.msf`, and
  `crsp.ccmxpf_lnkhist`. Place the credentials in `WRDS.txt` (two lines:
  `Username: ...` and `Password: ...`).
* A LaTeX distribution (TeX Live, MiKTeX) with the `metropolis` Beamer theme
  installed.

```bash
pip install -r requirements.txt
```

## Reproducing the data pipeline

```bash
python code/run_all.py
```

The script runs steps 01--07:

1. `code/01_pull_wrds.py` writes `.pgpass` (using `WRDS.txt`) and pulls the
   four raw tables to `data/raw/`.
2. `code/02_clean_merge.py` merges Compustat ↔ CRSP through CCM and applies
   sample filters (US common stock, NYSE/AMEX/NASDAQ, December fiscal
   year-ends, non-financial, price > $1).
3. `code/03_compute_returns.py` compounds 12 monthly returns into the
   fiscal-year return $R_{it}$.
4. `code/04_construct_vars.py` constructs $EP_{it}$, $D_{it}$ and the
   interaction, winsorises at 1% / 99%.
5. `code/05_descriptive.py` writes Table 1 and the yearly-count table;
   `code/06_basu_regression.py` writes Table 2 and the year-by-year
   coefficient CSV; `code/07_make_figures.py` turns that CSV into Figure 1.

To re-run just the analysis after the data has been pulled once:

```bash
python code/run_all.py --skip-data
```

## Compiling the paper

The PDF (`paper/main.pdf`) and the slide deck (`paper/slides.pdf`) are
already included in this repository. To rebuild them from source after editing:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex slides.tex
```

## Key specification

The Basu (1997) regression estimated by `06_basu_regression.py`:

$$
EP_{it} \;=\; \beta_0 + \beta_1 D_{it} + \beta_2 R_{it}
        + \beta_3 \, D_{it} R_{it} + \varepsilon_{it}
$$

where $EP_{it} = \mathrm{EPS}_{it} / P_{i,t-1}$ (split-adjusted),
$R_{it}$ is the 12-month fiscal-year stock return, and $D_{it}=\mathbf{1}\{R_{it}<0\}$.
Standard errors are clustered by firm (`gvkey`).

## Notes / caveats

* `WRDS.txt` is project-local and excluded from any redistribution; the WRDS
  EULA prohibits sharing credentials.
* Coverage of `comp.funda` for the most recent fiscal year (2025) is
  incomplete at the time of writing; the analysis sample reports the actual
  range printed by `04_construct_vars.py`.
* Standard errors are firm-clustered only; two-way (firm × year) clustering
  is left for future work.
