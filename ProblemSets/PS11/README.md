# PS11 — Cheng

Rough-draft submission for the Econ 5253 final project (Problem Set 11,
due 2026-04-28).

## Topic

*Predicting High Earnings: A Machine Learning Comparison Using the
U.S. Census Adult Income Data.*

The draft tunes and compares five supervised classifiers (LASSO logit,
classification tree, single-hidden-layer neural net, k-NN, and RBF-SVM)
on the UCI Adult Income data, building directly on the Problem Set 10
pipeline.

## Files

| File | Purpose |
|------|---------|
| `PS11_Cheng.tex` | LaTeX source for the rough draft |
| `PS11_Cheng.bib` | Bibliography (12 references) |
| `PS11_Cheng.pdf` | Compiled PDF (13 pages) |
| `PS11_Cheng.R`   | Reproducibility driver: refits all five tuned classifiers |
| `PS11_Cheng_figure.py` | Generates `fig_accuracy.pdf` from the PS10 numerics |
| `fig_accuracy.pdf` / `fig_accuracy.png` | Figure 1 in the draft |
| `PS11_Cheng_results.txt` | Numeric summary written by `PS11_Cheng.R` |

## Replication

From this folder:

```bash
# Regenerate the figure
python PS11_Cheng_figure.py

# Refit all five tuned classifiers (re-creates the Table 2 numbers)
Rscript PS11_Cheng.R

# Compile the report
pdflatex PS11_Cheng.tex
bibtex   PS11_Cheng
pdflatex PS11_Cheng.tex
pdflatex PS11_Cheng.tex
```

The `Rscript` step downloads the UCI Adult Income data, sets seed 100,
constructs the 80/20 split, tunes each model with 3-fold cross-validation,
and writes a numeric summary to `PS11_Cheng_results.txt`. The numeric
entries in Table 2 of the PDF are the same ones computed in Problem Set 10
(see `PS 10/PS10_Cheng.R`).

## Course-rubric checklist

- [x] Six sections: introduction, literature review, data, methods,
      findings, conclusion
- [x] At least one properly formatted equation (Eq. 1, Section 4.1)
- [x] At least five references in the bibliography (12 total)
- [x] In-text citations via `\citet{}`
- [x] At least one table and one figure (2 tables, 1 figure)
- [x] Compiled in LaTeX
- [x] Tables and figures appear after the references
- [x] Source code of report included alongside scripts
