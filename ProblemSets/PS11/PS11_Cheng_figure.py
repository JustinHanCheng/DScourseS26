"""
PS11_Cheng_figure.py
Generates the bar chart comparing test-set accuracy of five tuned classifiers
on the UCI Adult Income data. The numerical inputs come from the tuned
results reported in PS10_Cheng.R / PS10_Cheng.tex.

Output: fig_accuracy.pdf  (referenced by PS11_Cheng.tex)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent

models = [
    "Logistic\n(LASSO)",
    "Decision\nTree",
    "Neural\nNetwork",
    "$k$-NN",
    "SVM\n(RBF)",
]

cv_acc = [0.8452, 0.8611, 0.8471, 0.8362, 0.8524]
test_acc = [0.8531, 0.8684, 0.8544, 0.8442, 0.8634]

x = range(len(models))
width = 0.38

fig, ax = plt.subplots(figsize=(7.0, 4.2))

bars_cv = ax.bar([i - width / 2 for i in x], cv_acc, width=width,
                 label="3-fold CV accuracy", color="#4477AA", edgecolor="black")
bars_te = ax.bar([i + width / 2 for i in x], test_acc, width=width,
                 label="Test-set accuracy", color="#EE6677", edgecolor="black")

for bars in (bars_cv, bars_te):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.001,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylim(0.80, 0.88)
ax.set_ylabel("Classification accuracy")
ax.set_title("Cross-validated and out-of-sample accuracy by classifier")
ax.legend(loc="lower right", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.5)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_accuracy.pdf")
fig.savefig(OUT_DIR / "fig_accuracy.png", dpi=150)
print("Wrote", OUT_DIR / "fig_accuracy.pdf")
