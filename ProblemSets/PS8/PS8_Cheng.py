#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS


OUTPUT_DIR = Path(__file__).resolve().parent
ESTIMATE_TABLE_PATH = OUTPUT_DIR / "estimate_comparison.tex"
REGRESSION_TABLE_PATH = OUTPUT_DIR / "regression_table.tex"
SUMMARY_PATH = OUTPUT_DIR / "ps8_summary.txt"

SEED = 100
N = 100_000
K = 10
SIGMA_TRUE = 0.5
LEARNING_RATE = 0.0000003
MAX_GD_ITER = 5_000
GD_TOL = 1e-12

BETA_TRUE = np.array([1.5, -1.0, -0.25, 0.75, 3.5, -2.0, 0.5, 1.0, 1.25, 2.0])
PARAMETER_NAMES = [
    "Constant",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "x8",
    "x9",
    "x10",
]


def simulate_data():
    np.random.seed(SEED)
    x = np.random.normal(size=(N, K))
    x[:, 0] = 1.0
    eps = np.random.normal(loc=0.0, scale=SIGMA_TRUE, size=N)
    y = x @ BETA_TRUE + eps
    return x, y, eps


def ols_closed_form(x, y):
    return np.linalg.solve(x.T @ x, x.T @ y)


def sse(beta, x, y):
    resid = y - x @ beta
    return float(resid @ resid)


def sse_gradient(beta, x, y):
    resid = y - x @ beta
    return -2.0 * (x.T @ resid)


def gradient_descent(x, y):
    beta = np.zeros(x.shape[1])
    iterations = 0
    for iteration in range(1, MAX_GD_ITER + 1):
        grad = sse_gradient(beta, x, y)
        beta_new = beta - LEARNING_RATE * grad
        iterations = iteration
        if np.max(np.abs(beta_new - beta)) < GD_TOL:
            beta = beta_new
            break
        beta = beta_new
    return beta, iterations


def ols_lbfgs(x, y):
    return minimize(
        sse,
        x0=np.zeros(x.shape[1]),
        args=(x, y),
        method="L-BFGS-B",
        jac=sse_gradient,
        options={"maxiter": 1_000},
    )


def ols_nelder_mead(x, y):
    return minimize(
        sse,
        x0=np.zeros(x.shape[1]),
        args=(x, y),
        method="Nelder-Mead",
        options={
            "maxiter": 60_000,
            "xatol": 1e-10,
            "fatol": 1e-6,
        },
    )


def negative_log_likelihood(theta, x, y):
    beta = theta[:-1]
    sigma = theta[-1]
    if sigma <= 0:
        return np.inf
    resid = y - x @ beta
    sum_sq = float(resid @ resid)
    return 0.5 * len(y) * np.log(2.0 * np.pi) + len(y) * np.log(sigma) + sum_sq / (
        2.0 * sigma**2
    )


def negative_log_likelihood_gradient(theta, x, y):
    beta = theta[:-1]
    sigma = theta[-1]
    resid = y - x @ beta
    grad_beta = -(x.T @ resid) / (sigma**2)
    grad_sigma = len(y) / sigma - float(resid @ resid) / (sigma**3)
    return np.concatenate([grad_beta, [grad_sigma]])


def mle_lbfgs(x, y, beta_start):
    sigma_start = np.sqrt(np.mean((y - x @ beta_start) ** 2))
    theta_start = np.concatenate([beta_start, [sigma_start]])
    bounds = [(None, None)] * x.shape[1] + [(1e-8, None)]
    return minimize(
        negative_log_likelihood,
        x0=theta_start,
        args=(x, y),
        method="L-BFGS-B",
        jac=negative_log_likelihood_gradient,
        bounds=bounds,
        options={"maxiter": 1_000},
    )


def statsmodels_ols(x, y):
    return OLS(y, x).fit()


def build_estimate_table(beta_closed, beta_gd, lbfgs_result, nm_result, mle_result, ols_model):
    comparison = pd.DataFrame(
        {
            "Parameter": PARAMETER_NAMES,
            "True $\\beta$": BETA_TRUE,
            "Closed form": beta_closed,
            "Gradient descent": beta_gd,
            "L-BFGS": lbfgs_result.x,
            "Nelder-Mead": nm_result.x,
            "MLE": mle_result.x[:-1],
            "OLS (easy way)": ols_model.params,
            "Closed form - true": beta_closed - BETA_TRUE,
        }
    )

    latex = comparison.to_latex(
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.6f}",
        column_format="lrrrrrrrr",
    )
    latex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Coefficient estimates across methods}\n"
        "\\label{tab:comparison}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{latex}"
        "}\n"
        "\\end{table}\n"
    )
    ESTIMATE_TABLE_PATH.write_text(latex, encoding="utf-8")
    return comparison


def build_regression_table(ols_model):
    reg_table = pd.DataFrame(
        {
            "Regressor": PARAMETER_NAMES,
            "Estimate": ols_model.params,
            "Std. Error": ols_model.bse,
            "t statistic": ols_model.tvalues,
            "p value": ols_model.pvalues,
        }
    )
    latex_body = reg_table.to_latex(
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.4f}",
        column_format="lrrrr",
    )
    latex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{OLS regression output using the direct matrix call}\n"
        "\\label{tab:ols}\n"
        f"{latex_body}"
        f"\\vspace{{0.15cm}}\n"
        f"\\parbox{{0.9\\textwidth}}{{\\small Notes: The regression is estimated with "
        f"\\texttt{{statsmodels}} using \\texttt{{OLS(Y, X)}}. The sample size is {int(ols_model.nobs):,} "
        f"and $R^2 = {ols_model.rsquared:.4f}$.}}\n"
        "\\end{table}\n"
    )
    REGRESSION_TABLE_PATH.write_text(latex, encoding="utf-8")
    return latex


def write_summary(beta_closed, beta_gd, lbfgs_result, nm_result, mle_result, ols_model, gd_iterations):
    max_abs_gap = float(np.max(np.abs(beta_closed - BETA_TRUE)))
    max_diff_gd = float(np.max(np.abs(beta_closed - beta_gd)))
    max_diff_lbfgs = float(np.max(np.abs(beta_closed - lbfgs_result.x)))
    max_diff_nm = float(np.max(np.abs(beta_closed - nm_result.x)))
    max_diff_mle = float(np.max(np.abs(beta_closed - mle_result.x[:-1])))
    max_diff_easy = float(np.max(np.abs(beta_closed - ols_model.params)))
    sigma_mle = float(mle_result.x[-1])

    lines = [
        f"Seed: {SEED}",
        f"N: {N}",
        f"K: {K}",
        f"True sigma: {SIGMA_TRUE:.6f}",
        f"Gradient descent learning rate: {LEARNING_RATE:.7f}",
        f"Gradient descent iterations: {gd_iterations}",
        f"Maximum absolute deviation of closed-form OLS from true beta: {max_abs_gap:.6f}",
        f"Maximum absolute difference between closed-form OLS and gradient descent: {max_diff_gd:.12f}",
        f"Maximum absolute difference between closed-form OLS and L-BFGS: {max_diff_lbfgs:.12f}",
        f"Maximum absolute difference between closed-form OLS and Nelder-Mead: {max_diff_nm:.12f}",
        f"Maximum absolute difference between closed-form OLS and MLE beta: {max_diff_mle:.12f}",
        f"Maximum absolute difference between closed-form OLS and OLS easy way: {max_diff_easy:.12f}",
        f"MLE sigma: {sigma_mle:.6f}",
        f"L-BFGS success: {lbfgs_result.success}",
        f"Nelder-Mead success: {nm_result.success}",
        f"MLE success: {mle_result.success}",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main():
    x, y, eps = simulate_data()
    beta_closed = ols_closed_form(x, y)
    beta_gd, gd_iterations = gradient_descent(x, y)
    lbfgs_result = ols_lbfgs(x, y)
    nm_result = ols_nelder_mead(x, y)
    mle_result = mle_lbfgs(x, y, beta_closed)
    ols_model = statsmodels_ols(x, y)

    build_estimate_table(
        beta_closed=beta_closed,
        beta_gd=beta_gd,
        lbfgs_result=lbfgs_result,
        nm_result=nm_result,
        mle_result=mle_result,
        ols_model=ols_model,
    )
    build_regression_table(ols_model)
    summary_lines = write_summary(
        beta_closed=beta_closed,
        beta_gd=beta_gd,
        lbfgs_result=lbfgs_result,
        nm_result=nm_result,
        mle_result=mle_result,
        ols_model=ols_model,
        gd_iterations=gd_iterations,
    )

    print("Closed-form OLS estimates:")
    for name, value, truth in zip(PARAMETER_NAMES, beta_closed, BETA_TRUE):
        print(f"{name:>8s}: {value: .6f}   true beta = {truth: .6f}   gap = {value - truth: .6f}")

    print()
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
