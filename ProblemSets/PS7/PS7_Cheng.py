#!/usr/bin/env python3
"""
Econ 5253 - Problem Set 7
Last name: Cheng

This script reproduces the PS7 analysis in Python. It:
1. Loads wages.csv from the current folder
2. Drops observations with missing hgc or tenure
3. Creates a summary-statistics table
4. Estimates four regressions under different treatments of missing log wages
5. Exports LaTeX tables used in PS7_Cheng.tex
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.imputation.mice import MICEData, MICE
from statsmodels.iolib.summary2 import summary_col


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUTPUT_DIR / "wages.csv"
SUMMARY_TABLE_PATH = OUTPUT_DIR / "summary_table.tex"
REGRESSION_TABLE_PATH = OUTPUT_DIR / "regression_table.tex"
RANDOM_SEED = 5253
TRUE_BETA1 = 0.093

FORMULA = (
    "logwage ~ hgc + college_grad + tenure + I(tenure ** 2) + age + married_dummy"
)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "wages.csv was not found in the current folder: {0}".format(DATA_PATH)
        )
    return pd.read_csv(DATA_PATH)


def prepare_data(raw_df):
    df = raw_df.dropna(subset=["hgc", "tenure"]).copy()
    df["college_grad"] = (df["college"] == "college grad").astype(int)
    df["married_dummy"] = (df["married"] == "married").astype(int)
    df["missing_logwage"] = df["logwage"].isna().astype(int)
    return df


def summary_statistics_table(df):
    summary_df = pd.DataFrame(
        [
            {
                "Variable": "Log wage",
                "N": int(df["logwage"].notna().sum()),
                "Mean": df["logwage"].mean(),
                "SD": df["logwage"].std(),
                "Min": df["logwage"].min(),
                "Median": df["logwage"].median(),
                "Max": df["logwage"].max(),
                "Missing": int(df["logwage"].isna().sum()),
            },
            {
                "Variable": "Years of schooling",
                "N": int(df["hgc"].notna().sum()),
                "Mean": df["hgc"].mean(),
                "SD": df["hgc"].std(),
                "Min": df["hgc"].min(),
                "Median": df["hgc"].median(),
                "Max": df["hgc"].max(),
                "Missing": int(df["hgc"].isna().sum()),
            },
            {
                "Variable": "College graduate",
                "N": int(df["college_grad"].notna().sum()),
                "Mean": df["college_grad"].mean(),
                "SD": df["college_grad"].std(),
                "Min": df["college_grad"].min(),
                "Median": df["college_grad"].median(),
                "Max": df["college_grad"].max(),
                "Missing": int(df["college_grad"].isna().sum()),
            },
            {
                "Variable": "Tenure",
                "N": int(df["tenure"].notna().sum()),
                "Mean": df["tenure"].mean(),
                "SD": df["tenure"].std(),
                "Min": df["tenure"].min(),
                "Median": df["tenure"].median(),
                "Max": df["tenure"].max(),
                "Missing": int(df["tenure"].isna().sum()),
            },
            {
                "Variable": "Age",
                "N": int(df["age"].notna().sum()),
                "Mean": df["age"].mean(),
                "SD": df["age"].std(),
                "Min": df["age"].min(),
                "Median": df["age"].median(),
                "Max": df["age"].max(),
                "Missing": int(df["age"].isna().sum()),
            },
            {
                "Variable": "Married",
                "N": int(df["married_dummy"].notna().sum()),
                "Mean": df["married_dummy"].mean(),
                "SD": df["married_dummy"].std(),
                "Min": df["married_dummy"].min(),
                "Median": df["married_dummy"].median(),
                "Max": df["married_dummy"].max(),
                "Missing": int(df["married_dummy"].isna().sum()),
            },
        ]
    )

    display_df = summary_df.copy()
    display_df["N"] = display_df["N"].map("{:d}".format)
    display_df["Missing"] = display_df["Missing"].map("{:d}".format)
    for column in ["Mean", "SD", "Min", "Median", "Max"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.3f}")

    latex_body = display_df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrrrr",
    )

    latex_table = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Summary statistics after dropping missing schooling or tenure observations}\n"
        "\\label{tab:summary}\n"
        f"{latex_body}"
        "\\vspace{0.15cm}\n"
        "\\parbox{0.92\\textwidth}{\\small Notes: The cleaned sample contains 2,229 observations. "
        "For indicator variables, the mean can be interpreted as the sample share.}\n"
        "\\end{table}\n"
    )

    SUMMARY_TABLE_PATH.write_text(latex_table, encoding="utf-8")
    return summary_df


def run_models(df):
    np.random.seed(RANDOM_SEED)

    complete_cases = df.dropna(subset=["logwage"]).copy()
    complete_model = smf.ols(FORMULA, data=complete_cases).fit()

    mean_imputed = df.copy()
    mean_imputed["logwage"] = mean_imputed["logwage"].fillna(
        mean_imputed["logwage"].mean()
    )
    mean_model = smf.ols(FORMULA, data=mean_imputed).fit()

    predicted_imputed = df.copy()
    predicted_values = complete_model.predict(predicted_imputed)
    missing_mask = predicted_imputed["logwage"].isna()
    predicted_imputed.loc[missing_mask, "logwage"] = predicted_values[missing_mask]
    predicted_model = smf.ols(FORMULA, data=predicted_imputed).fit()

    imputation_df = df[
        ["logwage", "hgc", "college_grad", "tenure", "age", "married_dummy"]
    ].copy()
    mice_data = MICEData(imputation_df)
    mice_model = MICE(FORMULA, sm.OLS, mice_data)
    mice_results = mice_model.fit(20, 20)

    return {
        "complete": complete_model,
        "mean": mean_model,
        "predicted": predicted_model,
        "mice": mice_results,
    }


def regression_table(models):
    results = [
        models["complete"],
        models["mean"],
        models["predicted"],
        models["mice"],
    ]

    info_dict = {
        "N": lambda result: (
            f"{int(result.nobs)}"
            if hasattr(result, "nobs")
            else f"{int(result.model.data.data.shape[0])}"
        ),
        "R2": lambda result: (
            f"{result.rsquared:.3f}" if hasattr(result, "rsquared") else ""
        ),
    }

    table = summary_col(
        results,
        model_names=[
            "Complete cases",
            "Mean imputation",
            "Predicted imputation",
            "MICE",
        ],
        float_format="%.4f",
        stars=True,
        info_dict=info_dict,
        regressor_order=[
            "hgc",
            "college_grad",
            "tenure",
            "I(tenure ** 2)",
            "age",
            "married_dummy",
            "Intercept",
        ],
        include_r2=False,
    )

    latex = table.as_latex()
    latex = latex.replace("\\begin{table}", "\\begin{table}[H]")
    latex = latex.replace(
        "\\caption{}",
        "\\caption{Regression estimates under alternative treatments of missing log wages}",
    )
    latex = latex.replace("\\label{}", "\\label{tab:regression}")

    row_replacements = {
        r"^\s*hgc\s*&": "Years of schooling &",
        r"^\s*college\\_grad\s*&": "College graduate &",
        r"^\s*tenure\s*&": "Tenure &",
        r"^\s*I\(tenure \*\* 2\)\s*&": "Tenure squared &",
        r"^\s*age\s*&": "Age &",
        r"^\s*married\\_dummy\s*&": "Married &",
        r"^\s*Intercept\s*&": "Constant &",
        r"^\s*R2\s*&": "$R^2$ &",
    }
    for pattern, replacement in row_replacements.items():
        latex = re.sub(pattern, replacement, latex, flags=re.MULTILINE)

    REGRESSION_TABLE_PATH.write_text(latex, encoding="utf-8")
    return table


def format_share(value):
    return "{0:.1f}\\%".format(100 * value)


def print_key_results(df, models):
    missing_rate = df["missing_logwage"].mean()
    college_missing_rates = (
        df.groupby("college_grad")["missing_logwage"]
        .mean()
        .rename({0: "not college grad", 1: "college grad"})
    )

    beta_values = {
        "Complete cases": models["complete"].params["hgc"],
        "Mean imputation": models["mean"].params["hgc"],
        "Predicted imputation": models["predicted"].params["hgc"],
        "MICE": dict(zip(models["mice"].exog_names, models["mice"].params))["hgc"],
    }

    print("Cleaned sample size:", len(df))
    print(
        "Missing log wage rate: {0} ({1}/{2})".format(
            format_share(missing_rate),
            int(df["missing_logwage"].sum()),
            len(df),
        )
    )
    print(
        "Missing rate by college status: college grad = {0}; not college grad = {1}".format(
            format_share(college_missing_rates["college grad"]),
            format_share(college_missing_rates["not college grad"]),
        )
    )
    print("\nEstimated returns to schooling (beta_1):")
    for name, value in beta_values.items():
        gap = value - TRUE_BETA1
        print(
            "  {0}: {1:.4f} (difference from true value 0.093 = {2:+.4f})".format(
                name,
                value,
                gap,
            )
        )


def main():
    raw_df = load_data()
    clean_df = prepare_data(raw_df)
    summary_statistics_table(clean_df)
    models = run_models(clean_df)
    regression_table(models)
    print_key_results(clean_df, models)


if __name__ == "__main__":
    main()
