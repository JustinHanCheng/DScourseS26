#!/usr/bin/env python3
"""
Econ 5253 - Problem Set 6
Last name: Cheng

This script downloads a public emissions dataset, cleans it, and creates
three publication-style visualizations for the assignment.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


DATA_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

SELECTED_COUNTRIES = [
    "Saudi Arabia",
    "United States",
    "China",
    "Germany",
    "Brazil",
    "India",
]

LABEL_COUNTRIES = [
    "United States",
    "China",
    "India",
    "Germany",
    "Brazil",
    "Saudi Arabia",
    "France",
    "Indonesia",
    "Australia",
    "South Korea",
]

COUNTRY_COLORS = {
    "Saudi Arabia": "#9f2d20",
    "United States": "#1d3557",
    "China": "#c1121f",
    "Germany": "#6c757d",
    "Brazil": "#2a9d8f",
    "India": "#f4a261",
}

LINE_LABEL_Y_OFFSETS = {
    "Saudi Arabia": 0.0,
    "United States": 0.0,
    "China": 0.18,
    "Germany": -0.18,
    "Brazil": 0.14,
    "India": -0.14,
}

SOURCE_LABELS = {
    "coal_share_pct": "Coal",
    "oil_share_pct": "Oil",
    "gas_share_pct": "Gas",
    "cement_share_pct": "Cement",
    "flaring_share_pct": "Flaring",
    "other_industry_share_pct": "Other industry",
}


def load_data():
    columns = [
        "country",
        "year",
        "iso_code",
        "population",
        "gdp",
        "co2",
        "co2_per_capita",
        "energy_per_capita",
        "coal_co2",
        "oil_co2",
        "gas_co2",
        "cement_co2",
        "flaring_co2",
        "other_industry_co2",
    ]
    return pd.read_csv(DATA_URL, usecols=columns)


def latest_complete_year(df):
    required_columns = [
        "population",
        "gdp",
        "co2",
        "co2_per_capita",
        "coal_co2",
        "oil_co2",
        "gas_co2",
        "cement_co2",
        "flaring_co2",
    ]
    complete_counts = (
        df[required_columns]
        .notna()
        .all(axis=1)
        .groupby(df["year"])
        .sum()
    )
    return int(complete_counts[complete_counts >= 100].index.max())


def clean_data(raw_df):
    df = raw_df[raw_df["iso_code"].fillna("").str.len() == 3].copy()
    df = df[df["year"] >= 1990].copy()

    complete_year = latest_complete_year(df)
    df = df[df["year"] <= complete_year].copy()
    df["gdp_per_capita"] = df["gdp"] / df["population"]

    source_columns = [
        "coal",
        "oil",
        "gas",
        "cement",
        "flaring",
        "other_industry",
    ]
    for source in source_columns:
        share_name = f"{source}_share_pct"
        df[share_name] = 100 * df[f"{source}_co2"] / df["co2"]

    df.loc[df["co2"] <= 0, list(SOURCE_LABELS)] = np.nan
    return df, complete_year


def style_plots():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.edgecolor": "#d0d0d0",
            "grid.color": "#d9d9d9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "font.size": 11,
        },
    )


def population_size(population):
    return np.sqrt(population / 1_000_000) * 18


def save_line_chart(df, final_year):
    plot_df = df[df["country"].isin(SELECTED_COUNTRIES)].copy()
    fig, ax = plt.subplots(figsize=(12, 7))

    for country in SELECTED_COUNTRIES:
        country_df = plot_df[plot_df["country"] == country].sort_values("year")
        ax.plot(
            country_df["year"],
            country_df["co2_per_capita"],
            color=COUNTRY_COLORS[country],
            linewidth=3,
            alpha=0.95,
        )
        last_row = country_df.iloc[-1]
        ax.scatter(
            last_row["year"],
            last_row["co2_per_capita"],
            color=COUNTRY_COLORS[country],
            s=70,
            zorder=3,
        )
        ax.text(
            final_year + 0.5,
            last_row["co2_per_capita"] + LINE_LABEL_Y_OFFSETS[country],
            "{0} ({1:.1f})".format(country, last_row["co2_per_capita"]),
            color=COUNTRY_COLORS[country],
            fontsize=10.5,
            va="center",
        )

    ax.set_title(
        "Per-capita CO2 emissions have followed very different paths",
        loc="left",
        fontsize=18,
        pad=18,
        fontweight="bold",
    )
    ax.text(
        0,
        1.02,
        "Selected large economies, 1990-{0}; labels show tonnes of CO2 per person in {0}".format(
            final_year
        ),
        transform=ax.transAxes,
        fontsize=11,
        color="#555555",
    )
    ax.set_xlim(1990, final_year + 4)
    ax.set_ylim(0, 24)
    ax.set_xlabel("")
    ax.set_ylabel("Tonnes of CO2 per person")
    ax.grid(axis="y", alpha=0.6)
    ax.grid(axis="x", alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "PS6a_Cheng.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_scatter(df, final_year):
    plot_df = df[df["year"] == final_year].copy()
    plot_df = plot_df[
        (plot_df["population"] >= 10_000_000)
        & plot_df[["gdp_per_capita", "co2_per_capita", "coal_share_pct"]]
        .notna()
        .all(axis=1)
        & (plot_df["gdp_per_capita"] > 0)
        & (plot_df["co2_per_capita"] > 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        plot_df["gdp_per_capita"],
        plot_df["co2_per_capita"],
        s=population_size(plot_df["population"]),
        c=plot_df["coal_share_pct"],
        cmap="YlOrRd",
        alpha=0.82,
        edgecolors="white",
        linewidth=0.7,
    )

    label_offsets = {
        "United States": (8, -8),
        "China": (8, 8),
        "India": (8, -2),
        "Germany": (8, 6),
        "Brazil": (8, -10),
        "Saudi Arabia": (8, 4),
        "France": (8, -10),
        "Indonesia": (8, 4),
        "Australia": (8, -8),
        "South Korea": (8, 6),
    }
    for country in LABEL_COUNTRIES:
        row = plot_df.loc[plot_df["country"] == country]
        if row.empty:
            continue
        row = row.iloc[0]
        ax.annotate(
            country,
            xy=(row["gdp_per_capita"], row["co2_per_capita"]),
            xytext=label_offsets[country],
            textcoords="offset points",
            fontsize=10,
            color="#222222",
        )

    ax.set_xscale("log")
    tick_values = [2_500, 5_000, 10_000, 20_000, 40_000, 80_000]
    ax.set_xticks(tick_values)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "${0:,.0f}".format(value)))
    ax.set_ylabel("Tonnes of CO2 per person")
    ax.set_xlabel("GDP per capita (constant 2021 international $; log scale)")
    ax.set_title(
        "Higher incomes tend to come with higher emissions, but energy mix matters",
        loc="left",
        fontsize=18,
        pad=18,
        fontweight="bold",
    )
    ax.text(
        0,
        1.02,
        "Countries with population of at least 10 million in {0}; bubble size reflects population".format(
            final_year
        ),
        transform=ax.transAxes,
        fontsize=11,
        color="#555555",
    )
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Coal share of national CO2 emissions (%)")

    legend_populations = [25_000_000, 100_000_000, 500_000_000]
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#bbbbbb",
            markeredgecolor="white",
            markeredgewidth=0.8,
            alpha=0.8,
            markersize=np.sqrt(population_size(population)),
            label="{0:,} people".format(population),
        )
        for population in legend_populations
    ]
    ax.legend(
        handles=legend_handles,
        title="Bubble size",
        frameon=False,
        loc="upper left",
        fontsize=9.5,
        title_fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "PS6b_Cheng.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(df, final_year):
    plot_df = (
        df[df["year"] == final_year]
        .dropna(subset=["co2"])
        .sort_values("co2", ascending=False)
        .head(12)
        .copy()
    )

    heatmap_df = plot_df.set_index("country")[list(SOURCE_LABELS)].copy()
    heatmap_df = heatmap_df.fillna(0).rename(columns=SOURCE_LABELS).round(1)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        heatmap_df,
        cmap=sns.light_palette("#9f2d20", as_cmap=True),
        annot=True,
        fmt=".1f",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Share of national CO2 emissions (%)"},
        ax=ax,
    )
    ax.set_title(
        "The largest emitters reach similar totals through different fuel mixes",
        loc="left",
        fontsize=18,
        pad=18,
        fontweight="bold",
    )
    ax.text(
        0,
        1.02,
        "Top 12 countries by total CO2 emissions in {0}".format(final_year),
        transform=ax.transAxes,
        fontsize=11,
        color="#555555",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "PS6c_Cheng.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(df, final_year):
    latest_df = df[df["year"] == final_year].copy()
    summary = latest_df.sort_values("co2", ascending=False).head(10)[
        ["country", "co2", "co2_per_capita", "gdp_per_capita"]
    ]

    print("Cleaned observations:", len(df))
    print("Countries retained:", df["country"].nunique())
    print("Years covered: {0}-{1}".format(int(df["year"].min()), int(df["year"].max())))
    print("Latest complete year used for the cross-section:", final_year)
    print("\nTop emitters in the analysis year:")
    print(summary.to_string(index=False, float_format=lambda value: "{0:,.2f}".format(value)))


def main():
    style_plots()
    raw_df = load_data()
    clean_df, final_year = clean_data(raw_df)
    save_line_chart(clean_df, final_year)
    save_scatter(clean_df, final_year)
    save_heatmap(clean_df, final_year)
    print_summary(clean_df, final_year)


if __name__ == "__main__":
    main()
