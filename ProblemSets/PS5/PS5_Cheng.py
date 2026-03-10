#!/usr/bin/env python3
"""
Econ 5253 - Problem Set 5
Last name: Cheng

This script completes both required tasks:
1) Scrape tabular data from a webpage that does not provide an API.
2) Pull tabular data from a public API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup


USER_AGENT = "PS5_Cheng/1.0 (educational use; contact: cheng@ou.edu)"
WEB_URL = "https://www.worldometers.info/world-population/population-by-country/"
API_URL = "https://api.worldbank.org/v2/country/USA;CHN;IND/indicator/NY.GDP.PCAP.CD"


@dataclass(frozen=True)
class Selectors:
    """CSS selectors identified with SelectorGadget."""

    table_rows: str = "table.datatable tbody tr"
    row_cells: str = "td"


def _clean_numeric(text: str, keep_percent: bool = False) -> float:
    """Convert raw HTML cell text to a numeric value when possible."""
    if text is None:
        return float("nan")

    normalized = text.strip()
    normalized = (
        normalized.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("‑", "-")
        .replace(",", "")
    )
    if keep_percent:
        normalized = normalized.replace("%", "")

    normalized = re.sub(r"[^0-9.\-]", "", normalized)
    return pd.to_numeric(normalized, errors="coerce")


def scrape_population_table() -> pd.DataFrame:
    """Scrape country population data from a non-API web page."""
    response = requests.get(WEB_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    # Worldometer may default to ISO-8859-1 in requests; force UTF-8 to keep minus signs.
    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select(Selectors.table_rows)
    if not rows:
        raise RuntimeError("Could not find table rows using the CSS selector.")

    records: list[dict[str, object]] = []
    for row in rows:
        cells = [cell.get_text(" ", strip=True) for cell in row.select(Selectors.row_cells)]
        if len(cells) < 12:
            continue

        records.append(
            {
                "rank": _clean_numeric(cells[0]),
                "country": cells[1],
                "population_2026": _clean_numeric(cells[2]),
                "yearly_change_pct": _clean_numeric(cells[3], keep_percent=True),
                "net_change": _clean_numeric(cells[4]),
                "median_age": _clean_numeric(cells[9]),
                "urban_pop_pct": _clean_numeric(cells[10], keep_percent=True),
                "world_share_pct": _clean_numeric(cells[11], keep_percent=True),
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("rank").reset_index(drop=True)
    return df


def fetch_world_bank_api() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch GDP per capita data from the World Bank API."""
    response = requests.get(API_URL, params={"format": "json", "per_page": 200}, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("Unexpected API response structure from World Bank.")

    records: list[dict[str, object]] = []
    for entry in payload[1]:
        value = entry.get("value")
        if value is None:
            continue

        year = int(entry["date"])
        if year < 2015:
            continue

        records.append(
            {
                "country": entry["country"]["value"],
                "country_code": entry["countryiso3code"],
                "year": year,
                "gdp_per_capita_usd": float(value),
            }
        )

    long_df = pd.DataFrame(records).sort_values(["year", "country"]).reset_index(drop=True)
    wide_df = (
        long_df.pivot(index="year", columns="country", values="gdp_per_capita_usd")
        .sort_index()
        .round(2)
    )
    return long_df, wide_df


def main() -> None:
    # Task 1: web scraping without an API
    population_df = scrape_population_table()
    top10_population = population_df.nlargest(10, "population_2026")[
        ["country", "population_2026", "yearly_change_pct", "world_share_pct"]
    ].copy()
    top10_population["population_2026"] = top10_population["population_2026"].astype("Int64")

    print("\n=== Web Scraping Task (No API) ===")
    print(f"Source URL: {WEB_URL}")
    print(f"Rows scraped: {len(population_df)}")
    print("Top 10 countries by 2026 population:")
    print(top10_population.to_string(index=False))

    # Task 2: API collection
    gdp_long_df, gdp_wide_df = fetch_world_bank_api()
    recent_gdp = gdp_wide_df.loc[2019:2024]

    print("\n=== API Task (World Bank) ===")
    print("Source API endpoint:")
    print(f"{API_URL}?format=json&per_page=200")
    print(f"Rows collected (long format): {len(gdp_long_df)}")
    print("GDP per capita (current US$), 2019-2024:")
    print(recent_gdp.to_string())


if __name__ == "__main__":
    main()
