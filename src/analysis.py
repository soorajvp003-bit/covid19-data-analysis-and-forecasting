"""Exploratory analysis, country comparisons, and statistical summaries."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .preprocessing import TARGET_COUNTRIES, filter_countries

SNAPSHOT_COLUMNS = [
    "location",
    "date",
    "population",
    "total_cases",
    "total_deaths",
    "death_rate",
    "estimated_recoveries",
    "recovery_rate_proxy",
    "cases_per_million",
    "deaths_per_million",
    "total_vaccinations",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
    "vaccinated_share",
    "fully_vaccinated_share",
    "booster_share",
]


def build_global_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Create a global time series for cases, deaths, and vaccinations."""
    world_rows = df[df["location"] == "World"].copy()
    columns = [
        "date",
        "total_cases",
        "new_cases",
        "total_deaths",
        "new_deaths",
        "new_cases_7day_avg",
        "new_deaths_7day_avg",
        "total_vaccinations",
        "new_vaccinations_7day_avg",
    ]

    if not world_rows.empty:
        return world_rows[columns].sort_values("date").reset_index(drop=True)

    country_df = df[df["is_country"]].copy()
    global_df = (
        country_df.groupby("date", as_index=False)[
            [
                "total_cases",
                "new_cases",
                "total_deaths",
                "new_deaths",
                "total_vaccinations",
                "new_vaccinations",
            ]
        ]
        .sum()
        .sort_values("date")
    )
    global_df["new_cases_7day_avg"] = global_df["new_cases"].rolling(window=7, min_periods=1).mean()
    global_df["new_deaths_7day_avg"] = global_df["new_deaths"].rolling(window=7, min_periods=1).mean()
    global_df["new_vaccinations_7day_avg"] = global_df["new_vaccinations"].rolling(window=7, min_periods=1).mean()
    return global_df[columns]


def build_country_snapshot(df: pd.DataFrame, countries: Iterable[str] | None = None) -> pd.DataFrame:
    """Return the latest available metrics for selected countries."""
    selected_countries = list(countries or TARGET_COUNTRIES)
    country_df = filter_countries(df, selected_countries).sort_values(["location", "date"])
    snapshot_df = country_df.groupby("location", as_index=False).tail(1).reset_index(drop=True)
    available_columns = [column for column in SNAPSHOT_COLUMNS if column in snapshot_df.columns]
    return snapshot_df[available_columns].sort_values("total_cases", ascending=False)


def build_vaccination_summary(df: pd.DataFrame, countries: Iterable[str] | None = None) -> pd.DataFrame:
    """Summarize vaccination progress and speed across countries."""
    selected_countries = list(countries or TARGET_COUNTRIES)
    country_df = filter_countries(df, selected_countries).sort_values(["location", "date"])
    latest_df = build_country_snapshot(country_df, selected_countries)

    first_vaccination = (
        country_df[country_df["people_vaccinated"] > 0]
        .groupby("location", as_index=False)["date"]
        .min()
        .rename(columns={"date": "first_vaccination_date"})
    )

    fifty_percent = (
        country_df[country_df["vaccinated_share"] >= 50]
        .groupby("location", as_index=False)["date"]
        .min()
        .rename(columns={"date": "date_reached_50pct_vaccinated"})
    )

    peak_vaccinations = (
        country_df.groupby("location", as_index=False)["new_vaccinations_7day_avg"]
        .max()
        .rename(columns={"new_vaccinations_7day_avg": "peak_daily_vaccinations_7day_avg"})
    )

    summary_df = latest_df.merge(first_vaccination, on="location", how="left")
    summary_df = summary_df.merge(fifty_percent, on="location", how="left")
    summary_df = summary_df.merge(peak_vaccinations, on="location", how="left")

    summary_df["days_to_50pct_vaccinated"] = (
        summary_df["date_reached_50pct_vaccinated"] - summary_df["first_vaccination_date"]
    ).dt.days

    return summary_df.sort_values(
        ["vaccinated_share", "fully_vaccinated_share", "booster_share"],
        ascending=False,
    ).reset_index(drop=True)


def measure_case_reduction_after_vaccination(
    df: pd.DataFrame,
    countries: Iterable[str] | None = None,
    threshold: float = 50.0,
) -> pd.DataFrame:
    """Measure whether case and death trends changed after strong vaccine coverage."""
    selected_countries = list(countries or TARGET_COUNTRIES)
    country_df = filter_countries(df, selected_countries).sort_values(["location", "date"])
    records: list[dict[str, float | str | pd.Timestamp]] = []

    for country in selected_countries:
        country_series = country_df[country_df["location"] == country].copy()
        if country_series.empty:
            continue

        threshold_rows = country_series[country_series["fully_vaccinated_share"] >= threshold]
        if threshold_rows.empty:
            threshold_rows = country_series[country_series["vaccinated_share"] >= threshold]
        if threshold_rows.empty:
            records.append(
                {
                    "location": country,
                    "milestone_date": pd.NaT,
                    "avg_cases_before": np.nan,
                    "avg_cases_after": np.nan,
                    "case_change_pct": np.nan,
                    "avg_deaths_before": np.nan,
                    "avg_deaths_after": np.nan,
                    "death_change_pct": np.nan,
                }
            )
            continue

        milestone_date = threshold_rows["date"].iloc[0]
        before_window = country_series[
            (country_series["date"] >= milestone_date - pd.Timedelta(days=30))
            & (country_series["date"] < milestone_date)
        ]
        after_window = country_series[
            (country_series["date"] > milestone_date)
            & (country_series["date"] <= milestone_date + pd.Timedelta(days=30))
        ]

        avg_cases_before = before_window["new_cases_7day_avg"].mean()
        avg_cases_after = after_window["new_cases_7day_avg"].mean()
        avg_deaths_before = before_window["new_deaths_7day_avg"].mean()
        avg_deaths_after = after_window["new_deaths_7day_avg"].mean()

        records.append(
            {
                "location": country,
                "milestone_date": milestone_date,
                "avg_cases_before": avg_cases_before,
                "avg_cases_after": avg_cases_after,
                "case_change_pct": _percentage_change(avg_cases_before, avg_cases_after),
                "avg_deaths_before": avg_deaths_before,
                "avg_deaths_after": avg_deaths_after,
                "death_change_pct": _percentage_change(avg_deaths_before, avg_deaths_after),
            }
        )

    return pd.DataFrame(records).sort_values("case_change_pct")


def calculate_correlation_matrix(
    df: pd.DataFrame,
    countries: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate the correlation matrix for important numerical indicators."""
    selected_countries = list(countries or TARGET_COUNTRIES)
    country_df = filter_countries(df, selected_countries)

    numeric_columns = [
        "total_cases",
        "new_cases_7day_avg",
        "total_deaths",
        "new_deaths_7day_avg",
        "death_rate",
        "cases_per_million",
        "deaths_per_million",
        "vaccinated_share",
        "fully_vaccinated_share",
        "booster_share",
        "recovery_rate_proxy",
    ]
    available_columns = [column for column in numeric_columns if column in country_df.columns]
    return country_df[available_columns].corr(numeric_only=True)


def calculate_covariance_matrix(
    df: pd.DataFrame,
    countries: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate covariance for the same indicators used in the heatmap."""
    selected_countries = list(countries or TARGET_COUNTRIES)
    country_df = filter_countries(df, selected_countries)

    numeric_columns = [
        "total_cases",
        "new_cases_7day_avg",
        "total_deaths",
        "new_deaths_7day_avg",
        "vaccinated_share",
        "fully_vaccinated_share",
        "booster_share",
        "recovery_rate_proxy",
    ]
    available_columns = [column for column in numeric_columns if column in country_df.columns]
    return country_df[available_columns].cov(numeric_only=True)


def generate_country_insights(
    snapshot_df: pd.DataFrame,
    vaccination_summary_df: pd.DataFrame,
    vaccination_impact_df: pd.DataFrame,
) -> list[str]:
    """Generate business-style insights for README files and reports."""
    insights: list[str] = []

    if not snapshot_df.empty:
        top_cases = snapshot_df.nlargest(1, "total_cases").iloc[0]
        top_deaths = snapshot_df.nlargest(1, "total_deaths").iloc[0]
        top_cases_per_million = snapshot_df.nlargest(1, "cases_per_million").iloc[0]
        highest_death_rate = snapshot_df.nlargest(1, "death_rate").iloc[0]

        insights.append(
            f"{top_cases['location']} has the highest total case count among the selected countries, showing the largest absolute pandemic burden."
        )
        insights.append(
            f"{top_deaths['location']} records the highest cumulative deaths, making it a key country for mortality trend comparison."
        )
        insights.append(
            f"{top_cases_per_million['location']} leads on cases per million, which highlights a high infection load even after adjusting for population size."
        )
        insights.append(
            f"{highest_death_rate['location']} shows the highest case fatality ratio in the comparison set, so mortality management deserves closer inspection."
        )

    if not vaccination_summary_df.empty:
        fastest_vaccinated = vaccination_summary_df.dropna(subset=["days_to_50pct_vaccinated"])
        if not fastest_vaccinated.empty:
            leader = fastest_vaccinated.nsmallest(1, "days_to_50pct_vaccinated").iloc[0]
            insights.append(
                f"{leader['location']} reached 50% vaccination coverage fastest, indicating a comparatively strong rollout pace."
            )

        top_fully_vaccinated = vaccination_summary_df.nlargest(1, "fully_vaccinated_share").iloc[0]
        insights.append(
            f"{top_fully_vaccinated['location']} currently has the strongest full-vaccination coverage within the selected portfolio countries."
        )

    if not vaccination_impact_df.empty:
        best_case_reduction = vaccination_impact_df.dropna(subset=["case_change_pct"])
        if not best_case_reduction.empty:
            best_case_country = best_case_reduction.nsmallest(1, "case_change_pct").iloc[0]
            insights.append(
                f"After crossing the vaccination milestone, {best_case_country['location']} shows the largest short-term decline in average cases in this comparison."
            )

        best_death_reduction = vaccination_impact_df.dropna(subset=["death_change_pct"])
        if not best_death_reduction.empty:
            best_death_country = best_death_reduction.nsmallest(1, "death_change_pct").iloc[0]
            insights.append(
                f"{best_death_country['location']} shows the strongest post-vaccination reduction in average deaths, suggesting a meaningful mortality benefit."
            )

    insights.append(
        "Public-health decisions should combine vaccination coverage with per-capita case trends because absolute counts alone can hide population-adjusted risk."
    )
    return insights


def _percentage_change(before: float, after: float) -> float:
    """Compute percentage change safely."""
    if pd.isna(before) or before == 0:
        return np.nan
    return ((after - before) / before) * 100
