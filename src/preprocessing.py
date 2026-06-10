"""Data loading, cleaning, and feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
TARGET_COUNTRIES = ["India", "United States", "China", "Brazil", "United Kingdom"]

SELECTED_COLUMNS = [
    "iso_code",
    "continent",
    "location",
    "date",
    "population",
    "total_cases",
    "new_cases",
    "total_deaths",
    "new_deaths",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
    "total_vaccinations",
    "new_vaccinations",
]

CUMULATIVE_COLUMNS = [
    "total_cases",
    "total_deaths",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
    "total_vaccinations",
]

DAILY_COLUMNS = ["new_cases", "new_deaths", "new_vaccinations"]


@dataclass(slots=True)
class PreprocessingSummary:
    """Tracks the main cleaning operations applied to the dataset."""

    raw_rows: int
    cleaned_rows: int
    duplicate_rows_removed: int
    selected_columns: list[str]
    notes: list[str]


class CovidDataPreprocessor:
    """Loads the OWID dataset and prepares it for analysis."""

    def __init__(self, data_source: str | Path = DEFAULT_DATA_URL) -> None:
        self.data_source = data_source

    @staticmethod
    def preprocessing_steps() -> list[dict[str, str]]:
        """Return a student-friendly explanation of each preprocessing step."""
        return [
            {
                "step": "Dataset loading",
                "description": (
                    "Read the CSV from a local file or the official Our World in Data URL "
                    "into a Pandas DataFrame."
                ),
            },
            {
                "step": "Feature selection",
                "description": (
                    "Keep only the columns needed for cases, deaths, vaccination, "
                    "population, and country/date identifiers to reduce noise."
                ),
            },
            {
                "step": "Date formatting",
                "description": (
                    "Convert the `date` column to a datetime format so trend plots, "
                    "rolling averages, and forecasting models work correctly."
                ),
            },
            {
                "step": "Duplicate removal",
                "description": (
                    "Remove duplicate `location` + `date` records to ensure one daily "
                    "observation per region."
                ),
            },
            {
                "step": "Missing value handling",
                "description": (
                    "Forward-fill cumulative metrics within each country and replace "
                    "missing daily metrics with 0 where appropriate."
                ),
            },
            {
                "step": "Data transformation",
                "description": (
                    "Create rolling averages, growth rates, per-million indicators, "
                    "death rate, vaccination shares, and recovery trend proxies."
                ),
            },
        ]

    def load_raw_data(self, data_source: str | Path | None = None) -> pd.DataFrame:
        """Load the raw CSV from disk or a URL."""
        source = data_source or self.data_source

        try:
            return pd.read_csv(source, low_memory=False)
        except Exception as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError(
                "Unable to load the COVID-19 dataset from the configured source. "
                "If you are offline, provide a local CSV path with `--data-source <path>` "
                "or place a local OWID file at data/raw/owid-covid-data.csv."
            ) from exc

    def prepare_dataset(
        self,
        raw_output_path: str | Path | None = None,
        processed_output_path: str | Path | None = None,
    ) -> tuple[pd.DataFrame, PreprocessingSummary]:
        """Run the full data preparation pipeline."""
        try:
            raw_df = self.load_raw_data()
        except RuntimeError as exc:
            if raw_output_path is not None and Path(raw_output_path).exists():
                raw_df = pd.read_csv(Path(raw_output_path), low_memory=False)
            else:
                raise exc

        if raw_output_path is not None:
            raw_path = Path(raw_output_path)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(raw_path, index=False)

        cleaned_df, summary = self.clean_data(raw_df)

        if processed_output_path is not None:
            processed_path = Path(processed_output_path)
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_df.to_csv(processed_path, index=False)

        return cleaned_df, summary

    def clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingSummary]:
        """Clean the dataset and engineer project-ready features."""
        working_df = df.copy()
        raw_rows = len(working_df)

        if "date" not in working_df.columns and "last_updated_date" in working_df.columns:
            working_df = working_df.rename(columns={"last_updated_date": "date"})

        available_columns = [column for column in SELECTED_COLUMNS if column in working_df.columns]
        working_df = working_df[available_columns]

        if "date" not in working_df.columns or "location" not in working_df.columns:
            raise ValueError("The source data must include `location` and `date` columns.")

        working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")
        working_df = working_df.dropna(subset=["date", "location"])

        numeric_columns = [
            column
            for column in available_columns
            if column not in {"iso_code", "continent", "location", "date"}
        ]
        for column in numeric_columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

        working_df = working_df.sort_values(["location", "date"]).reset_index(drop=True)

        duplicate_rows_removed = int(working_df.duplicated(subset=["location", "date"]).sum())
        working_df = working_df.drop_duplicates(subset=["location", "date"], keep="last")

        if "iso_code" in working_df.columns:
            working_df["is_country"] = ~working_df["iso_code"].fillna("").str.startswith("OWID_")
        else:
            working_df["is_country"] = True

        if "continent" not in working_df.columns:
            working_df["continent"] = "Unknown"
        working_df["continent"] = working_df["continent"].fillna("Unknown")

        grouped = working_df.groupby("location", group_keys=False)

        for column in CUMULATIVE_COLUMNS:
            if column in working_df.columns:
                working_df[column] = grouped[column].ffill().fillna(0)

        for column in DAILY_COLUMNS:
            if column in working_df.columns:
                working_df[column] = working_df[column].fillna(0)

        if "population" in working_df.columns:
            working_df["population"] = grouped["population"].transform(
                lambda series: series.ffill().bfill()
            )
        else:
            working_df["population"] = np.nan

        working_df["new_cases_7day_avg"] = grouped["new_cases"].transform(
            lambda series: series.rolling(window=7, min_periods=1).mean()
        )
        working_df["new_deaths_7day_avg"] = grouped["new_deaths"].transform(
            lambda series: series.rolling(window=7, min_periods=1).mean()
        )
        working_df["new_vaccinations_7day_avg"] = grouped["new_vaccinations"].transform(
            lambda series: series.rolling(window=7, min_periods=1).mean()
        )

        working_df["cases_14day_total"] = grouped["new_cases"].transform(
            lambda series: series.rolling(window=14, min_periods=1).sum()
        )
        working_df["deaths_14day_total"] = grouped["new_deaths"].transform(
            lambda series: series.rolling(window=14, min_periods=1).sum()
        )

        working_df["case_growth_rate"] = grouped["new_cases_7day_avg"].pct_change() * 100
        working_df["death_growth_rate"] = grouped["new_deaths_7day_avg"].pct_change() * 100
        working_df["vaccination_growth_rate"] = grouped["new_vaccinations_7day_avg"].pct_change() * 100

        working_df["cases_per_million"] = self._ratio(
            working_df["total_cases"], working_df["population"], 1_000_000
        )
        working_df["deaths_per_million"] = self._ratio(
            working_df["total_deaths"], working_df["population"], 1_000_000
        )
        working_df["vaccinations_per_hundred"] = self._ratio(
            working_df["total_vaccinations"], working_df["population"], 100
        )
        working_df["vaccinated_share"] = self._ratio(
            working_df["people_vaccinated"], working_df["population"], 100
        )
        working_df["fully_vaccinated_share"] = self._ratio(
            working_df["people_fully_vaccinated"], working_df["population"], 100
        )
        working_df["booster_share"] = self._ratio(
            working_df["total_boosters"], working_df["population"], 100
        )
        working_df["death_rate"] = self._ratio(
            working_df["total_deaths"], working_df["total_cases"], 100
        )

        working_df["estimated_active_cases"] = working_df["cases_14day_total"].clip(lower=0)
        working_df["estimated_recoveries"] = (
            working_df["total_cases"] - working_df["total_deaths"] - working_df["estimated_active_cases"]
        ).clip(lower=0)
        working_df["recovery_rate_proxy"] = self._ratio(
            working_df["estimated_recoveries"], working_df["total_cases"], 100
        )

        working_df = working_df.replace([np.inf, -np.inf], np.nan)
        working_df = working_df.fillna(
            {
                "case_growth_rate": 0,
                "death_growth_rate": 0,
                "vaccination_growth_rate": 0,
                "cases_per_million": 0,
                "deaths_per_million": 0,
                "vaccinations_per_hundred": 0,
                "vaccinated_share": 0,
                "fully_vaccinated_share": 0,
                "booster_share": 0,
                "death_rate": 0,
                "recovery_rate_proxy": 0,
            }
        )

        notes = [
            "Cumulative columns were forward-filled inside each country to preserve monotonic totals.",
            "Daily case, death, and vaccination gaps were treated as zero because missing daily values usually mean no new report for that day.",
            "Recovery trends are represented with an estimated recovery proxy because OWID does not publish official recovery counts in the main file.",
            "Rows such as World, continents, and income groups are retained but tagged with `is_country = False` for flexible filtering.",
        ]

        summary = PreprocessingSummary(
            raw_rows=raw_rows,
            cleaned_rows=len(working_df),
            duplicate_rows_removed=duplicate_rows_removed,
            selected_columns=available_columns,
            notes=notes,
        )
        return working_df, summary

    @staticmethod
    def save_preprocessing_steps(path: str | Path) -> None:
        """Export preprocessing notes for the project report."""
        step_df = pd.DataFrame(CovidDataPreprocessor.preprocessing_steps())
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        step_df.to_csv(output_path, index=False)

    @staticmethod
    def _ratio(numerator: pd.Series, denominator: pd.Series, scale: float) -> pd.Series:
        """Safely divide one series by another."""
        valid = denominator.gt(0)
        output = pd.Series(np.zeros(len(numerator)), index=numerator.index, dtype=float)
        output.loc[valid] = (numerator.loc[valid] / denominator.loc[valid]) * scale
        return output


def filter_countries(df: pd.DataFrame, countries: Iterable[str]) -> pd.DataFrame:
    """Convenience function for country-based filtering."""
    return df[df["location"].isin(list(countries))].copy()
