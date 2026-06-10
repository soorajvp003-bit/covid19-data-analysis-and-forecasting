"""Generate the project Jupyter notebook programmatically."""

from __future__ import annotations

import json
from pathlib import Path


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


def build_notebook() -> dict:
    return {
        "cells": [
            markdown_cell(
                """
                # COVID-19 Global Data Analysis and Trend Forecasting

                This notebook demonstrates the full workflow for loading the Our World in Data COVID-19 dataset, preprocessing it, exploring trends, comparing countries, analyzing vaccination progress, and forecasting future case and death patterns.
                """
            ),
            markdown_cell(
                """
                ## Objectives

                - Load and clean public COVID-19 data
                - Study global trends for cases and deaths
                - Compare India, the United States, China, Brazil, and the United Kingdom
                - Analyze vaccination progress and post-vaccination trend changes
                - Run Linear Regression, Polynomial Regression, and Time-Series Random Forest forecasts
                """
            ),
            code_cell(
                """
                from pathlib import Path
                import sys

                PROJECT_ROOT = Path.cwd().resolve().parent
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.append(str(PROJECT_ROOT))

                import pandas as pd
                import seaborn as sns
                import matplotlib.pyplot as plt
                from IPython.display import display

                from src.preprocessing import CovidDataPreprocessor, TARGET_COUNTRIES
                from src.analysis import (
                    build_country_snapshot,
                    build_global_trends,
                    build_vaccination_summary,
                    calculate_correlation_matrix,
                    measure_case_reduction_after_vaccination,
                    generate_country_insights,
                )
                from src.forecasting import run_forecasting_suite

                sns.set_theme(style="whitegrid")
                plt.rcParams["figure.figsize"] = (12, 6)
                """
            ),
            markdown_cell(
                """
                ## 1. Data Collection and Preprocessing

                The preprocessing pipeline:

                1. Loads the OWID CSV
                2. Selects relevant features
                3. Converts dates to datetime
                4. Removes duplicate country-date rows
                5. Fills cumulative gaps with forward fill
                6. Fills missing daily values with 0
                7. Creates rolling averages, growth rates, vaccination shares, death rate, and recovery proxies
                """
            ),
            code_cell(
                """
                preprocessor = CovidDataPreprocessor()
                cleaned_df, summary = preprocessor.prepare_dataset(
                    processed_output_path=PROJECT_ROOT / "data" / "processed" / "covid19_cleaned.csv"
                )

                display(pd.DataFrame(preprocessor.preprocessing_steps()))
                summary
                """
            ),
            code_cell(
                """
                cleaned_df.head()
                """
            ),
            markdown_cell(
                """
                ## 2. Global Exploratory Data Analysis
                """
            ),
            code_cell(
                """
                global_df = build_global_trends(cleaned_df)
                global_df.tail()
                """
            ),
            code_cell(
                """
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                metrics = [
                    ("total_cases", "Global Total Cases"),
                    ("new_cases_7day_avg", "Global New Cases (7-day Avg)"),
                    ("total_deaths", "Global Total Deaths"),
                    ("new_deaths_7day_avg", "Global New Deaths (7-day Avg)"),
                ]

                for axis, (column, title) in zip(axes.flat, metrics):
                    axis.plot(global_df["date"], global_df[column], linewidth=2)
                    axis.set_title(title)
                    axis.set_xlabel("Date")
                    axis.set_ylabel(title)

                plt.tight_layout()
                plt.show()
                """
            ),
            markdown_cell(
                """
                ## 3. Country-wise Comparisons
                """
            ),
            code_cell(
                """
                snapshot_df = build_country_snapshot(cleaned_df, TARGET_COUNTRIES)
                vaccination_summary_df = build_vaccination_summary(cleaned_df, TARGET_COUNTRIES)
                vaccination_impact_df = measure_case_reduction_after_vaccination(cleaned_df, TARGET_COUNTRIES)

                display(snapshot_df)
                display(vaccination_summary_df)
                display(vaccination_impact_df)
                """
            ),
            code_cell(
                """
                country_df = cleaned_df[cleaned_df["location"].isin(TARGET_COUNTRIES)].copy()

                plt.figure(figsize=(14, 7))
                sns.lineplot(data=country_df, x="date", y="new_cases_7day_avg", hue="location", linewidth=2)
                plt.title("Country-wise New Cases (7-day Avg)")
                plt.show()
                """
            ),
            markdown_cell(
                """
                ## 4. Vaccination Analysis
                """
            ),
            code_cell(
                """
                plt.figure(figsize=(14, 7))
                sns.lineplot(data=country_df, x="date", y="fully_vaccinated_share", hue="location", linewidth=2)
                plt.title("Fully Vaccinated Share by Country")
                plt.ylabel("Population Share (%)")
                plt.show()
                """
            ),
            markdown_cell(
                """
                ## 5. Advanced Statistical Analysis
                """
            ),
            code_cell(
                """
                correlation_df = calculate_correlation_matrix(cleaned_df, TARGET_COUNTRIES)
                correlation_df
                """
            ),
            code_cell(
                """
                plt.figure(figsize=(12, 8))
                sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f")
                plt.title("Correlation Heatmap")
                plt.show()
                """
            ),
            markdown_cell(
                """
                ## 6. Trend Forecasting

                The notebook forecasts daily smoothed cases and deaths, which is usually more stable than forecasting noisy raw daily values or cumulative totals.
                """
            ),
            code_cell(
                """
                forecast_bundle = run_forecasting_suite(cleaned_df, "India")
                metrics_df = forecast_bundle["metrics"]
                display(metrics_df)
                """
            ),
            code_cell(
                """
                best_case_model = (
                    metrics_df[metrics_df["target_column"] == "new_cases_7day_avg"]
                    .sort_values("rmse")
                    .iloc[0]["model_name"]
                )
                best_result = next(
                    result for result in forecast_bundle["results"]
                    if result.target_column == "new_cases_7day_avg" and result.model_name == best_case_model
                )

                plt.figure(figsize=(14, 7))
                plt.plot(best_result.test_frame["date"], best_result.test_frame["actual"], label="Actual")
                plt.plot(best_result.test_frame["date"], best_result.test_frame["predicted"], label="Predicted", linestyle="--")
                plt.title(f"India - Actual vs Predicted New Cases (7-day Avg) - {best_case_model}")
                plt.legend()
                plt.show()
                """
            ),
            code_cell(
                """
                insights = generate_country_insights(snapshot_df, vaccination_summary_df, vaccination_impact_df)
                for insight in insights:
                    print("-", insight)
                """
            ),
            markdown_cell(
                """
                ## 7. Dashboard

                Run the Streamlit app from the project root:

                ```bash
                streamlit run dashboard/app.py
                ```
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "covid19_global_analysis.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(f"Notebook generated at: {notebook_path}")


if __name__ == "__main__":
    main()
