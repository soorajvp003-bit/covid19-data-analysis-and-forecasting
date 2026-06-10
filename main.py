"""Project entry point for the full COVID-19 analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis import (
    build_country_snapshot,
    build_global_trends,
    build_vaccination_summary,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    generate_country_insights,
    measure_case_reduction_after_vaccination,
)
from src.forecasting import run_forecasting_suite
from src.preprocessing import CovidDataPreprocessor, DEFAULT_DATA_URL, TARGET_COUNTRIES
from src.visualization import (
    create_correlation_plotly_heatmap,
    create_country_comparison_plotly_figure,
    create_global_plotly_figure,
    create_vaccination_ranking_plotly_figure,
    plot_correlation_heatmap,
    plot_country_comparison_bars,
    plot_country_metric_lines,
    plot_forecast_actual_vs_predicted,
    plot_global_trends,
    plot_vaccination_progress,
    save_plotly_figure,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="COVID-19 Global Data Analysis and Trend Forecasting")
    parser.add_argument("--data-source", default=DEFAULT_DATA_URL, help="CSV path or URL for the source dataset.")
    parser.add_argument(
        "--countries",
        default=",".join(TARGET_COUNTRIES),
        help="Comma-separated list of countries for comparison and reporting.",
    )
    parser.add_argument(
        "--forecast-country",
        default="India",
        help="Country used for saved forecast visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the entire data pipeline and save portfolio-ready outputs."""
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    images_dir = project_root / "images"
    reports_dir = project_root / "reports"
    raw_data_path = data_dir / "raw" / "owid-covid-data.csv"
    processed_data_path = data_dir / "processed" / "covid19_cleaned.csv"

    for directory in [data_dir / "raw", data_dir / "processed", images_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    countries = [country.strip() for country in args.countries.split(",") if country.strip()]
    preprocessor = CovidDataPreprocessor(data_source=args.data_source)
    cleaned_df, summary = preprocessor.prepare_dataset(raw_data_path, processed_data_path)

    preprocessing_steps_path = reports_dir / "preprocessing_steps.csv"
    preprocessor.save_preprocessing_steps(preprocessing_steps_path)
    _save_preprocessing_summary(summary, reports_dir / "preprocessing_summary.md")

    global_df = build_global_trends(cleaned_df)
    snapshot_df = build_country_snapshot(cleaned_df, countries)
    vaccination_summary_df = build_vaccination_summary(cleaned_df, countries)
    vaccination_impact_df = measure_case_reduction_after_vaccination(cleaned_df, countries)
    correlation_df = calculate_correlation_matrix(cleaned_df, countries)
    covariance_df = calculate_covariance_matrix(cleaned_df, countries)

    global_df.to_csv(reports_dir / "global_trends.csv", index=False)
    snapshot_df.to_csv(reports_dir / "country_snapshot.csv", index=False)
    vaccination_summary_df.to_csv(reports_dir / "vaccination_summary.csv", index=False)
    vaccination_impact_df.to_csv(reports_dir / "vaccination_case_reduction.csv", index=False)
    correlation_df.to_csv(reports_dir / "correlation_matrix.csv")
    covariance_df.to_csv(reports_dir / "covariance_matrix.csv")

    plot_global_trends(global_df, images_dir / "global_trends.png")
    plot_country_comparison_bars(snapshot_df, images_dir / "country_comparison_bars.png")
    plot_country_metric_lines(
        cleaned_df,
        countries,
        metric="total_cases",
        title="Total Cases Over Time",
        output_path=images_dir / "country_total_cases.png",
    )
    plot_country_metric_lines(
        cleaned_df,
        countries,
        metric="new_cases_7day_avg",
        title="New Cases Over Time (7-day Average)",
        output_path=images_dir / "country_new_cases.png",
    )
    plot_country_metric_lines(
        cleaned_df,
        countries,
        metric="total_deaths",
        title="Total Deaths Over Time",
        output_path=images_dir / "country_total_deaths.png",
    )
    plot_vaccination_progress(cleaned_df, countries, images_dir / "vaccination_progress.png")
    plot_correlation_heatmap(correlation_df, images_dir / "correlation_heatmap.png")

    save_plotly_figure(create_global_plotly_figure(global_df), reports_dir / "interactive_global_dashboard.html")
    save_plotly_figure(
        create_country_comparison_plotly_figure(cleaned_df, countries),
        reports_dir / "interactive_country_comparison.html",
    )
    save_plotly_figure(
        create_vaccination_ranking_plotly_figure(vaccination_summary_df),
        reports_dir / "interactive_vaccination_ranking.html",
    )
    save_plotly_figure(
        create_correlation_plotly_heatmap(correlation_df),
        reports_dir / "interactive_correlation_heatmap.html",
    )

    metrics_tables: list[pd.DataFrame] = []
    future_tables: list[pd.DataFrame] = []
    forecast_countries = sorted(set(countries + [args.forecast_country]))
    forecast_visualizations_written = False

    for country in forecast_countries:
        forecast_suite = run_forecasting_suite(cleaned_df, country)
        if not forecast_suite["results"]:
            print(f"No forecasts generated for {country}: not enough historical data.")
            continue

        metrics_df = forecast_suite["metrics"].copy()
        future_df = forecast_suite["future_forecasts"].copy()

        if not metrics_df.empty:
            metrics_tables.append(metrics_df)
        if not future_df.empty:
            future_tables.append(future_df)

        if country == args.forecast_country:
            for result in forecast_suite["results"]:
                safe_model = (
                    result.model_name.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("=", "_")
                )
                plot_forecast_actual_vs_predicted(
                    result,
                    images_dir / f"{country.lower().replace(' ', '_')}_{result.target_column}_{safe_model}.png",
                )
            forecast_visualizations_written = True

    if metrics_tables:
        forecast_metrics_df = pd.concat(metrics_tables, ignore_index=True)
    else:
        forecast_metrics_df = pd.DataFrame(
            columns=["country", "target_column", "target_label", "model_name", "mae", "rmse", "r2_score"]
        )

    if future_tables:
        forecast_future_df = pd.concat(future_tables, ignore_index=True)
    else:
        forecast_future_df = pd.DataFrame(
            columns=["date", "predicted", "horizon_day", "horizon_label", "country", "target_column", "model_name"]
        )
    forecast_metrics_df.to_csv(reports_dir / "forecast_metrics.csv", index=False)
    forecast_future_df.to_csv(reports_dir / "forecast_future_predictions.csv", index=False)

    insights = generate_country_insights(snapshot_df, vaccination_summary_df, vaccination_impact_df)
    _save_insight_report(
        insights=insights,
        countries=countries,
        summary=summary,
        forecast_country=args.forecast_country,
        forecast_visualizations_written=forecast_visualizations_written,
        output_path=reports_dir / "project_insights.md",
    )

    print("COVID-19 analysis pipeline completed successfully.")
    print(f"Processed dataset saved to: {processed_data_path}")
    print(f"Reports saved to: {reports_dir}")
    print(f"Images saved to: {images_dir}")


def _save_preprocessing_summary(summary, output_path: Path) -> None:
    """Save preprocessing summary in markdown format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preprocessing Summary",
        "",
        f"- Raw rows loaded: {summary.raw_rows}",
        f"- Cleaned rows saved: {summary.cleaned_rows}",
        f"- Duplicate rows removed: {summary.duplicate_rows_removed}",
        f"- Selected columns: {', '.join(summary.selected_columns)}",
        "",
        "## Cleaning Notes",
    ]
    lines.extend([f"- {note}" for note in summary.notes])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _save_insight_report(
    insights: list[str],
    countries: list[str],
    summary,
    forecast_country: str,
    forecast_visualizations_written: bool,
    output_path: Path,
) -> None:
    """Save human-readable insights as a markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# COVID-19 Project Insights",
        "",
        "## Portfolio Highlights",
        f"- Countries analyzed: {', '.join(countries)}",
        f"- Forecast visualization focus country: {forecast_country}",
        f"- Forecast images generated: {'Yes' if forecast_visualizations_written else 'No'}",
        f"- Raw rows loaded: {summary.raw_rows}",
        f"- Cleaned rows available: {summary.cleaned_rows}",
        "",
        "## Key Insights",
    ]
    lines.extend([f"- {insight}" for insight in insights])
    lines.extend(
        [
            "",
            "## Public Health Recommendations",
            "- Pair vaccination strategy with per-capita monitoring because large countries can hide severe local outbreaks in absolute counts alone.",
            "- Use 7-day rolling averages to guide response decisions instead of raw daily spikes, which are often distorted by reporting delays.",
            "- Track both case reduction and death reduction after vaccination milestones to evaluate whether coverage is translating into lower health-system strain.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
