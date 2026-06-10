"""Streamlit dashboard for interactive COVID-19 analysis and forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis import (  # noqa: E402
    build_country_snapshot,
    build_global_trends,
    build_vaccination_summary,
    generate_country_insights,
    measure_case_reduction_after_vaccination,
)
from src.forecasting import TARGET_LABELS, run_forecasting_suite  # noqa: E402
from src.preprocessing import CovidDataPreprocessor, DEFAULT_DATA_URL, TARGET_COUNTRIES  # noqa: E402
from src.visualization import (  # noqa: E402
    create_country_comparison_plotly_figure,
    create_global_plotly_figure,
    create_vaccination_ranking_plotly_figure,
)

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "covid19_cleaned.csv"
RAW_CSV_FALLBACK_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "owid-covid-data.csv",
    PROJECT_ROOT / "data" / "raw" / "owid-covid-latest.csv",
]


def build_dataset_profile(df: pd.DataFrame, source_label: str) -> dict[str, object]:
    """Summarize whether the dataset supports historical trend analysis."""
    working_df = df[df["is_country"]].copy() if "is_country" in df.columns else df.copy()
    if working_df.empty:
        working_df = df.copy()

    if working_df.empty or "location" not in working_df.columns or "date" not in working_df.columns:
        return {
            "source_label": source_label,
            "row_count": len(df),
            "location_count": 0,
            "unique_dates": 0,
            "max_dates_per_location": 0,
            "is_snapshot": True,
            "date_min": None,
            "date_max": None,
        }

    dates_per_location = working_df.groupby("location")["date"].nunique()
    return {
        "source_label": source_label,
        "row_count": len(df),
        "location_count": int(working_df["location"].nunique()),
        "unique_dates": int(working_df["date"].nunique()),
        "max_dates_per_location": int(dates_per_location.max()),
        "is_snapshot": bool(dates_per_location.max() <= 1),
        "date_min": working_df["date"].min(),
        "date_max": working_df["date"].max(),
    }


def dataset_has_history(df: pd.DataFrame) -> bool:
    """Return True when the frame contains more than one dated observation."""
    return "date" in df.columns and df["date"].nunique() > 1


def build_dataset_status_message(profile: dict[str, object]) -> str:
    """Explain whether the dashboard is using historical or snapshot data."""
    source_label = str(profile["source_label"])
    date_min = profile["date_min"]
    date_max = profile["date_max"]
    if isinstance(date_min, pd.Timestamp) and isinstance(date_max, pd.Timestamp):
        date_label = (
            date_min.strftime("%b %d, %Y")
            if date_min == date_max
            else f"{date_min.strftime('%b %d, %Y')} to {date_max.strftime('%b %d, %Y')}"
        )
    else:
        date_label = "unknown date range"

    if bool(profile["is_snapshot"]):
        return (
            "Snapshot mode: the current dataset has one dated row per country "
            f"from `{source_label}` ({date_label}), so trend and forecast views are "
            "replaced with latest-value comparisons."
        )

    return (
        "Historical mode: loaded "
        f"{profile['unique_dates']} dates across {profile['location_count']} countries "
        f"from `{source_label}` ({date_label})."
    )


def load_prepared_dataset(source: str | Path, persist_processed_output: bool = True) -> pd.DataFrame:
    """Prepare a dataset from a URL or CSV path."""
    preprocessor = CovidDataPreprocessor(data_source=source)
    cleaned_df, _ = preprocessor.prepare_dataset(
        processed_output_path=PROCESSED_DATA_PATH if persist_processed_output else None
    )
    return cleaned_df


@st.cache_data(show_spinner=False)
def load_dashboard_data(data_source: str = DEFAULT_DATA_URL) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the best available dataset, preferring full history over a latest snapshot."""
    snapshot_candidates: list[tuple[pd.DataFrame, dict[str, object]]] = []
    load_errors: list[Exception] = []

    if PROCESSED_DATA_PATH.exists():
        processed_df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["date"])
        processed_profile = build_dataset_profile(processed_df, str(PROCESSED_DATA_PATH))
        if not processed_profile["is_snapshot"]:
            return processed_df, processed_profile
        snapshot_candidates.append((processed_df, processed_profile))

    seen_sources = {str(PROCESSED_DATA_PATH)}
    candidate_sources: list[str | Path] = [data_source]
    candidate_sources.extend(path for path in RAW_CSV_FALLBACK_PATHS if path.exists())

    for source in candidate_sources:
        source_label = str(source)
        if source_label in seen_sources:
            continue
        seen_sources.add(source_label)

        try:
            cleaned_df = load_prepared_dataset(source)
        except Exception as exc:
            load_errors.append(exc)
            continue

        profile = build_dataset_profile(cleaned_df, source_label)
        if not profile["is_snapshot"]:
            return cleaned_df, profile
        snapshot_candidates.append((cleaned_df, profile))

    if snapshot_candidates:
        best_snapshot_df, best_snapshot_profile = max(
            snapshot_candidates,
            key=lambda item: (
                int(item[1]["max_dates_per_location"]),
                int(item[1]["unique_dates"]),
                int(item[1]["row_count"]),
            ),
        )
        return best_snapshot_df, best_snapshot_profile

    if load_errors:
        raise load_errors[0]
    raise RuntimeError("No readable COVID-19 dataset was found for the dashboard.")


@st.cache_data(show_spinner=False)
def get_forecast_bundle(df: pd.DataFrame, country: str) -> dict[str, pd.DataFrame | list]:
    """Run forecasting for the selected country."""
    return run_forecasting_suite(df, country)


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="COVID-19 Global Data Analysis",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.title("COVID-19 Global Data Analysis and Trend Forecasting")
    st.caption(
        "Interactive portfolio dashboard using Our World in Data, Pandas, Plotly, "
        "Scikit-learn, and Streamlit."
    )

    data_source = st.sidebar.text_input("Dataset URL or local CSV path", value=DEFAULT_DATA_URL)
    try:
        cleaned_df, data_profile = load_dashboard_data(data_source)
    except Exception as exc:
        st.error(
            "The dashboard could not load the dataset. Install the requirements and ensure "
            "the machine can access the OWID data source or provide a local CSV path."
        )
        st.exception(exc)
        st.stop()

    st.caption(build_dataset_status_message(data_profile))
    if bool(data_profile["is_snapshot"]):
        st.warning(
            "This run is using a latest-only snapshot dataset, so the dashboard switches "
            "to comparison views instead of pretending it has a full timeline."
        )

    country_options = sorted(cleaned_df.loc[cleaned_df["is_country"], "location"].dropna().unique())
    default_country = "India" if "India" in country_options else country_options[0]
    selected_country = st.sidebar.selectbox("Country selector", options=country_options, index=country_options.index(default_country))
    comparison_countries = st.sidebar.multiselect(
        "Comparison countries",
        options=country_options,
        default=[country for country in TARGET_COUNTRIES if country in country_options],
    )
    if not comparison_countries:
        comparison_countries = [selected_country]

    forecast_target = st.sidebar.selectbox(
        "Forecast target",
        options=list(TARGET_LABELS.keys()),
        format_func=lambda key: TARGET_LABELS[key],
    )
    forecast_horizon = st.sidebar.selectbox("Forecast horizon", options=[30, 60, 90], index=0)

    selected_country_df = (
        cleaned_df[cleaned_df["location"] == selected_country].sort_values("date").copy()
    )
    selected_country_has_history = dataset_has_history(selected_country_df)
    latest_snapshot = build_country_snapshot(cleaned_df, [selected_country]).iloc[0]
    vaccination_summary_df = build_vaccination_summary(cleaned_df, comparison_countries)
    vaccination_impact_df = measure_case_reduction_after_vaccination(cleaned_df, comparison_countries)
    comparison_snapshot_df = build_country_snapshot(cleaned_df, comparison_countries)
    global_trends_df = build_global_trends(cleaned_df)
    global_has_history = dataset_has_history(global_trends_df)

    st.subheader(f"Latest Snapshot: {selected_country}")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Cases", f"{latest_snapshot['total_cases']:,.0f}")
    metric_columns[1].metric("Total Deaths", f"{latest_snapshot['total_deaths']:,.0f}")
    metric_columns[2].metric("Death Rate", f"{latest_snapshot['death_rate']:.2f}%")
    metric_columns[3].metric("Fully Vaccinated", f"{latest_snapshot['fully_vaccinated_share']:.2f}%")

    tab_cases, tab_deaths, tab_vaccination, tab_comparison, tab_forecast = st.tabs(
        ["Cases", "Deaths", "Vaccination", "Comparison", "Forecasting"]
    )

    with tab_cases:
        if selected_country_has_history:
            case_fig = px.line(
                selected_country_df,
                x="date",
                y=["total_cases", "new_cases_7day_avg"],
                title=f"{selected_country}: Total Cases and New Cases (7-day Avg)",
                labels={"value": "Count", "date": "Date", "variable": "Metric"},
            )
            st.plotly_chart(case_fig, width="stretch")
        else:
            st.info(
                f"{selected_country} has only one dated record in the current dataset, so this tab shows latest case comparisons instead of a timeline."
            )
            case_snapshot_fig = px.bar(
                comparison_snapshot_df.sort_values("total_cases", ascending=False),
                x="location",
                y=["total_cases", "cases_per_million"],
                barmode="group",
                title="Latest Case Snapshot Across Selected Countries",
                labels={"value": "Latest Value", "variable": "Metric", "location": "Country"},
            )
            st.plotly_chart(case_snapshot_fig, width="stretch")

        global_fig = create_global_plotly_figure(global_trends_df)
        if not global_has_history:
            st.info("Global trends are also in snapshot mode because the dataset contains only one date.")
        st.plotly_chart(global_fig, width="stretch")

    with tab_deaths:
        if selected_country_has_history:
            death_fig = px.line(
                selected_country_df,
                x="date",
                y=["total_deaths", "new_deaths_7day_avg"],
                title=f"{selected_country}: Total Deaths and New Deaths (7-day Avg)",
                labels={"value": "Count", "date": "Date", "variable": "Metric"},
            )
            st.plotly_chart(death_fig, width="stretch")
        else:
            death_snapshot_fig = px.bar(
                comparison_snapshot_df.sort_values("total_deaths", ascending=False),
                x="location",
                y=["total_deaths", "death_rate"],
                barmode="group",
                title="Latest Death Snapshot Across Selected Countries",
                labels={"value": "Latest Value", "variable": "Metric", "location": "Country"},
            )
            st.plotly_chart(death_snapshot_fig, width="stretch")

    with tab_vaccination:
        if selected_country_has_history:
            vaccination_fig = px.line(
                selected_country_df,
                x="date",
                y=["vaccinated_share", "fully_vaccinated_share", "booster_share"],
                title=f"{selected_country}: Vaccination Progress Curves",
                labels={"value": "Population Share (%)", "date": "Date", "variable": "Metric"},
            )
            st.plotly_chart(vaccination_fig, width="stretch")
        else:
            vaccination_snapshot_fig = px.bar(
                vaccination_summary_df.sort_values("fully_vaccinated_share", ascending=False),
                x="location",
                y=["vaccinated_share", "fully_vaccinated_share", "booster_share"],
                barmode="group",
                title="Latest Vaccination Snapshot Across Selected Countries",
                labels={"value": "Population Share (%)", "variable": "Metric", "location": "Country"},
            )
            st.plotly_chart(vaccination_snapshot_fig, width="stretch")
        st.plotly_chart(
            create_vaccination_ranking_plotly_figure(vaccination_summary_df),
            width="stretch",
        )

    with tab_comparison:
        st.plotly_chart(
            create_country_comparison_plotly_figure(cleaned_df, comparison_countries),
            width="stretch",
        )

        heatmap_df = comparison_snapshot_df[
            [
                "location",
                "total_cases",
                "total_deaths",
                "death_rate",
                "cases_per_million",
                "fully_vaccinated_share",
            ]
        ].set_index("location")
        comparison_heatmap = px.imshow(
            heatmap_df,
            text_auto=".2f",
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Country-wise Comparison Heatmap",
        )
        st.plotly_chart(comparison_heatmap, width="stretch")
        st.dataframe(comparison_snapshot_df, width="stretch")

    with tab_forecast:
        if not selected_country_has_history:
            st.warning(
                "Forecasting is unavailable in snapshot mode because the dataset has only one dated row for the selected country."
            )
            st.info(
                "Use a historical OWID CSV or keep the default online source available so the dashboard can build a real time series."
            )
        else:
            forecast_bundle = get_forecast_bundle(cleaned_df, selected_country)
            metrics_df = forecast_bundle["metrics"]
            future_df = forecast_bundle["future_forecasts"]
            results = forecast_bundle["results"]

            if not results or metrics_df.empty:
                st.warning(
                    "Forecasting is unavailable for this dataset or country because there is not enough historical daily data. "
                    "Use a full OWID time-series dataset to enable model forecasts."
                )
                st.info(
                    "Forecasting requires at least 120 historical rows for each country target. "
                    "This dashboard will still show descriptive analytics and comparisons."
                )
            else:
                target_metrics_df = metrics_df[metrics_df["target_column"] == forecast_target].reset_index(drop=True)
                if target_metrics_df.empty:
                    st.warning(
                        f"Forecast metrics are not available for {TARGET_LABELS[forecast_target]}. "
                        "Try a different target or a dataset with longer history."
                    )
                else:
                    st.write("Model evaluation using MAE, RMSE, and R2 score:")
                    st.dataframe(target_metrics_df, width="stretch")

                    best_model_row = target_metrics_df.sort_values("rmse").iloc[0]
                    best_result = next(
                        result
                        for result in results
                        if result.target_column == forecast_target and result.model_name == best_model_row["model_name"]
                    )

                    actual_vs_predicted_fig = px.line(
                        best_result.test_frame,
                        x="date",
                        y=["actual", "predicted"],
                        title=f"{selected_country}: Actual vs Predicted {TARGET_LABELS[forecast_target]} ({best_result.model_name})",
                        labels={"value": "Count", "date": "Date", "variable": "Series"},
                    )
                    st.plotly_chart(actual_vs_predicted_fig, width="stretch")

                    filtered_future_df = future_df[
                        (future_df["target_column"] == forecast_target)
                        & (future_df["model_name"] == best_model_row["model_name"])
                        & (future_df["horizon_day"] <= forecast_horizon)
                    ].copy()

                    forecast_fig = px.line(
                        filtered_future_df,
                        x="date",
                        y="predicted",
                        title=f"{selected_country}: {forecast_horizon}-Day Forecast for {TARGET_LABELS[forecast_target]}",
                        labels={"predicted": "Predicted Count", "date": "Date"},
                    )
                    st.plotly_chart(forecast_fig, width="stretch")
                    st.dataframe(filtered_future_df, width="stretch")

    insights = generate_country_insights(
        comparison_snapshot_df,
        vaccination_summary_df,
        vaccination_impact_df,
    )
    st.subheader("Business Insights")
    for insight in insights:
        st.write(f"- {insight}")

    report_text = build_download_report(
        selected_country=selected_country,
        latest_snapshot=latest_snapshot,
        comparison_snapshot_df=comparison_snapshot_df,
        vaccination_summary_df=vaccination_summary_df,
        vaccination_impact_df=vaccination_impact_df,
        insights=insights,
    )
    st.download_button(
        label="Download report",
        data=report_text,
        file_name=f"{selected_country.lower().replace(' ', '_')}_covid19_report.md",
        mime="text/markdown",
    )

    country_csv = selected_country_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download selected country data",
        data=country_csv,
        file_name=f"{selected_country.lower().replace(' ', '_')}_covid19_data.csv",
        mime="text/csv",
    )


def build_download_report(
    selected_country: str,
    latest_snapshot: pd.Series,
    comparison_snapshot_df: pd.DataFrame,
    vaccination_summary_df: pd.DataFrame,
    vaccination_impact_df: pd.DataFrame,
    insights: list[str],
) -> str:
    """Create a downloadable markdown report."""
    lines = [
        f"# COVID-19 Dashboard Report: {selected_country}",
        "",
        "## Latest Metrics",
        f"- Total cases: {latest_snapshot['total_cases']:,.0f}",
        f"- Total deaths: {latest_snapshot['total_deaths']:,.0f}",
        f"- Death rate: {latest_snapshot['death_rate']:.2f}%",
        f"- Fully vaccinated share: {latest_snapshot['fully_vaccinated_share']:.2f}%",
        "",
        "## Comparison Table",
        comparison_snapshot_df.to_markdown(index=False),
        "",
        "## Vaccination Summary",
        vaccination_summary_df.to_markdown(index=False),
        "",
        "## Vaccination Impact",
        vaccination_impact_df.to_markdown(index=False),
        "",
        "## Insights",
    ]
    lines.extend([f"- {insight}" for insight in insights])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
