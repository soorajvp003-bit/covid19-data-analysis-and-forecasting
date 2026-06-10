"""Static and interactive visualizations for the project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from .forecasting import ForecastResult, TARGET_LABELS

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_global_trends(global_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot total and daily global case/death metrics."""
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
        axis.ticklabel_format(style="plain", axis="y")

    fig.tight_layout()
    destination = _save_matplotlib_figure(fig, output_path)
    plt.close(fig)
    return destination


def plot_country_comparison_bars(snapshot_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Create bar charts for the main country comparison metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    chart_specs = [
        ("total_cases", "Total Cases"),
        ("total_deaths", "Total Deaths"),
        ("cases_per_million", "Cases per Million"),
        ("death_rate", "Death Rate (%)"),
    ]

    for axis, (metric, title) in zip(axes.flat, chart_specs):
        sns.barplot(data=snapshot_df.sort_values(metric, ascending=False), x="location", y=metric, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Country")
        axis.set_ylabel(title)
        axis.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    destination = _save_matplotlib_figure(fig, output_path)
    plt.close(fig)
    return destination


def plot_country_metric_lines(
    df: pd.DataFrame,
    countries: list[str],
    metric: str,
    title: str,
    output_path: str | Path,
) -> Path:
    """Plot a selected metric over time for multiple countries."""
    filtered_df = df[df["location"].isin(countries)].copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=filtered_df, x="date", y=metric, hue="location", ax=ax, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(title)
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()

    destination = _save_matplotlib_figure(fig, output_path)
    plt.close(fig)
    return destination


def plot_correlation_heatmap(correlation_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Create a correlation heatmap using Seaborn."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("COVID-19 Correlation Heatmap")
    fig.tight_layout()

    destination = _save_matplotlib_figure(fig, output_path)
    plt.close(fig)
    return destination


def plot_vaccination_progress(df: pd.DataFrame, countries: list[str], output_path: str | Path) -> Path:
    """Plot vaccination progress curves for selected countries."""
    filtered_df = df[df["location"].isin(countries)].copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(
        data=filtered_df,
        x="date",
        y="fully_vaccinated_share",
        hue="location",
        ax=ax,
        linewidth=2,
    )
    ax.set_title("Fully Vaccinated Population Share Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fully Vaccinated Share (%)")
    fig.tight_layout()

    destination = _save_matplotlib_figure(fig, output_path)
    plt.close(fig)
    return destination


def plot_forecast_actual_vs_predicted(result: ForecastResult, output_path: str | Path) -> Path:
    """Plot actual vs predicted values for a forecasting model."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result.test_frame["date"], result.test_frame["actual"], label="Actual", linewidth=2)
    ax.plot(
        result.test_frame["date"],
        result.test_frame["predicted"],
        label="Predicted",
        linewidth=2,
        linestyle="--",
    )
    ax.set_title(
        f"{result.country} - {TARGET_LABELS.get(result.target_column, result.target_column)} - {result.model_name}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    destination = _save_matplotlib_figure(fig, output_path)
    plt.close(fig)
    return destination


def create_global_plotly_figure(global_df: pd.DataFrame) -> go.Figure:
    """Build an interactive four-panel global trend dashboard."""
    if "date" in global_df.columns and global_df["date"].nunique() <= 1:
        snapshot_df = global_df.tail(1).melt(
            id_vars="date",
            value_vars=[
                "total_cases",
                "new_cases_7day_avg",
                "total_deaths",
                "new_deaths_7day_avg",
            ],
            var_name="metric",
            value_name="value",
        )
        snapshot_df["metric"] = snapshot_df["metric"].map(
            {
                "total_cases": "Total Cases",
                "new_cases_7day_avg": "New Cases (7-day Avg)",
                "total_deaths": "Total Deaths",
                "new_deaths_7day_avg": "New Deaths (7-day Avg)",
            }
        )
        fig = px.bar(
            snapshot_df,
            x="metric",
            y="value",
            color="metric",
            title="Global COVID-19 Snapshot",
            labels={"metric": "Metric", "value": "Latest Value"},
        )
        fig.update_layout(showlegend=False)
        return fig

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Total Cases",
            "New Cases (7-day Avg)",
            "Total Deaths",
            "New Deaths (7-day Avg)",
        ),
    )

    specs = [
        ("total_cases", 1, 1),
        ("new_cases_7day_avg", 1, 2),
        ("total_deaths", 2, 1),
        ("new_deaths_7day_avg", 2, 2),
    ]

    for column, row, col in specs:
        fig.add_trace(
            go.Scatter(x=global_df["date"], y=global_df[column], mode="lines", name=column),
            row=row,
            col=col,
        )

    fig.update_layout(height=800, width=1100, title_text="Global COVID-19 Trend Dashboard")
    return fig


def create_country_comparison_plotly_figure(df: pd.DataFrame, countries: list[str]) -> go.Figure:
    """Build an interactive comparison figure for selected countries."""
    filtered_df = df[df["location"].isin(countries)].copy()
    if "date" in filtered_df.columns and filtered_df["date"].nunique() <= 1:
        fig = px.bar(
            filtered_df.sort_values("new_cases_7day_avg", ascending=False),
            x="location",
            y="new_cases_7day_avg",
            color="location",
            title="Latest Country-wise New Cases (7-day Avg)",
            labels={"new_cases_7day_avg": "New Cases (7-day Avg)", "location": "Country"},
        )
        fig.update_layout(showlegend=False)
        return fig

    fig = px.line(
        filtered_df,
        x="date",
        y="new_cases_7day_avg",
        color="location",
        title="Interactive Country-wise New Case Trend",
        labels={"new_cases_7day_avg": "New Cases (7-day Avg)", "date": "Date", "location": "Country"},
    )
    fig.update_layout(hovermode="x unified")
    return fig


def create_vaccination_ranking_plotly_figure(vaccination_df: pd.DataFrame) -> go.Figure:
    """Create an interactive vaccination ranking chart."""
    fig = px.bar(
        vaccination_df.sort_values("fully_vaccinated_share", ascending=False),
        x="location",
        y="fully_vaccinated_share",
        color="booster_share",
        title="Country Ranking by Fully Vaccinated Population Share",
        labels={"fully_vaccinated_share": "Fully Vaccinated Share (%)", "location": "Country"},
    )
    fig.update_layout(coloraxis_colorbar_title="Booster Share (%)")
    return fig


def create_correlation_plotly_heatmap(correlation_df: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation heatmap."""
    return px.imshow(
        correlation_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Interactive Correlation Heatmap",
        aspect="auto",
    )


def save_plotly_figure(fig: go.Figure, output_path: str | Path) -> Path:
    """Persist a Plotly figure to HTML."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(destination)
    return destination


def _save_matplotlib_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    """Save a Matplotlib figure and return the file path."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    return destination
