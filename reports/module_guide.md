# Module Guide

## `src/preprocessing.py`

This module handles dataset loading, cleaning, missing value handling, date conversion, duplicate removal, and feature engineering.

- `CovidDataPreprocessor.load_raw_data()` reads the OWID CSV from a URL or local file.
- `CovidDataPreprocessor.clean_data()` selects useful columns, formats dates, removes duplicate daily records, fills cumulative values, and creates transformed features.
- Derived features include rolling averages, growth rates, cases per million, deaths per million, vaccination percentages, death rate, and an estimated recovery trend proxy.
- `PreprocessingSummary` stores the most important cleaning statistics so they can be reused in reports and the README.

## `src/analysis.py`

This module performs exploratory analysis and generates comparison-ready tables.

- `build_global_trends()` creates the world-level case, death, and vaccination trend series.
- `build_country_snapshot()` extracts the latest metrics for the selected countries.
- `build_vaccination_summary()` compares vaccination speed and current coverage.
- `measure_case_reduction_after_vaccination()` compares case and death averages before and after a vaccination milestone.
- `calculate_correlation_matrix()` and `calculate_covariance_matrix()` support advanced statistical analysis and heatmaps.
- `generate_country_insights()` converts tables into business-style portfolio conclusions.

## `src/visualization.py`

This module creates both static and interactive charts.

- Matplotlib and Seaborn functions generate exportable PNG figures for the README, reports, and presentations.
- Plotly functions create interactive HTML dashboards for global trends, country comparisons, vaccination ranking, and correlation heatmaps.
- `plot_forecast_actual_vs_predicted()` saves model evaluation charts for the forecasting section.

## `src/forecasting.py`

This module contains the predictive models.

- `forecast_with_linear_regression()` provides a simple baseline trend model.
- `forecast_with_polynomial_regression()` captures curved growth and decline patterns more flexibly.
- `forecast_with_random_forest_time_series()` uses lag features and rolling statistics for a stronger time-series style model.
- `run_forecasting_suite()` runs all three models for cases and deaths, then returns evaluation metrics plus 30, 60, and 90 day forecasts.
- Metrics include MAE, RMSE, and R2 score.

## `dashboard/app.py`

This Streamlit application turns the project into an interactive portfolio dashboard.

- Users can select a country and comparison set.
- The dashboard displays case, death, vaccination, and comparison charts.
- It runs forecasting dynamically for the chosen country.
- It includes download buttons for a markdown report and country-level CSV data.

## `main.py`

This is the end-to-end pipeline runner.

- Downloads or loads the dataset
- Cleans and transforms the data
- Saves reports and CSV summaries
- Generates charts and interactive HTML files
- Runs forecasting models
- Creates markdown insight reports for portfolio usage
