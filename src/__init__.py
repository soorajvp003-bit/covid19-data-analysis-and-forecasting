"""Reusable modules for the COVID-19 data analysis project."""

from .analysis import (
    build_country_snapshot,
    build_global_trends,
    build_vaccination_summary,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    generate_country_insights,
    measure_case_reduction_after_vaccination,
)
from .forecasting import ForecastResult, run_forecasting_suite
from .preprocessing import CovidDataPreprocessor, DEFAULT_DATA_URL, TARGET_COUNTRIES

__all__ = [
    "CovidDataPreprocessor",
    "DEFAULT_DATA_URL",
    "ForecastResult",
    "TARGET_COUNTRIES",
    "build_country_snapshot",
    "build_global_trends",
    "build_vaccination_summary",
    "calculate_correlation_matrix",
    "calculate_covariance_matrix",
    "generate_country_insights",
    "measure_case_reduction_after_vaccination",
    "run_forecasting_suite",
]
