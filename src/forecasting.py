"""Forecasting models for COVID-19 case and death trends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class ForecastingDataError(ValueError):
    """Raised when a country time series does not have enough observations."""

TARGET_LABELS = {
    "new_cases_7day_avg": "New Cases (7-day avg)",
    "new_deaths_7day_avg": "New Deaths (7-day avg)",
}


@dataclass(slots=True)
class ForecastResult:
    """Container for one forecasting experiment."""

    model_name: str
    country: str
    target_column: str
    metrics: dict[str, float]
    test_frame: pd.DataFrame
    future_frame: pd.DataFrame


def prepare_country_target(
    df: pd.DataFrame,
    country: str,
    target_column: str,
    min_history: int = 120,
) -> pd.DataFrame:
    """Prepare a single country time series for forecasting."""
    country_df = (
        df[df["location"] == country][["date", target_column]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(country_df) < min_history:
        raise ForecastingDataError(
            f"Not enough history for {country} and {target_column}. "
            f"Expected at least {min_history} rows, found {len(country_df)}."
        )
    return country_df


def forecast_with_linear_regression(
    series_df: pd.DataFrame,
    country: str,
    target_column: str,
    horizons: Iterable[int] = (30, 60, 90),
) -> ForecastResult:
    """Forecast a trend with basic linear regression."""
    working_df = _prepare_time_index(series_df, target_column)
    train_df, test_df = _chronological_split(working_df)

    model = LinearRegression()
    model.fit(train_df[["day_index"]], train_df[target_column])

    test_predictions = np.clip(model.predict(test_df[["day_index"]]), a_min=0, a_max=None)
    future_frame = _make_future_frame(working_df, max(horizons))
    future_predictions = np.clip(model.predict(future_frame[["day_index"]]), a_min=0, a_max=None)

    return ForecastResult(
        model_name="Linear Regression",
        country=country,
        target_column=target_column,
        metrics=_evaluate(test_df[target_column], test_predictions),
        test_frame=_build_test_frame(test_df, target_column, test_predictions),
        future_frame=_build_future_output(future_frame["date"], future_predictions, horizons),
    )


def forecast_with_polynomial_regression(
    series_df: pd.DataFrame,
    country: str,
    target_column: str,
    horizons: Iterable[int] = (30, 60, 90),
    degree: int = 3,
) -> ForecastResult:
    """Forecast a trend with polynomial regression."""
    working_df = _prepare_time_index(series_df, target_column)
    train_df, test_df = _chronological_split(working_df)

    model = Pipeline(
        steps=[
            ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(train_df[["day_index"]], train_df[target_column])

    test_predictions = np.clip(model.predict(test_df[["day_index"]]), a_min=0, a_max=None)
    future_frame = _make_future_frame(working_df, max(horizons))
    future_predictions = np.clip(model.predict(future_frame[["day_index"]]), a_min=0, a_max=None)

    return ForecastResult(
        model_name=f"Polynomial Regression (degree={degree})",
        country=country,
        target_column=target_column,
        metrics=_evaluate(test_df[target_column], test_predictions),
        test_frame=_build_test_frame(test_df, target_column, test_predictions),
        future_frame=_build_future_output(future_frame["date"], future_predictions, horizons),
    )


def forecast_with_random_forest_time_series(
    series_df: pd.DataFrame,
    country: str,
    target_column: str,
    horizons: Iterable[int] = (30, 60, 90),
) -> ForecastResult:
    """Forecast a time series using lag-based Random Forest features."""
    feature_df, feature_columns = _build_lag_features(series_df, target_column)
    train_df, test_df = _chronological_split(feature_df)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[feature_columns], train_df[target_column])

    test_predictions = np.clip(model.predict(test_df[feature_columns]), a_min=0, a_max=None)
    future_predictions = _recursive_random_forest_forecast(
        model=model,
        history_df=series_df,
        feature_columns=feature_columns,
        target_column=target_column,
        periods=max(horizons),
    )

    return ForecastResult(
        model_name="Time-Series Random Forest",
        country=country,
        target_column=target_column,
        metrics=_evaluate(test_df[target_column], test_predictions),
        test_frame=_build_test_frame(test_df, target_column, test_predictions),
        future_frame=_build_future_output(
            future_predictions["date"], future_predictions["prediction"], horizons
        ),
    )


def run_forecasting_suite(
    df: pd.DataFrame,
    country: str,
    target_columns: Iterable[str] = ("new_cases_7day_avg", "new_deaths_7day_avg"),
    horizons: Iterable[int] = (30, 60, 90),
) -> dict[str, pd.DataFrame | list[ForecastResult]]:
    """Run all forecasting models for the requested country and targets."""
    results: list[ForecastResult] = []

    for target_column in target_columns:
        try:
            series_df = prepare_country_target(df, country, target_column)
        except ForecastingDataError:
            continue
        results.append(forecast_with_linear_regression(series_df, country, target_column, horizons))
        results.append(forecast_with_polynomial_regression(series_df, country, target_column, horizons))
        results.append(forecast_with_random_forest_time_series(series_df, country, target_column, horizons))

    if not results:
        return {
            "results": [],
            "metrics": pd.DataFrame(
                columns=[
                    "country",
                    "target_column",
                    "target_label",
                    "model_name",
                    "mae",
                    "rmse",
                    "r2_score",
                ]
            ),
            "future_forecasts": pd.DataFrame(
                columns=[
                    "date",
                    "predicted",
                    "horizon_day",
                    "horizon_label",
                    "country",
                    "target_column",
                    "model_name",
                ]
            ),
        }

    metrics_records: list[dict[str, float | str]] = []
    future_records: list[dict[str, float | str | int | pd.Timestamp]] = []

    for result in results:
        metrics_records.append(
            {
                "country": result.country,
                "target_column": result.target_column,
                "target_label": TARGET_LABELS.get(result.target_column, result.target_column),
                "model_name": result.model_name,
                "mae": result.metrics["mae"],
                "rmse": result.metrics["rmse"],
                "r2_score": result.metrics["r2_score"],
            }
        )

        future_frame = result.future_frame.copy()
        future_frame["country"] = result.country
        future_frame["target_column"] = result.target_column
        future_frame["model_name"] = result.model_name
        future_records.extend(future_frame.to_dict(orient="records"))

    return {
        "results": results,
        "metrics": pd.DataFrame(metrics_records).sort_values(["target_column", "rmse"]),
        "future_forecasts": pd.DataFrame(future_records).sort_values(
            ["target_column", "model_name", "date"]
        ),
    }


def _prepare_time_index(series_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Create a day index for regression-based models."""
    working_df = series_df.copy()
    working_df["day_index"] = (working_df["date"] - working_df["date"].min()).dt.days
    working_df = working_df.dropna(subset=[target_column]).reset_index(drop=True)
    return working_df


def _build_lag_features(
    series_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Create lag-based features for a machine-learning time-series model."""
    feature_df = series_df.copy()
    lags = [1, 7, 14]
    for lag in lags:
        feature_df[f"lag_{lag}"] = feature_df[target_column].shift(lag)

    shifted_target = feature_df[target_column].shift(1)
    feature_df["rolling_mean_7"] = shifted_target.rolling(window=7).mean()
    feature_df["rolling_mean_14"] = shifted_target.rolling(window=14).mean()
    feature_df["rolling_std_7"] = shifted_target.rolling(window=7).std().fillna(0)
    feature_df["day_of_week"] = feature_df["date"].dt.dayofweek
    feature_df["month"] = feature_df["date"].dt.month
    feature_df["day_of_year"] = feature_df["date"].dt.dayofyear
    feature_df["day_index"] = (feature_df["date"] - feature_df["date"].min()).dt.days

    feature_df = feature_df.dropna().reset_index(drop=True)
    feature_columns = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
        "day_of_week",
        "month",
        "day_of_year",
        "day_index",
    ]
    return feature_df, feature_columns


def _recursive_random_forest_forecast(
    model: RandomForestRegressor,
    history_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    periods: int,
) -> pd.DataFrame:
    """Generate future predictions by feeding each prediction back into history."""
    history = history_df[["date", target_column]].copy().reset_index(drop=True)
    predictions: list[dict[str, float | pd.Timestamp]] = []

    for _ in range(periods):
        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        target_history = history[target_column]

        feature_row = pd.DataFrame(
            [
                {
                    "lag_1": float(target_history.iloc[-1]),
                    "lag_7": float(target_history.iloc[-7])
                    if len(target_history) >= 7
                    else float(target_history.mean()),
                    "lag_14": float(target_history.iloc[-14])
                    if len(target_history) >= 14
                    else float(target_history.mean()),
                    "rolling_mean_7": float(target_history.tail(7).mean()),
                    "rolling_mean_14": float(target_history.tail(14).mean()),
                    "rolling_std_7": float(target_history.tail(7).std(ddof=0)),
                    "day_of_week": next_date.dayofweek,
                    "month": next_date.month,
                    "day_of_year": next_date.dayofyear,
                    "day_index": int((next_date - history["date"].iloc[0]).days),
                }
            ],
            columns=feature_columns,
        )

        prediction = float(np.clip(model.predict(feature_row)[0], a_min=0, a_max=None))
        predictions.append({"date": next_date, "prediction": prediction})
        history.loc[len(history)] = {"date": next_date, target_column: prediction}

    return pd.DataFrame(predictions)


def _chronological_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    min_test_rows: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series into train and test sets without shuffling."""
    test_size = max(int(len(df) * test_fraction), min_test_rows)
    if test_size >= len(df):
        test_size = max(1, len(df) // 3)

    split_index = len(df) - test_size
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def _make_future_frame(working_df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Create future dates and matching day indices."""
    start_date = working_df["date"].iloc[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=periods, freq="D")
    day_index_start = int(working_df["day_index"].iloc[-1]) + 1
    return pd.DataFrame(
        {"date": future_dates, "day_index": range(day_index_start, day_index_start + periods)}
    )


def _build_test_frame(
    test_df: pd.DataFrame,
    target_column: str,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Create a common actual-vs-predicted table."""
    output_df = test_df[["date", target_column]].copy()
    output_df = output_df.rename(columns={target_column: "actual"})
    output_df["predicted"] = predictions
    return output_df


def _build_future_output(
    dates: pd.Series,
    predictions: np.ndarray | pd.Series,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Attach 30/60/90-day markers to a future forecast table."""
    future_df = pd.DataFrame({"date": dates, "predicted": predictions})
    future_df["horizon_day"] = np.arange(1, len(future_df) + 1)
    selected_horizons = set(horizons)
    future_df["horizon_label"] = future_df["horizon_day"].apply(
        lambda horizon_day: f"{horizon_day}-day"
        if horizon_day in selected_horizons
        else "intermediate"
    )
    return future_df


def _evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute forecasting metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_score": float(r2_score(y_true, y_pred)),
    }
