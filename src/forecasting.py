"""Leakage-safe rolling-origin models for positive monthly PM2.5."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class ForecastResult:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _interval(mean: np.ndarray, residuals: np.ndarray, horizon: int, log_scale: bool = False) -> ForecastResult:
    sigma = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    width = 1.96 * sigma * np.sqrt(np.arange(1, horizon + 1))
    if log_scale:
        return ForecastResult(np.maximum(0, np.expm1(mean)), np.maximum(0, np.expm1(mean - width)), np.maximum(0, np.expm1(mean + width)))
    return ForecastResult(np.maximum(0, mean), np.maximum(0, mean - width), np.maximum(0, mean + width))


def forecast_model(train: np.ndarray, horizon: int, model: str) -> ForecastResult:
    train = np.asarray(train, dtype=float)
    if np.any(train < 0):
        raise ValueError("Forecast target contains negative concentration")
    if model == "seasonal_naive":
        mean = np.resize(train[-12:], horizon)
        residuals = train[12:] - train[:-12]
        return _interval(mean, residuals, horizon)
    if model == "naive":
        mean = np.repeat(train[-1], horizon)
        return _interval(mean, np.diff(train), horizon)
    if model == "drift":
        slope = (train[-1] - train[0]) / max(1, len(train) - 1)
        mean = train[-1] + slope * np.arange(1, horizon + 1)
        residuals = train[1:] - (train[:-1] + slope)
        return _interval(mean, residuals, horizon)

    logged = np.log1p(train)
    if model == "ets":
        fit = ExponentialSmoothing(logged, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12, initialization_method="estimated").fit(optimized=True)
        mean = np.asarray(fit.forecast(horizon))
        return _interval(mean, np.asarray(fit.resid), horizon, log_scale=True)
    if model == "sarima":
        fit = SARIMAX(logged, order=(1, 0, 1), seasonal_order=(1, 1, 0, 12), trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
        prediction = fit.get_forecast(horizon)
        ci = np.asarray(prediction.conf_int(alpha=0.05))
        return ForecastResult(np.maximum(0, np.expm1(prediction.predicted_mean)), np.maximum(0, np.expm1(ci[:, 0])), np.maximum(0, np.expm1(ci[:, 1])))
    if model == "regression":
        time = np.arange(len(train), dtype=float)
        month = np.arange(len(train)) % 12
        dummies = pd.get_dummies(month, prefix="month", drop_first=True, dtype=float)
        design = sm.add_constant(pd.concat([pd.Series(time, name="time"), dummies], axis=1))
        fit = sm.OLS(logged, design).fit()
        future_time = np.arange(len(train), len(train) + horizon, dtype=float)
        future_month = np.arange(len(train), len(train) + horizon) % 12
        future_dummies = pd.get_dummies(future_month, prefix="month", dtype=float).reindex(columns=dummies.columns, fill_value=0)
        future_design = sm.add_constant(pd.concat([pd.Series(future_time, name="time"), future_dummies.reset_index(drop=True)], axis=1), has_constant="add")
        prediction = fit.get_prediction(future_design).summary_frame(alpha=0.05)
        return ForecastResult(np.maximum(0, np.expm1(prediction["mean"].to_numpy())), np.maximum(0, np.expm1(prediction["obs_ci_lower"].to_numpy())), np.maximum(0, np.expm1(prediction["obs_ci_upper"].to_numpy())))
    raise ValueError(f"Unknown model {model}")


def metrics(actual: np.ndarray, forecast: ForecastResult, train: np.ndarray) -> dict[str, float]:
    error = actual - forecast.mean
    scale = np.mean(np.abs(train[12:] - train[:-12]))
    denominator = (np.abs(actual) + np.abs(forecast.mean)) / 2
    smape = np.mean(np.divide(np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0)) * 100
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error**2))),
        "mase": float(np.mean(np.abs(error)) / scale),
        "smape_pct": float(smape),
        "interval_coverage_pct": float(np.mean((actual >= forecast.lower) & (actual <= forecast.upper)) * 100),
    }


def rolling_origin(values: np.ndarray, dates: pd.Series, horizon: int = 12) -> pd.DataFrame:
    models = ["seasonal_naive", "naive", "drift", "ets", "sarima", "regression"]
    rows = []
    initial = 36
    origins = list(range(initial, len(values) - horizon + 1, 12))
    for origin in origins:
        train = values[:origin]
        actual = values[origin : origin + horizon]
        for model in models:
            result = forecast_model(train, horizon, model)
            row = {
                "origin": pd.to_datetime(dates.iloc[origin - 1]).date().isoformat(),
                "test_start": pd.to_datetime(dates.iloc[origin]).date().isoformat(),
                "test_end": pd.to_datetime(dates.iloc[origin + horizon - 1]).date().isoformat(),
                "horizon_months": horizon,
                "model": model,
            }
            row.update(metrics(actual, result, train))
            rows.append(row)
    return pd.DataFrame(rows)

