#!/usr/bin/env python3
"""Backtest models, select by MASE, and make a 24-month concentration forecast."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dhaka-aqi-matplotlib")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.forecasting import forecast_model, rolling_origin  # noqa: E402


def save(fig: plt.Figure, stem: str) -> None:
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    monthly = pd.read_csv(ROOT / "data/processed/primary_observed_monthly.csv")
    monthly = monthly[pd.to_datetime(monthly["month_start"]) <= pd.Timestamp("2025-02-01")].copy()
    # Three internal months with 66-71% day coverage are retained only to avoid
    # inventing values in a regular monthly model; this is explicit in the report.
    values = monthly["pm25_mean"].to_numpy(dtype=float)
    dates = pd.to_datetime(monthly["month_start"])
    cv = rolling_origin(values, dates, horizon=12)
    tables = ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cv.to_csv(tables / "model_cross_validation.csv", index=False)
    ranking = cv.groupby("model").agg(mae=("mae", "mean"), rmse=("rmse", "mean"), mase=("mase", "mean"), smape_pct=("smape_pct", "mean"), interval_coverage_pct=("interval_coverage_pct", "mean"), folds=("origin", "count")).reset_index().sort_values("mase")
    ranking.to_csv(tables / "model_ranking.csv", index=False)
    best_model = ranking.iloc[0]["model"]
    horizon = 24
    result = forecast_model(values, horizon, str(best_model))
    future_dates = pd.date_range(dates.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    summary = pd.DataFrame({"month_start": future_dates, "target": "pm25", "unit": "ug/m3", "model": best_model, "forecast": result.mean, "lower_95": result.lower, "upper_95": result.upper, "forecast_type": "empirical 24-month forecast"})
    summary.to_csv(tables / "forecast_summary.csv", index=False)

    annual_2024 = monthly[pd.to_datetime(monthly["month_start"]).dt.year == 2024]["pm25_mean"].mean()
    scenario = pd.DataFrame(
        [
            {"scenario": "No-additional-change benchmark", "year": 2030, "pm25_ug_m3": annual_2024, "lower_95": np.nan, "upper_95": np.nan, "type": "deterministic scenario", "assumption": "2024 observed annual mean held constant; not a forecast"},
            {"scenario": "Bangladesh annual-standard target", "year": 2030, "pm25_ug_m3": 35.0, "lower_95": np.nan, "upper_95": np.nan, "type": "policy target", "assumption": "linear benchmark path to Bangladesh 2022 annual PM2.5 standard"},
            {"scenario": "WHO AQG target", "year": 2030, "pm25_ug_m3": 5.0, "lower_95": np.nan, "upper_95": np.nan, "type": "health guideline target", "assumption": "linear benchmark path to WHO 2021 annual AQG"},
        ]
    )
    scenario.to_csv(tables / "scenario_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model, group in cv.groupby("model"):
        ax.scatter(group["origin"], group["mase"], label=model, s=28)
    ax.axhline(1, color="black", ls="--", lw=1)
    ax.tick_params(axis="x", rotation=30)
    ax.set(xlabel="Training endpoint", ylabel="MASE (12-month test horizon)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, "forecast_backtesting")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(dates, values, color="#35618f", label="Observed monthly mean")
    ax.plot(future_dates, result.mean, color="#b44b4b", label=f"{best_model} forecast")
    ax.fill_between(future_dates, result.lower, result.upper, color="#b44b4b", alpha=0.2, label="95% prediction interval")
    ax.set(xlabel="Month", ylabel="PM$_{2.5}$ (µg m$^{-3}$)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "forecast_intervals")

    years = np.arange(2024, 2031)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for _, row in scenario.iterrows():
        path = np.linspace(annual_2024, row["pm25_ug_m3"], len(years))
        ax.plot(years, path, marker="o", label=row["scenario"])
    ax.set(xlabel="Year", ylabel="Conditional annual PM$_{2.5}$ benchmark (µg m$^{-3}$)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "scenarios_2030")

    report = [
        "# Forecast backtesting",
        "",
        "Models were evaluated by expanding-window rolling origins with a 12-month test horizon.",
        "All complex models were compared with seasonal-naive, naive, and drift baselines.",
        "MASE uses the in-sample seasonal-naive scale. sMAPE is reported instead of MAPE.",
        "All statistical models are fit to log1p concentration, so forecasts and intervals are non-negative without clipping a negative-scale model.",
        "",
        f"Selected model: `{best_model}` (lowest mean cross-validated MASE).",
        f"Primary horizon: {horizon} months ({future_dates.min().date()} to {future_dates.max().date()}).",
        "",
        "Three internal months (2020-05, 2021-09, 2022-07) have 66-71% valid-day coverage. They remain explicit partial-month means in forecasting to preserve the regular calendar without interpolation. This is a material limitation; no value was filled.",
        "",
        "The 2030 lines are deterministic benchmarks, not empirical forecasts. Uncertainty intervals are intentionally blank for policy targets because no defensible implementation-probability model exists; inventing one would create false precision.",
        "",
        "```csv",
        ranking.to_csv(index=False).strip(),
        "```",
    ]
    (ROOT / "reports/forecast_backtesting.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
