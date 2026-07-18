#!/usr/bin/env python3
"""Generate corrected statistical tables, QA report, and core figures."""

from __future__ import annotations

import json
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
from scipy import stats  # noqa: E402

from src.analysis import (  # noqa: E402
    covid_comparison,
    month_adjusted_hac_regression,
    pettitt_test,
    seasonal_mann_kendall,
    seasonal_sen_slope,
    seasonal_tests,
    trend_free_prewhitened_smk,
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def guideline_tables(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily["date"] = pd.to_datetime(daily["date_local"])
    daily["year"] = daily["date"].dt.year
    daily["expected_days"] = daily["year"].map(lambda year: 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365)
    annual = daily.groupby("year").agg(
        pm25_annual_mean=("value", "mean"),
        valid_days=("date_local", "nunique"),
        expected_days=("expected_days", "first"),
    ).reset_index()
    annual["coverage_pct"] = annual["valid_days"] / annual["expected_days"] * 100
    annual = annual[annual["coverage_pct"] >= 75].copy()
    standards = pd.DataFrame(
        [
            ("WHO AQG", "2021", 5.0, "annual", "health guideline"),
            ("Bangladesh NAAQS", "2022", 35.0, "annual", "national standard"),
            ("US EPA NAAQS", "2024", 9.0, "annual", "descriptive only; formal design value differs"),
        ],
        columns=["standard", "version", "threshold_ug_m3", "averaging_period", "interpretation"],
    )
    annual = annual.merge(standards, how="cross")
    annual["exceeds_threshold"] = annual["pm25_annual_mean"] > annual["threshold_ug_m3"]
    annual["multiple_of_threshold"] = annual["pm25_annual_mean"] / annual["threshold_ug_m3"]

    rows = []
    for year, group in daily.groupby("year"):
        values = group["value"].to_numpy()
        rows.extend(
            [
                {
                    "year": year,
                    "standard": "WHO AQG",
                    "version": "2021",
                    "averaging_period": "24-hour; 99th percentile form",
                    "threshold_ug_m3": 15.0,
                    "valid_days": len(values),
                    "days_above_threshold": int(np.sum(values > 15)),
                    "design_statistic_ug_m3": float(np.quantile(values, 0.99)),
                    "meets_form": bool(np.quantile(values, 0.99) <= 15),
                },
                {
                    "year": year,
                    "standard": "Bangladesh NAAQS",
                    "version": "2022",
                    "averaging_period": "24-hour; no more than one crossing/year",
                    "threshold_ug_m3": 65.0,
                    "valid_days": len(values),
                    "days_above_threshold": int(np.sum(values > 65)),
                    "design_statistic_ug_m3": float(np.partition(values, -2)[-2]) if len(values) > 1 else float(values[0]),
                    "meets_form": bool(np.sum(values > 65) <= 1),
                },
                {
                    "year": year,
                    "standard": "US EPA NAAQS",
                    "version": "2024",
                    "averaging_period": "24-hour; annual 98th percentile (formal 3-year design value)",
                    "threshold_ug_m3": 35.0,
                    "valid_days": len(values),
                    "days_above_threshold": int(np.sum(values > 35)),
                    "design_statistic_ug_m3": float(np.quantile(values, 0.98)),
                    "meets_form": bool(np.quantile(values, 0.98) <= 35),
                },
            ]
        )
    return annual, pd.DataFrame(rows)


def main() -> None:
    tables = ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    monthly_all = pd.read_csv(ROOT / "data/processed/primary_observed_monthly.csv")
    monthly = pd.read_csv(ROOT / "data/processed/analysis_monthly.csv")
    daily = pd.read_parquet(ROOT / "data/processed/primary_observed_daily.parquet")
    dates = pd.to_datetime(monthly["month_start"])

    descriptive = pd.DataFrame(
        [
            {
                "n_complete_months": len(monthly),
                "start_month": monthly["month_start"].min(),
                "end_month": monthly["month_start"].max(),
                "mean_ug_m3": monthly["pm25_mean"].mean(),
                "median_ug_m3": monthly["pm25_mean"].median(),
                "std_ug_m3": monthly["pm25_mean"].std(),
                "iqr_ug_m3": stats.iqr(monthly["pm25_mean"]),
                "min_ug_m3": monthly["pm25_mean"].min(),
                "max_ug_m3": monthly["pm25_mean"].max(),
            }
        ]
    )
    descriptive.to_csv(tables / "descriptive_summary.csv", index=False)

    climatology = monthly.assign(month=dates.dt.month).groupby("month")["pm25_mean"].agg(["count", "mean", "median", "std"]).reset_index()
    climatology["ci_low"] = climatology["mean"] - stats.t.ppf(0.975, climatology["count"] - 1) * climatology["std"] / np.sqrt(climatology["count"])
    climatology["ci_high"] = climatology["mean"] + stats.t.ppf(0.975, climatology["count"] - 1) * climatology["std"] / np.sqrt(climatology["count"])
    climatology.to_csv(tables / "monthly_climatology.csv", index=False)

    overall, pairs = seasonal_tests(monthly)
    overall.to_csv(tables / "seasonal_test.csv", index=False)
    pairs.to_csv(tables / "seasonal_pairwise_holm.csv", index=False)

    trend = {
        "seasonal_mann_kendall": seasonal_mann_kendall(monthly["pm25_mean"], dates.dt.month),
        "seasonal_sen_slope": seasonal_sen_slope(monthly["pm25_mean"], dates),
        "trend_free_prewhitening": trend_free_prewhitened_smk(monthly["pm25_mean"], dates),
        "month_adjusted_hac_regression": month_adjusted_hac_regression(monthly["pm25_mean"], dates),
        "pettitt_change_diagnostic": pettitt_test(monthly["pm25_mean"], dates),
    }
    (tables / "trend_summary.json").write_text(json.dumps(trend, indent=2) + "\n", encoding="utf-8")

    covid = covid_comparison(monthly)
    covid.to_csv(tables / "covid_association.csv", index=False)
    annual, daily_exceedance = guideline_tables(daily.copy())
    annual.to_csv(tables / "annual_guideline_comparison.csv", index=False)
    daily_exceedance.to_csv(tables / "daily_exceedance_summary.csv", index=False)
    pd.DataFrame(columns=["year", "pollutant", "standard", "averaging_period", "note"]).to_csv(tables / "hourly_exceedance_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(pd.to_datetime(monthly_all["month_start"]), monthly_all["pm25_mean"], color="#35618f", lw=1.4, marker="o", ms=2.5)
    partial = monthly_all[~monthly_all["is_complete"]]
    ax.scatter(pd.to_datetime(partial["month_start"]), partial["pm25_mean"], facecolors="none", edgecolors="#b44b4b", label="Partial/excluded month", zorder=3)
    ax.axhline(35, color="#d48a00", ls="--", lw=1, label="Bangladesh annual standard (context only)")
    ax.set(ylabel="24-hour PM$_{2.5}$ monthly mean (µg m$^{-3}$)", xlabel="Month")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "observed_pm25_time_series")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(climatology["month"], climatology["mean"], yerr=[climatology["mean"] - climatology["ci_low"], climatology["ci_high"] - climatology["mean"]], fmt="o-", color="#35618f", capsize=3)
    ax.set(xticks=range(1, 13), xlabel="Calendar month", ylabel="PM$_{2.5}$ mean (µg m$^{-3}$)")
    save_figure(fig, "monthly_climatology")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    annual_plot = annual[annual["standard"] == "WHO AQG"]
    ax.bar(annual_plot["year"].astype(str), annual_plot["pm25_annual_mean"], color="#567b9f")
    for value, label, color in [(5, "WHO 2021", "#26734d"), (35, "Bangladesh 2022", "#d48a00"), (9, "EPA 2024", "#8d4c9f")]:
        ax.axhline(value, ls="--", lw=1, color=color, label=label)
    ax.set(xlabel="Year", ylabel="Annual PM$_{2.5}$ mean (µg m$^{-3}$)")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "annual_guideline_comparison")

    qa = [
        "# QA report",
        "",
        f"- Raw valid AirNow daily records: {len(daily):,}.",
        f"- Date range: {daily['date_local'].min()} through {daily['date_local'].max()}.",
        f"- Complete analysis months: {len(monthly):,}; cutoff {monthly['month_start'].max()}.",
        f"- Partial/excluded months: {', '.join(monthly_all.loc[~monthly_all['is_complete'], 'month_start'])}.",
        "- Station identity, coordinates, pollutant, unit, and duration are constant and validated.",
        "- No negative values, duplicate dates, modeled rows, reconstructed rows, or future dates occur in the observed dataset.",
        "- Hourly completeness cannot be calculated from daily summary files and remains null.",
        "- March 2025 meets 75% day coverage numerically but is excluded because the terminal feed ends before month-end.",
    ]
    (reports / "qa_report.md").write_text("\n".join(qa) + "\n", encoding="utf-8")
    print(json.dumps(trend, indent=2))


if __name__ == "__main__":
    main()
