#!/usr/bin/env python3
"""Produce paper-ready statistical analysis of the official DoE dataset."""

from __future__ import annotations

import calendar
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dhaka-aqi-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data/processed"
CONTEXT = ROOT / "data/context"
OUTPUT = ROOT / "analysis"
FIGURES = OUTPUT / "figures"

VARIABLES = {
    "aqi": ("aqi_mean", "AQI", "index"),
    "pm25": ("pm25_mean", "PM2.5", "µg/m³"),
    "pm10": ("pm10_mean", "PM10", "µg/m³"),
    "no2": ("no2_mean", "NO2", "ppb / unresolved in some reports"),
    "so2": ("so2_mean", "SO2", "ppb / unresolved in some reports"),
    "co": ("co_mean", "CO (8-hour)", "ppm / unresolved in some reports"),
    "o3": ("o3_mean", "O3 (8-hour)", "ppb / unresolved in some reports"),
}
SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
COLORS = {
    "aqi": "#6A3D9A",
    "pm25": "#D73027",
    "pm10": "#FC8D59",
    "no2": "#4575B4",
    "so2": "#74ADD1",
    "co": "#1A9850",
    "o3": "#66BD63",
}
FIGURE_VARIABLES = {
    "aqi": ("aqi_mean", r"AQI", r"index"),
    "pm25": ("pm25_mean", r"PM$_{2.5}$", r"$\mu\mathrm{g}\,\mathrm{m}^{-3}$"),
    "pm10": ("pm10_mean", r"PM$_{10}$", r"$\mu\mathrm{g}\,\mathrm{m}^{-3}$"),
    "no2": ("no2_mean", r"NO$_2$", r"ppb$^{*}$"),
    "so2": ("so2_mean", r"SO$_2$", r"ppb$^{*}$"),
    "co": ("co_mean", r"CO (8 h)", r"ppm$^{*}$"),
    "o3": ("o3_mean", r"O$_3$ (8 h)", r"ppb$^{*}$"),
}


def season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Pre-monsoon"
    if month in {6, 7, 8, 9}:
        return "Monsoon"
    return "Post-monsoon"


def false_discovery_rate(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, preserving missing values."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    count = len(valid)
    if count == 0:
        return result
    adjusted = valid * count / np.arange(1, count + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    result.loc[adjusted.index] = adjusted
    return result


def descriptive_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = {
        "all_available": wide,
        "recent_comparable_2022_2025": wide[wide["year"].between(2022, 2025)],
    }
    for period_name, frame in periods.items():
        for key, (column, label, unit) in VARIABLES.items():
            values = frame[column].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "period": period_name,
                    "variable": key,
                    "label": label,
                    "unit": unit,
                    "n_months": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std_dev": values.std(ddof=1),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "iqr": values.quantile(0.75) - values.quantile(0.25),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "coefficient_of_variation_pct": 100 * values.std(ddof=1) / values.mean(),
                    "first_month": frame.loc[frame[column].notna(), "month_start"].min().date(),
                    "last_month": frame.loc[frame[column].notna(), "month_start"].max().date(),
                }
            )
    return pd.DataFrame(rows)


def annual_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in wide.groupby("year"):
        for key, (column, label, unit) in VARIABLES.items():
            values = group[column].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "year": year,
                    "variable": key,
                    "label": label,
                    "unit": unit,
                    "n_months": len(values),
                    "annual_mean_of_monthly_values": values.mean(),
                    "annual_median_of_monthly_values": values.median(),
                    "annual_minimum_monthly_value": values.min(),
                    "annual_maximum_monthly_value": values.max(),
                    "complete_12_month_year": len(values) == 12,
                }
            )
    return pd.DataFrame(rows)


def seasonal_tables(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent = wide[wide["year"].between(2022, 2025)].copy()
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for key, (column, label, unit) in VARIABLES.items():
        for season in SEASON_ORDER:
            values = recent.loc[recent["season"].eq(season), column].dropna()
            rows.append(
                {
                    "variable": key,
                    "label": label,
                    "unit": unit,
                    "season": season,
                    "n_months": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std_dev": values.std(ddof=1),
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
        groups = [
            recent.loc[recent["season"].eq(season), column].dropna().to_numpy()
            for season in SEASON_ORDER
        ]
        groups = [group for group in groups if len(group)]
        statistic, p_value = stats.kruskal(*groups)
        n = sum(len(group) for group in groups)
        k = len(groups)
        epsilon_squared = max(0, (statistic - k + 1) / (n - k)) if n > k else np.nan
        tests.append(
            {
                "variable": key,
                "label": label,
                "period": "2022-2025",
                "test": "Kruskal-Wallis across four seasons",
                "n_months": n,
                "h_statistic": statistic,
                "p_value": p_value,
                "epsilon_squared": epsilon_squared,
            }
        )
    test_frame = pd.DataFrame(tests)
    test_frame["q_value_bh"] = false_discovery_rate(test_frame["p_value"])
    return pd.DataFrame(rows), test_frame


def trend_table(wide: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "legacy_2013_2018": (2013, 2018),
        "recent_2022_2025": (2022, 2025),
    }
    rows: list[dict[str, Any]] = []
    for period_name, (start_year, end_year) in periods.items():
        for key, (column, label, unit) in VARIABLES.items():
            frame = wide.loc[
                wide["year"].between(start_year, end_year) & wide[column].notna(),
                ["month_start", "month", column],
            ].copy()
            if len(frame) < 24:
                continue
            frame["elapsed_years"] = (
                (frame["month_start"] - frame["month_start"].min()).dt.days / 365.2425
            )
            month_dummies = pd.get_dummies(
                frame["month"].astype(str), prefix="month", drop_first=True, dtype=float
            )
            design = sm.add_constant(
                pd.concat([frame[["elapsed_years"]].reset_index(drop=True), month_dummies.reset_index(drop=True)], axis=1),
                has_constant="add",
            )
            model = sm.OLS(frame[column].to_numpy(), design).fit(
                cov_type="HAC", cov_kwds={"maxlags": min(12, max(1, len(frame) // 4))}
            )
            low, high = model.conf_int().loc["elapsed_years"]
            rows.append(
                {
                    "period": period_name,
                    "variable": key,
                    "label": label,
                    "unit": unit,
                    "n_months": len(frame),
                    "first_month": frame["month_start"].min().date(),
                    "last_month": frame["month_start"].max().date(),
                    "season_adjusted_slope_per_year": model.params["elapsed_years"],
                    "ci95_low": low,
                    "ci95_high": high,
                    "hac_p_value": model.pvalues["elapsed_years"],
                    "model_r_squared": model.rsquared,
                    "method": "OLS with month fixed effects and Newey-West HAC SE (max lag 12)",
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh_within_period"] = result.groupby("period")["hac_p_value"].transform(
        false_discovery_rate
    )
    result["direction"] = np.where(
        result["season_adjusted_slope_per_year"].gt(0), "increasing", "decreasing"
    )
    result["fdr_significant_0_05"] = result["q_value_bh_within_period"].lt(0.05)
    return result


def correlation_table(wide: pd.DataFrame) -> pd.DataFrame:
    recent = wide[wide["year"].between(2022, 2025)].copy()
    columns = {key: spec[0] for key, spec in VARIABLES.items()}
    residual = recent.copy()
    for column in columns.values():
        residual[column] = residual[column] - residual.groupby("month")[column].transform("mean")
    rows: list[dict[str, Any]] = []
    keys = list(columns)
    for basis, frame in [("raw_monthly", recent), ("month_deseasonalized", residual)]:
        for first_index, first in enumerate(keys):
            for second in keys[first_index + 1 :]:
                pair = frame[[columns[first], columns[second]]].dropna()
                if len(pair) < 8:
                    continue
                coefficient, p_value = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                rows.append(
                    {
                        "basis": basis,
                        "period": "2022-2025",
                        "variable_1": first,
                        "variable_2": second,
                        "n_months": len(pair),
                        "spearman_rho": coefficient,
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["q_value_bh_within_basis"] = result.groupby("basis")["p_value"].transform(
        false_discovery_rate
    )
    return result


def daily_tables(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    selected = daily[
        daily["selected_record"].astype(str).str.lower().isin({"true", "1"})
    ].copy()
    selected["year"] = selected["report_date"].dt.year
    selected["month"] = selected["report_date"].dt.month
    selected["season"] = selected["month"].map(season_for_month)
    selected["category_normalized"] = selected["aqi_category_as_reported"].str.upper()

    year_rows: list[dict[str, Any]] = []
    for year, group in selected.groupby("year"):
        expected = 366 if calendar.isleap(year) else 365
        if year == selected["year"].min():
            expected = (pd.Timestamp(year, 12, 31) - group["report_date"].min()).days + 1
        if year == selected["year"].max():
            expected = (group["report_date"].max() - pd.Timestamp(year, 1, 1)).days + 1
        year_rows.append(
            {
                "year": year,
                "first_report_date": group["report_date"].min().date(),
                "last_report_date": group["report_date"].max().date(),
                "reported_days": len(group),
                "expected_days_within_archive_span": expected,
                "coverage_pct_within_archive_span": 100 * len(group) / expected,
                "mean_aqi": group["aqi"].mean(),
                "median_aqi": group["aqi"].median(),
                "minimum_aqi": group["aqi"].min(),
                "maximum_aqi": group["aqi"].max(),
                "days_aqi_gt_100": group["aqi"].gt(100).sum(),
                "pct_days_aqi_gt_100": 100 * group["aqi"].gt(100).mean(),
                "days_aqi_gt_150": group["aqi"].gt(150).sum(),
                "pct_days_aqi_gt_150": 100 * group["aqi"].gt(150).mean(),
                "days_aqi_gt_200": group["aqi"].gt(200).sum(),
                "pct_days_aqi_gt_200": 100 * group["aqi"].gt(200).mean(),
                "partial_calendar_year": year in {selected["year"].min(), selected["year"].max()},
            }
        )

    season_rows: list[dict[str, Any]] = []
    for season in SEASON_ORDER:
        group = selected[selected["season"].eq(season)]
        season_rows.append(
            {
                "season": season,
                "reported_days": len(group),
                "mean_aqi": group["aqi"].mean(),
                "median_aqi": group["aqi"].median(),
                "minimum_aqi": group["aqi"].min(),
                "maximum_aqi": group["aqi"].max(),
                "pct_days_aqi_gt_100": 100 * group["aqi"].gt(100).mean(),
                "pct_days_aqi_gt_150": 100 * group["aqi"].gt(150).mean(),
                "pct_days_aqi_gt_200": 100 * group["aqi"].gt(200).mean(),
            }
        )

    categories = (
        selected.groupby(["source_category_scheme", "category_normalized"], dropna=False)
        .size()
        .rename("days")
        .reset_index()
    )
    categories["pct_within_scheme"] = categories.groupby("source_category_scheme")["days"].transform(
        lambda values: 100 * values / values.sum()
    )
    top = selected.nlargest(20, "aqi")[
        [
            "report_date", "document_aqi_date", "aqi", "category_normalized",
            "responsible_pollutant", "source_category_scheme", "qa_status", "source_url",
        ]
    ]
    pollutant = (
        selected["responsible_pollutant"]
        .fillna("not_reported")
        .value_counts()
        .rename_axis("responsible_pollutant")
        .reset_index(name="days")
    )
    pollutant["pct_days"] = 100 * pollutant["days"] / len(selected)
    return {
        "daily_year": pd.DataFrame(year_rows),
        "daily_season": pd.DataFrame(season_rows),
        "daily_categories": categories,
        "top_daily_aqi": top,
        "responsible_pollutant": pollutant,
    }


def top_monthly_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key, (column, label, unit) in VARIABLES.items():
        top = wide.nlargest(10, column)[["month_start", "season", column]].copy()
        top.insert(0, "variable", key)
        top.insert(1, "label", label)
        top.insert(2, "unit", unit)
        top = top.rename(columns={column: "monthly_mean_value"})
        top["rank_within_variable"] = np.arange(1, len(top) + 1)
        rows.append(top)
    return pd.concat(rows, ignore_index=True)


def particulate_ratio_table(wide: pd.DataFrame) -> pd.DataFrame:
    result = wide.loc[
        wide["year"].between(2022, 2025),
        ["month_start", "year", "month", "season", "pm25_mean", "pm10_mean"],
    ].dropna().copy()
    result["pm10_to_pm25_ratio"] = result["pm10_mean"] / result["pm25_mean"]
    result["interpretation"] = (
        "Ratio of separate unweighted station-summary means; descriptive, not a source-apportionment measure"
    )
    return result


def source_qa_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    qa = pd.read_csv(PROCESSED / "doe_qa_issues.csv")
    qa_summary = (
        qa.groupby(["stage", "severity", "issue"], dropna=False)
        .size()
        .rename("records")
        .reset_index()
        .sort_values(["severity", "records"], ascending=[True, False])
    )
    manifest = pd.read_csv(PROCESSED / "doe_source_manifest.csv")
    partial = manifest.loc[
        manifest["extraction_status"].eq("partial"),
        ["period", "source_label", "archive_page", "source_url", "sha256"],
    ].sort_values("period")
    return qa_summary, partial


def quality_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = {
        "legacy_2013_2019": wide[wide["year"].between(2013, 2019)],
        "archive_gap_2020_2021": wide[wide["year"].between(2020, 2021)],
        "recent_2022_2025": wide[wide["year"].between(2022, 2025)],
        "partial_2026_through_July": wide[wide["year"].eq(2026)],
    }
    for period, frame in periods.items():
        for key, (column, label, unit) in VARIABLES.items():
            present = frame[column].notna()
            row: dict[str, Any] = {
                "period": period,
                "variable": key,
                "label": label,
                "unit": unit,
                "calendar_months": len(frame),
                "months_with_value": int(present.sum()),
                "coverage_pct": 100 * present.mean() if len(frame) else np.nan,
            }
            if key != "aqi":
                row.update(
                    {
                        "mean_reporting_station_count": frame.loc[
                            present, f"{key}_station_count"
                        ].mean(),
                        "mean_data_capture_pct": frame.loc[
                            present, f"{key}_mean_data_capture_pct"
                        ].mean(),
                        "months_unit_fully_resolved": int(
                            frame.loc[present, f"{key}_unit_status"].eq("resolved").sum()
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "mean_reporting_station_count": np.nan,
                        "mean_data_capture_pct": frame.loc[present, "aqi_coverage_pct"].mean(),
                        "months_unit_fully_resolved": np.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def context_associations(
    annual: pd.DataFrame, population: pd.DataFrame, hdi: pd.DataFrame
) -> pd.DataFrame:
    context = population.merge(
        hdi[["year", "hdi_undp_same_year"]], on="year", how="left"
    )
    annual_wide = annual.pivot(index="year", columns="variable", values="annual_mean_of_monthly_values")
    annual_counts = annual.pivot(index="year", columns="variable", values="n_months")
    merged = context.merge(annual_wide, on="year", how="left")
    rows: list[dict[str, Any]] = []
    context_variables = [
        "total_population", "urban_population", "urban_share_fraction", "hdi_undp_same_year"
    ]
    for air_variable in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        adequate_years = annual_counts[air_variable].ge(5)
        for context_variable in context_variables:
            pair = merged.loc[
                merged["year"].isin(adequate_years[adequate_years].index),
                [air_variable, context_variable],
            ].dropna()
            if len(pair) < 6:
                continue
            rho, p_value = stats.spearmanr(pair[air_variable], pair[context_variable])
            rows.append(
                {
                    "air_variable": air_variable,
                    "national_context_variable": context_variable,
                    "n_annual_observations": len(pair),
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "interpretation_limit": (
                        "Ecological time-series association; confounded by time, station changes, "
                        "missing 2020-2021 reports, and national-versus-city geography"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = false_discovery_rate(result["p_value"])
    return result


def backtest_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train = wide[wide["year"].between(2022, 2024)].copy()
    test = wide[wide["year"].eq(2025)].copy()
    for key, (column, label, unit) in VARIABLES.items():
        training = train.set_index("month_start")[column].asfreq("MS")
        actual = test.set_index("month_start")[column].asfreq("MS")
        if training.notna().sum() != 36:
            continue
        forecasts: dict[str, pd.Series] = {
            "seasonal_naive": pd.Series(training.iloc[-12:].to_numpy(), index=actual.index)
        }
        try:
            model = ExponentialSmoothing(
                training,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecasts["additive_ETS"] = model.forecast(12)
        except Exception:
            pass
        time = np.arange(len(training), dtype=float)
        harmonic_design = np.column_stack(
            [
                np.ones(len(time)),
                time,
                np.sin(2 * np.pi * time / 12),
                np.cos(2 * np.pi * time / 12),
                np.sin(4 * np.pi * time / 12),
                np.cos(4 * np.pi * time / 12),
            ]
        )
        coefficients = np.linalg.lstsq(harmonic_design, training.to_numpy(), rcond=None)[0]
        future_time = np.arange(len(training), len(training) + 12, dtype=float)
        future_design = np.column_stack(
            [
                np.ones(12),
                future_time,
                np.sin(2 * np.pi * future_time / 12),
                np.cos(2 * np.pi * future_time / 12),
                np.sin(4 * np.pi * future_time / 12),
                np.cos(4 * np.pi * future_time / 12),
            ]
        )
        forecasts["linear_trend_plus_harmonics"] = pd.Series(
            future_design @ coefficients, index=actual.index
        )
        for model_name, forecast in forecasts.items():
            paired = pd.concat([actual.rename("actual"), forecast.rename("forecast")], axis=1).dropna()
            error = paired["forecast"] - paired["actual"]
            rows.append(
                {
                    "variable": key,
                    "label": label,
                    "unit": unit,
                    "model": model_name,
                    "training_months": int(training.notna().sum()),
                    "validation_months": len(paired),
                    "mae": error.abs().mean(),
                    "rmse": np.sqrt(np.mean(error**2)),
                    "mape_pct": 100 * np.mean(np.abs(error / paired["actual"])),
                    "validation_period": "2025",
                    "forecast_readiness_note": (
                        "Only three complete training years; insufficient evidence for a defensible "
                        "projection through 2030"
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    wide: pd.DataFrame,
    daily: pd.DataFrame,
    correlations: pd.DataFrame,
    trends: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    use_latex = all(shutil.which(command) for command in ["latex", "dvipng"])
    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titleweight": "regular",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
        }
    )

    fig, axes = plt.subplots(7, 1, figsize=(12, 17), sharex=True)
    for axis, (key, (column, label, unit)) in zip(axes, FIGURE_VARIABLES.items(), strict=True):
        axis.plot(wide["month_start"], wide[column], color=COLORS[key], linewidth=1.6)
        axis.set_ylabel(f"{label}\n({unit})", fontsize=8)
        axis.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), color="#CCCCCC", alpha=0.35)
    axes[0].set_title(
        r"Official DoE monthly Dhaka air-quality series"
        "\n"
        r"Gray band: no linked monthly pollutant reports for 2020--2021"
    )
    axes[-1].set_xlabel("Month")
    axes[-1].text(
        0,
        -0.35,
        r"$^{*}$Unit not explicit in every parsed source summary table; see unit-status fields.",
        transform=axes[-1].transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "monthly_time_series.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    recent = wide[wide["year"].between(2022, 2025)]
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True)
    for axis, (key, (column, label, unit)) in zip(
        axes.flat, FIGURE_VARIABLES.items(), strict=False
    ):
        summary = recent.groupby("month")[column].agg(["mean", "std", "count"])
        error = summary["std"] / np.sqrt(summary["count"])
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=error,
            color=COLORS[key],
            marker="o",
            capsize=3,
        )
        axis.set_title(label)
        axis.set_ylabel(unit)
        axis.set_xticks(range(1, 13))
        axis.set_xticklabels([calendar.month_abbr[index] for index in range(1, 13)], rotation=45)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05,
        0.75,
        r"$^{*}$Unit not explicit in every parsed source summary table.",
        transform=axes.flat[-1].transAxes,
        fontsize=10,
    )
    fig.suptitle(r"Monthly climatology, 2022--2025 (mean $\pm$ standard error)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "monthly_climatology_2022_2025.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = LinearSegmentedColormap.from_list("correlation", ["#2166AC", "#FFFFFF", "#B2182B"])
    for axis, basis in zip(axes, ["raw_monthly", "month_deseasonalized"], strict=True):
        subset = correlations[correlations["basis"].eq(basis)]
        matrix = pd.DataFrame(
            np.eye(len(FIGURE_VARIABLES)), index=FIGURE_VARIABLES, columns=FIGURE_VARIABLES
        )
        for row in subset.itertuples():
            matrix.loc[row.variable_1, row.variable_2] = row.spearman_rho
            matrix.loc[row.variable_2, row.variable_1] = row.spearman_rho
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap=cmap)
        axis.grid(False)
        latex_ticks = [FIGURE_VARIABLES[key][1] for key in matrix.columns]
        axis.set_xticks(range(len(matrix)), latex_ticks, rotation=45, ha="right")
        axis.set_yticks(range(len(matrix)), latex_ticks)
        axis.set_title(basis.replace("_", " ").title())
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                value = matrix.iloc[i, j]
                red, green, blue, _ = cmap((value + 1) / 2)
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "white" if luminance < 0.62 else "#222222"
                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold" if luminance < 0.62 else "normal",
                )
    fig.colorbar(image, ax=axes, shrink=0.75, label=r"Spearman $\rho$")
    fig.suptitle(r"Pollutant and AQI correlations, 2022--2025")
    fig.savefig(FIGURES / "correlations_2022_2025.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    selected = daily[daily["selected_record"].astype(str).str.lower().isin({"true", "1"})].copy()
    selected = selected.sort_values("report_date")
    selected["rolling_30_day_mean"] = selected.set_index("report_date")["aqi"].rolling("30D").mean().to_numpy()
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.plot(selected["report_date"], selected["aqi"], color="#999999", linewidth=0.6, alpha=0.7, label="Daily AQI")
    axis.plot(selected["report_date"], selected["rolling_30_day_mean"], color="#6A3D9A", linewidth=2, label="30-day mean")
    for threshold in [100, 150, 200, 300]:
        axis.axhline(threshold, color="#444444", linewidth=0.6, linestyle="--", alpha=0.5)
    axis.set_title("DoE-published daily Dhaka AQI")
    axis.set_ylabel("AQI")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "daily_aqi.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    coverage_columns = [spec[0] for spec in FIGURE_VARIABLES.values()]
    coverage = wide.set_index("month_start")[coverage_columns].notna().astype(int).T
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.imshow(coverage, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axis.set_yticks(
        range(len(coverage)), [FIGURE_VARIABLES[key][1] for key in FIGURE_VARIABLES]
    )
    ticks = np.arange(0, len(coverage.columns), 12)
    axis.set_xticks(ticks, [coverage.columns[index].strftime("%Y") for index in ticks])
    axis.set_title("Monthly data availability (dark = reported value)")
    axis.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(FIGURES / "data_availability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True)
    annual = recent.groupby("year")[[spec[0] for spec in FIGURE_VARIABLES.values()]].mean()
    for axis, (key, (column, label, unit)) in zip(
        axes.flat, FIGURE_VARIABLES.items(), strict=False
    ):
        axis.plot(
            annual.index,
            annual[column],
            color=COLORS[key],
            marker="o",
            linewidth=2,
        )
        axis.set_title(label)
        axis.set_ylabel(unit)
        axis.set_xticks(annual.index)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05,
        0.75,
        r"$^{*}$Unit not explicit in every parsed source summary table.",
        transform=axes.flat[-1].transAxes,
        fontsize=10,
    )
    fig.suptitle(r"Annual mean air-quality indicators, 2022--2025", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "annual_means_2022_2025.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    selected["month"] = selected["report_date"].dt.month
    selected["season"] = selected["month"].map(season_for_month)
    seasonal_values = [
        selected.loc[selected["season"].eq(season), "aqi"].dropna().to_numpy()
        for season in SEASON_ORDER
    ]
    season_colors = ["#6A3D9A", "#D95F02", "#1B9E77", "#7570B3"]
    fig, axis = plt.subplots(figsize=(10, 6))
    boxes = axis.boxplot(
        seasonal_values,
        tick_labels=SEASON_ORDER,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops={"color": "white", "linewidth": 1.8},
    )
    for box, color in zip(boxes["boxes"], season_colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.85)
    means = [np.mean(values) for values in seasonal_values]
    axis.scatter(range(1, 5), means, color="black", marker="D", s=35, label="Mean", zorder=4)
    axis.set_title(r"Distribution of daily Dhaka AQI by season")
    axis.set_ylabel("AQI")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "daily_aqi_by_season.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    recent_trends = trends[trends["period"].eq("recent_2022_2025")].copy()
    recent_means = recent[[spec[0] for spec in FIGURE_VARIABLES.values()]].mean()
    recent_trends["period_mean"] = recent_trends["variable"].map(
        {key: recent_means[spec[0]] for key, spec in FIGURE_VARIABLES.items()}
    )
    for column in ["season_adjusted_slope_per_year", "ci95_low", "ci95_high"]:
        recent_trends[f"{column}_pct"] = 100 * recent_trends[column] / recent_trends["period_mean"]
    recent_trends = recent_trends.set_index("variable").loc[list(FIGURE_VARIABLES)].reset_index()
    positions = np.arange(len(recent_trends))
    estimates = recent_trends["season_adjusted_slope_per_year_pct"].to_numpy()
    lower = recent_trends["ci95_low_pct"].to_numpy()
    upper = recent_trends["ci95_high_pct"].to_numpy()
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.axvline(0, color="#444444", linewidth=1, linestyle="--")
    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="none",
        ecolor="#666666",
        capsize=4,
        linewidth=1.5,
    )
    for position, row in recent_trends.iterrows():
        significant = bool(row["fdr_significant_0_05"])
        axis.scatter(
            row["season_adjusted_slope_per_year_pct"],
            position,
            s=70,
            color=COLORS[row["variable"]] if significant else "white",
            edgecolor=COLORS[row["variable"]],
            linewidth=1.8,
            zorder=3,
        )
    axis.set_yticks(
        positions, [FIGURE_VARIABLES[key][1] for key in recent_trends["variable"]]
    )
    axis.invert_yaxis()
    axis.set_xlabel(r"Season-adjusted change per year (\% of 2022--2025 mean)")
    axis.set_title(
        r"Temporal trend estimates, 2022--2025"
        "\n"
        r"Filled markers: Benjamini--Hochberg $q<0.05$"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "trend_estimates_2022_2025.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for key, (_, label, _) in FIGURE_VARIABLES.items():
        if key == "aqi":
            continue
        station_column = f"{key}_station_count"
        capture_column = f"{key}_mean_data_capture_pct"
        annual_quality = recent.groupby("year")[[station_column, capture_column]].mean()
        axes[0].plot(
            annual_quality.index,
            annual_quality[station_column],
            color=COLORS[key],
            marker="o",
            label=label,
        )
        axes[1].plot(
            annual_quality.index,
            annual_quality[capture_column],
            color=COLORS[key],
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("Mean station count")
    axes[0].set_title(r"Monitoring support for monthly pollutant aggregates, 2022--2025")
    axes[1].set_ylabel(r"Mean data capture (\%)")
    axes[1].set_xlabel("Year")
    axes[1].set_xticks(sorted(recent["year"].unique()))
    axes[0].legend(ncol=3, loc="best")
    axes[1].text(
        0,
        -0.32,
        "Station count is the number of nonmissing station averages; capture is averaged across available stations.",
        transform=axes[1].transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "monitoring_support_2022_2025.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Compare seasonal timing across indicators without combining unlike measurement units.
    monthly_fingerprint = recent.groupby("month")[
        [spec[0] for spec in FIGURE_VARIABLES.values()]
    ].mean()
    standardized_fingerprint = (
        monthly_fingerprint - monthly_fingerprint.mean()
    ) / monthly_fingerprint.std(ddof=1)
    fingerprint_matrix = standardized_fingerprint.T
    fingerprint_cmap = LinearSegmentedColormap.from_list(
        "seasonal_fingerprint", ["#2166AC", "#F7F7F7", "#B2182B"]
    )
    fingerprint_limit = 2.2
    fig, axis = plt.subplots(figsize=(13, 5.5))
    image = axis.imshow(
        fingerprint_matrix,
        aspect="auto",
        cmap=fingerprint_cmap,
        vmin=-fingerprint_limit,
        vmax=fingerprint_limit,
    )
    axis.grid(False)
    axis.set_xticks(
        range(12), [calendar.month_abbr[index] for index in range(1, 13)]
    )
    axis.set_yticks(
        range(len(FIGURE_VARIABLES)),
        [FIGURE_VARIABLES[key][1] for key in FIGURE_VARIABLES],
    )
    for row_index in range(fingerprint_matrix.shape[0]):
        for column_index in range(fingerprint_matrix.shape[1]):
            value = fingerprint_matrix.iloc[row_index, column_index]
            red, green, blue, _ = fingerprint_cmap(
                (value + fingerprint_limit) / (2 * fingerprint_limit)
            )
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if luminance < 0.62 else "#222222",
                fontweight="bold" if luminance < 0.62 else "normal",
            )
    fig.colorbar(image, ax=axis, shrink=0.82, label=r"Within-indicator standardized mean ($z$)")
    axis.set_title(
        r"Seasonal air-quality fingerprint, 2022--2025"
        "\n"
        r"Red indicates months above an indicator's own annual profile; blue indicates below"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "seasonal_fingerprint_2022_2025.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    # Report a fixed numeric threshold because source category labels changed over time.
    daily_comparison = selected[selected["report_date"].dt.year.between(2024, 2025)].copy()
    daily_comparison["year"] = daily_comparison["report_date"].dt.year
    daily_comparison["month"] = daily_comparison["report_date"].dt.month
    exceedance = (
        daily_comparison.groupby(["year", "month"])["aqi"]
        .agg(
            reported_days="count",
            pct_days_above_150=lambda values: 100 * (values > 150).mean(),
        )
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(12, 6))
    positions = np.arange(1, 13)
    width = 0.36
    for offset, (year, color) in zip(
        [-width / 2, width / 2], [(2024, "#4575B4"), (2025, "#D73027")], strict=True
    ):
        annual_exceedance = exceedance[exceedance["year"].eq(year)].set_index("month")
        values = annual_exceedance.reindex(positions)["pct_days_above_150"]
        axis.bar(
            positions + offset,
            values,
            width=width,
            color=color,
            alpha=0.88,
            label=str(year),
        )
    axis.set_xticks(positions, [calendar.month_abbr[index] for index in positions])
    axis.set_ylim(0, 105)
    axis.set_ylabel(r"Reported days with AQI $>150$ (\%)")
    axis.set_title(r"Monthly frequency of high-AQI days in Dhaka, 2024--2025")
    axis.legend(title="Year")
    axis.text(
        0,
        -0.18,
        "Percentages use selected DoE daily reports; 2025 contains 357 reported days.",
        transform=axis.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "monthly_high_aqi_frequency_2024_2025.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Show the two particle indicators most closely associated with published monthly AQI.
    season_colors_map = dict(zip(SEASON_ORDER, season_colors, strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, key in zip(axes, ["pm25", "pm10"], strict=True):
        column, label, unit = FIGURE_VARIABLES[key]
        paired = recent[[column, "aqi_mean", "season"]].dropna()
        for season in SEASON_ORDER:
            season_data = paired[paired["season"].eq(season)]
            axis.scatter(
                season_data[column],
                season_data["aqi_mean"],
                color=season_colors_map[season],
                s=42,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.5,
                label=season,
            )
        slope, intercept, _, _ = stats.theilslopes(paired["aqi_mean"], paired[column])
        line_x = np.linspace(paired[column].min(), paired[column].max(), 100)
        axis.plot(line_x, intercept + slope * line_x, color="#222222", linewidth=1.5)
        rho, p_value = stats.spearmanr(paired[column], paired["aqi_mean"])
        axis.text(
            0.04,
            0.94,
            rf"Spearman $\rho={rho:.2f}$; $n={len(paired)}$",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.set_xlabel(f"{label} ({unit})")
        axis.set_ylabel("Monthly mean AQI")
        axis.set_title(f"AQI and {label}")
    axes[1].legend(title="Season", loc="lower right")
    fig.suptitle(
        r"Monthly particulate matter--AQI relationships, 2022--2025"
        "\n"
        r"Lines are Theil--Sen summaries; associations are descriptive, not causal"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "aqi_particulate_relationships_2022_2025.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # The ratio is descriptive because the source values are separate station aggregates.
    ratio_frame = recent[["season", "pm10_mean", "pm25_mean"]].dropna().copy()
    ratio_frame["pm10_to_pm25"] = ratio_frame["pm10_mean"] / ratio_frame["pm25_mean"]
    ratio_values = [
        ratio_frame.loc[ratio_frame["season"].eq(season), "pm10_to_pm25"].to_numpy()
        for season in SEASON_ORDER
    ]
    fig, axis = plt.subplots(figsize=(10, 6))
    boxes = axis.boxplot(
        ratio_values,
        tick_labels=SEASON_ORDER,
        patch_artist=True,
        showfliers=False,
        widths=0.58,
        medianprops={"color": "white", "linewidth": 1.8},
    )
    for box, color in zip(boxes["boxes"], season_colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.82)
    random_generator = np.random.default_rng(2025)
    for position, (values, color) in enumerate(
        zip(ratio_values, season_colors, strict=True), start=1
    ):
        jitter = random_generator.uniform(-0.11, 0.11, size=len(values))
        axis.scatter(
            position + jitter,
            values,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            s=28,
            alpha=0.75,
            zorder=3,
        )
    axis.axhline(1, color="#444444", linestyle="--", linewidth=1)
    axis.set_ylabel(r"Monthly PM$_{10}$:PM$_{2.5}$ ratio")
    axis.set_title(r"Seasonal variation in the PM$_{10}$:PM$_{2.5}$ ratio, 2022--2025")
    axis.text(
        0,
        -0.18,
        "Ratio of separate DoE monthly station-summary aggregates; not a source-apportionment measure.",
        transform=axis.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "particulate_ratio_by_season_2022_2025.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_frame(workbook: Workbook, name: str, frame: pd.DataFrame) -> None:
    sheet = workbook.create_sheet(name)
    sheet.append(frame.columns.tolist())
    for row in frame.itertuples(index=False, name=None):
        sheet.append([None if pd.isna(value) else value for value in row])
    fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = fill
        cell.alignment = Alignment(wrap_text=True)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, name_value in enumerate(frame.columns, start=1):
        sample = [str(name_value)] + [
            str(sheet.cell(row=row, column=index).value or "")
            for row in range(2, min(sheet.max_row, 102) + 1)
        ]
        sheet.column_dimensions[get_column_letter(index)].width = min(
            55, max(10, max(len(value) for value in sample) + 2)
        )


def write_analysis_workbook(tables: dict[str, pd.DataFrame]) -> None:
    workbook = Workbook()
    readme = workbook.active
    readme.title = "README"
    rows = [
        ("Purpose", "Paper-ready exploratory and statistical analysis of official Bangladesh DoE data"),
        ("Main analysis period", "2022-2025 for comparable monthly AQI and pollutant analysis"),
        ("Legacy period", "2013-2018 analyzed separately; no linked 2020-2021 pollutant reports"),
        ("Trend method", "Month-adjusted OLS with Newey-West HAC standard errors; descriptive, not causal"),
        ("Multiple testing", "Benjamini-Hochberg q-values reported for test families"),
        ("Correlations", "Both raw monthly and month-deseasonalized Spearman correlations are supplied"),
        ("Context warning", "Population/HDI are national annual variables; air quality is Dhaka and station coverage changes"),
        ("Forecast warning", "Backtests are diagnostics; three complete training years do not justify forecasts through 2030"),
        ("Missing values", "Not imputed"),
        ("Pollutant medians", "Not available in DoE source reports; analysis uses reported monthly station averages"),
    ]
    for row in rows:
        readme.append(row)
    readme.column_dimensions["A"].width = 23
    readme.column_dimensions["B"].width = 110
    for cell in readme[1]:
        cell.font = Font(bold=True)
    for name, frame in tables.items():
        write_frame(workbook, name[:31], frame)
    workbook.save(OUTPUT / "dhaka_doe_analysis.xlsx")


def fmt(value: float, decimals: int = 1) -> str:
    return "NA" if pd.isna(value) else f"{value:.{decimals}f}"


def build_report(
    wide: pd.DataFrame,
    descriptive: pd.DataFrame,
    annual: pd.DataFrame,
    seasonal: pd.DataFrame,
    season_tests: pd.DataFrame,
    trends: pd.DataFrame,
    correlations: pd.DataFrame,
    daily_tables_result: dict[str, pd.DataFrame],
    quality: pd.DataFrame,
    particulate_ratio: pd.DataFrame,
    qa_summary: pd.DataFrame,
    context: pd.DataFrame,
    backtests: pd.DataFrame,
) -> str:
    recent_desc = descriptive[descriptive["period"].eq("recent_comparable_2022_2025")].set_index("variable")
    recent_annual = annual[annual["year"].between(2022, 2025)].pivot(
        index="year", columns="variable", values="annual_mean_of_monthly_values"
    )
    recent_seasonal = seasonal.set_index(["variable", "season"])
    recent_trends = trends[trends["period"].eq("recent_2022_2025")].set_index("variable")
    raw_corr = correlations[correlations["basis"].eq("raw_monthly")]
    adjusted_corr = correlations[correlations["basis"].eq("month_deseasonalized")]
    daily_year = daily_tables_result["daily_year"]
    daily_season = daily_tables_result["daily_season"].set_index("season")
    responsible = daily_tables_result["responsible_pollutant"].set_index("responsible_pollutant")
    selected_daily = daily_tables_result["top_daily_aqi"]

    lines = [
        "# Research findings from the official DoE Dhaka air-quality dataset",
        "",
        "## Scope and defensible study design",
        "",
        f"The analysis-ready file contains {len(wide)} calendar-month rows from "
        f"{wide['month_start'].min():%B %Y} through {wide['month_start'].max():%B %Y}. "
        "Pollutant summaries begin in 2013, but the DoE master archive has no linked "
        "2020–2021 monthly reports. Numeric monthly AQI is available from January 2022. "
        "Consequently, **2022–2025 is the strongest common comparison window**; 2026 is partial.",
        "",
        "The original manuscript's complete 2017–2025 panel and 2030 forecasting design should not "
        "be retained unchanged. It would conceal a two-year pollutant-report gap and treat pre-2022 AQI "
        "as observed when no comparable numeric series exists in these official reports.",
        "",
        "## Descriptive results, 2022–2025",
        "",
        "| Variable | Months | Mean | Median | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, (_, label, unit) in VARIABLES.items():
        row = recent_desc.loc[key]
        lines.append(
            f"| {label} ({unit}) | {int(row.n_months)} | {fmt(row['mean'])} | "
            f"{fmt(row['median'])} | {fmt(row['minimum'])} | {fmt(row['maximum'])} |"
        )

    lines.extend(
        [
            "",
            "Annual means show why the overall trend requires nuance:",
            "",
            "| Year | AQI | PM2.5 | PM10 | O3 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for year, row in recent_annual.iterrows():
        lines.append(
            f"| {year} | {fmt(row.get('aqi'))} | {fmt(row.get('pm25'))} | "
            f"{fmt(row.get('pm10'))} | {fmt(row.get('o3'))} |"
        )
    lines.append(
        "AQI fell sharply from the unusually high 2022 average, but rebounded from 150.2 in 2024 "
        "to 159.5 in 2025. PM2.5 and PM10 annual means continued downward, while O3 increased sharply "
        "in 2025; the O3 result is provisional because source-unit visibility and station coverage vary."
    )

    lines.extend(["", "## Seasonality", ""])
    for key, (_, label, _) in VARIABLES.items():
        values = recent_seasonal.loc[key]
        highest = values["mean"].idxmax()
        lowest = values["mean"].idxmin()
        ratio = values.loc[highest, "mean"] / values.loc[lowest, "mean"]
        test = season_tests.set_index("variable").loc[key]
        lines.append(
            f"- **{label}:** highest in {highest} ({fmt(values.loc[highest, 'mean'])}) and "
            f"lowest in {lowest} ({fmt(values.loc[lowest, 'mean'])}); ratio {ratio:.2f}. "
            f"Kruskal–Wallis H={test.h_statistic:.2f}, q={test.q_value_bh:.4f}, "
            f"ε²={test.epsilon_squared:.2f}."
        )
    ratio_median = particulate_ratio["pm10_to_pm25_ratio"].median()
    ratio_season = particulate_ratio.groupby("season")["pm10_to_pm25_ratio"].mean()
    lines.append(
        f"The median monthly PM10:PM2.5 ratio was {ratio_median:.2f}. Its mean was highest in "
        f"{ratio_season.idxmax()} ({ratio_season.max():.2f}) and lowest in "
        f"{ratio_season.idxmin()} ({ratio_season.min():.2f}). This is descriptive only: the numerator "
        "and denominator are separate station-summary aggregates, not paired source-apportionment measurements."
    )

    lines.extend(["", "## Season-adjusted temporal trends", ""])
    for key, (_, label, unit) in VARIABLES.items():
        if key not in recent_trends.index:
            continue
        row = recent_trends.loc[key]
        significance = "FDR-significant" if row.fdr_significant_0_05 else "not FDR-significant"
        lines.append(
            f"- **{label}:** {row.direction} by {abs(row.season_adjusted_slope_per_year):.2f} "
            f"{unit} per year (95% CI {row.ci95_low:.2f} to {row.ci95_high:.2f}; "
            f"q={row.q_value_bh_within_period:.4f}; {significance})."
        )
    lines.append(
        "These are month-adjusted descriptive trends, not emission-source or causal effects. Station "
        "composition, data capture, and unresolved source units can influence apparent changes. The decline "
        "in AQI is strongly influenced by the high 2022 baseline and does not imply uninterrupted improvement."
    )

    lines.extend(["", "## Pollutant relationships", ""])
    for title, table in [("Raw monthly", raw_corr), ("After removing month-of-year means", adjusted_corr)]:
        strongest = table.reindex(table["spearman_rho"].abs().sort_values(ascending=False).index).head(5)
        lines.append(f"**{title} correlations (five strongest):**")
        lines.append("")
        for row in strongest.itertuples():
            lines.append(
                f"- {row.variable_1}–{row.variable_2}: Spearman ρ={row.spearman_rho:.2f}, "
                f"q={row.q_value_bh_within_basis:.4f}, n={row.n_months}."
            )
        lines.append("")
    lines.append(
        "The deseasonalized matrix is the better evidence for co-movement beyond the shared winter–monsoon "
        "cycle. AQI correlations are still not independent causal tests because AQI is calculated from pollutant "
        "sub-indices and DoE identifies the controlling pollutant."
    )

    full_years = daily_year[~daily_year["partial_calendar_year"]]
    lines.extend(
        [
            "",
            "## Daily AQI findings",
            "",
            f"The selected daily archive contains {int(daily_year['reported_days'].sum())} unambiguous reports. "
            f"Across complete calendar years 2024–2025, mean AQI was {fmt(full_years['mean_aqi'].mean())}. "
            f"Winter daily AQI averaged {fmt(daily_season.loc['Winter', 'mean_aqi'])}, compared with "
            f"{fmt(daily_season.loc['Monsoon', 'mean_aqi'])} in monsoon. "
            f"{daily_season.loc['Winter', 'pct_days_aqi_gt_150']:.1f}% of reported winter days exceeded 150, "
            f"versus {daily_season.loc['Monsoon', 'pct_days_aqi_gt_150']:.1f}% in monsoon.",
            "",
            f"DoE named PM2.5 as the responsible pollutant on "
            f"{responsible.loc['PM2.5', 'pct_days']:.2f}% of selected daily reports. This supports PM2.5's "
            "dominance in the published AQI product, but it does **not** make PM10, NO2, SO2, CO, or O3 "
            "unimportant exposure indicators.",
            "",
            f"The largest selected daily AQI was {int(selected_daily.iloc[0]['aqi'])} on "
            f"{selected_daily.iloc[0]['report_date']:%d %B %Y}. Category counts must be reported separately "
            "by `source_category_scheme` because DoE labels changed during the archive.",
        ]
    )

    lines.extend(["", "## Data quality and provenance diagnostics", ""])
    recent_quality = quality[
        quality["period"].eq("recent_2022_2025") & ~quality["variable"].eq("aqi")
    ]
    lines.append(
        f"Recent pollutant coverage is {recent_quality['coverage_pct'].min():.1f}%–"
        f"{recent_quality['coverage_pct'].max():.1f}% by variable. Only "
        f"{int(recent_quality['months_unit_fully_resolved'].median())} of 47 reported months have a "
        "fully explicit parsed unit for each pollutant; unresolved months are retained and flagged rather "
        "than silently converted. Mean reporting-station counts range from "
        f"{recent_quality['mean_reporting_station_count'].min():.2f} to "
        f"{recent_quality['mean_reporting_station_count'].max():.2f}."
    )
    issue_counts = qa_summary.set_index("issue")["records"]
    mismatch_count = int(issue_counts.get("document_date_mismatch", 0))
    conflict_count = int(issue_counts.get("conflicting_duplicate_date", 0))
    partial_count = int(
        qa_summary.loc[qa_summary["stage"].eq("monthly_record_validation"), "records"].sum()
    )
    lines.append(
        f"QA records include {mismatch_count} document-date mismatches, {conflict_count} conflicting "
        f"daily duplicate dates, and {partial_count} partially extracted monthly reports. Conflicting daily "
        "duplicates are excluded from selected-record analysis; all source files and hashes remain in the manifest."
    )

    lines.extend(["", "## Population and HDI context", ""])
    if not context.empty:
        strongest_context = context.reindex(context["spearman_rho"].abs().sort_values(ascending=False).index).head(5)
        for row in strongest_context.itertuples():
            lines.append(
                f"- {row.air_variable} versus {row.national_context_variable}: ρ={row.spearman_rho:.2f}, "
                f"q={row.q_value_bh:.4f}, n={row.n_annual_observations} annual observations."
            )
    lines.append(
        "These ecological correlations should not be presented as effects of population or development on "
        "Dhaka air pollution. Population and HDI are national annual measures, while air-quality observations "
        "represent changing Dhaka monitoring stations; both sets of variables also change with calendar time."
    )

    lines.extend(["", "## Forecasting readiness", ""])
    for key in VARIABLES:
        subset = backtests[backtests["variable"].eq(key)]
        if subset.empty:
            continue
        best = subset.loc[subset["mae"].idxmin()]
        lines.append(
            f"- **{VARIABLES[key][1]}:** best 2025 diagnostic was {best['model']} "
            f"(MAE={best['mae']:.2f}, RMSE={best['rmse']:.2f}, {int(best['validation_months'])} validation months)."
        )
    lines.append(
        "Only 36 complete common training months (2022–2024) precede this validation year. A forecast through "
        "2030 would extend five times farther than the one-year validation horizon and would be highly sensitive "
        "to station changes and unusual future conditions. The evidence supports seasonal description and "
        "short-horizon monitoring benchmarks, not a publication-grade 2030 projection yet."
    )

    lines.extend(
        [
            "",
            "## Suggested figure selection",
            "",
            "For the main paper, prioritize the monthly climatology, daily AQI seasonality, temporal-trend "
            "estimates, high-AQI frequency, and particulate matter–AQI relationship figures. Together they "
            "cover the core seasonal, temporal, public-health-threshold, and pollutant-association results.",
            "",
            "Use the correlation and seasonal-fingerprint heatmaps to compare indicators. The full time series, "
            "annual means, PM10:PM2.5 ratio, data availability, and monitoring-support figures are best suited "
            "to supplementary material or the data-quality section.",
            "",
            "## Recommended paper structure and claims",
            "",
            "1. Reframe the study as an **official-source exploratory and temporal analysis**, with forecasting "
            "readiness assessed rather than assumed.",
            "2. Use 2022–2025 for the primary multivariate AQI–pollutant analysis; present 2013–2019 pollutant "
            "history as a separate legacy-report era.",
            "3. Lead with the winter–monsoon contrast, PM2.5's dominance of published AQI, the persistence of "
            "high daily values, and the distinct behavior of gaseous pollutants.",
            "4. Report station counts, data capture, explicit-unit status, AQI coverage, and source-basis fields "
            "alongside results.",
            "5. Do not claim a COVID-period effect: the official linked pollutant archive is absent for 2020–2021.",
            "6. Do not interpret national population or HDI correlations causally or as Dhaka-specific demographic effects.",
            "7. Do not fill pollutant medians or pre-2022 numeric AQI. Those values are unavailable, not zero.",
            "",
            "## Core limitations",
            "",
            "- Monthly pollutant values are aggregates across a changing set of reporting stations, not a fixed-site city mean.",
            "- The wide-table mean is an unweighted mean of station monthly averages; stations with different capture rates receive equal weight.",
            "- Source summary tables sometimes omit or ambiguously expose pollutant units; values are not converted.",
            "- Pollutant medians cannot be recovered from the published summary statistics.",
            "- AQI and pollutant concentrations have different meanings and must not be merged as one measurement scale.",
            "- Daily AQI uses archive report date because some document-internal dates conflict; QA flags remain available.",
            "- Multiple comparisons are controlled with Benjamini–Hochberg q-values, but observational dependence remains.",
            "",
            "## Reproducible outputs",
            "",
            "- `analysis/dhaka_doe_analysis.xlsx`: all result tables.",
            "- `analysis/figures/`: thirteen paper-ready diagnostic figures.",
            "- `scripts/analyze_doe_dataset.py`: complete analysis code.",
            "- `data/processed/dhaka_doe_air_quality.xlsx`: source dataset and provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(PROCESSED / "doe_monthly_dataset_wide.csv", parse_dates=["month_start"])
    daily = pd.read_csv(PROCESSED / "doe_daily_dhaka_aqi.csv", parse_dates=["report_date"])
    population = pd.read_csv(CONTEXT / "bangladesh_population.csv")
    hdi = pd.read_csv(CONTEXT / "bangladesh_hdi.csv")

    descriptive = descriptive_table(wide)
    annual = annual_table(wide)
    seasonal, season_tests = seasonal_tables(wide)
    trends = trend_table(wide)
    correlations = correlation_table(wide)
    daily_result = daily_tables(daily)
    top_monthly = top_monthly_table(wide)
    particulate_ratio = particulate_ratio_table(wide)
    quality = quality_table(wide)
    qa_summary, partial_reports = source_qa_tables()
    context = context_associations(annual, population, hdi)
    backtests = backtest_table(wide)

    tables = {
        "descriptive": descriptive,
        "annual": annual,
        "seasonal": seasonal,
        "season_tests": season_tests,
        "trends": trends,
        "correlations": correlations,
        **daily_result,
        "top_monthly": top_monthly,
        "pm_ratio": particulate_ratio,
        "data_quality": quality,
        "qa_summary": qa_summary,
        "partial_reports": partial_reports,
        "context_assoc": context,
        "forecast_backtest": backtests,
    }
    make_figures(wide, daily, correlations, trends)
    write_analysis_workbook(tables)
    report = build_report(
        wide,
        descriptive,
        annual,
        seasonal,
        season_tests,
        trends,
        correlations,
        daily_result,
        quality,
        particulate_ratio,
        qa_summary,
        context,
        backtests,
    )
    (OUTPUT / "RESEARCH_FINDINGS.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'dhaka_doe_analysis.xlsx'}")
    print(f"Wrote {OUTPUT / 'RESEARCH_FINDINGS.md'}")
    print(f"Wrote {len(list(FIGURES.glob('*.png')))} figures to {FIGURES}")


if __name__ == "__main__":
    main()
