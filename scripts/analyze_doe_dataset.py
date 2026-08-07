#!/usr/bin/env python3
"""Generate the focused, paper-ready analysis of official Dhaka DoE data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dhaka-aqi-matplotlib")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
CONTEXT = ROOT / "data/context"
OUTPUT = ROOT / "analysis"
FIGURES = OUTPUT / "figures"

SEASONS = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
CORE_VARIABLES = {
    "aqi": ("aqi_mean", "AQI", "index"),
    "pm25": ("pm25_mean", r"PM$_{2.5}$", r"µg m$^{-3}$"),
    "pm10": ("pm10_mean", r"PM$_{10}$", r"µg m$^{-3}$"),
}
COLORS = {"aqi": "#5E3C99", "pm25": "#D73027", "pm10": "#F46D43"}


def season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Pre-monsoon"
    if month in {6, 7, 8, 9}:
        return "Monsoon"
    return "Post-monsoon"


def fdr(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    ordered = p_values.dropna().sort_values()
    if ordered.empty:
        return result
    adjusted = ordered * len(ordered) / np.arange(1, len(ordered) + 1)
    result.loc[ordered.index] = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return result


def bootstrap_mean_ci(values: pd.Series, seed: int, draws: int = 5_000) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def descriptive_table(wide: pd.DataFrame) -> pd.DataFrame:
    recent = wide[wide["year"].between(2022, 2025)]
    rows: list[dict[str, Any]] = []
    for key, (column, label, unit) in CORE_VARIABLES.items():
        values = recent[column].dropna()
        rows.append(
            {
                "variable": key,
                "label": label.replace("$", ""),
                "unit": unit.replace("$", ""),
                "analysis_period": "2022-01 through 2025-12",
                "n_months": len(values),
                "mean": values.mean(),
                "standard_deviation": values.std(ddof=1),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )
    return pd.DataFrame(rows)


def seasonal_tables(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent = wide[wide["year"].between(2022, 2025)].copy()
    recent["season"] = pd.Categorical(recent["season"], SEASONS, ordered=True)
    summaries: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for variable_index, (key, (column, label, unit)) in enumerate(CORE_VARIABLES.items()):
        groups = []
        for season_index, season in enumerate(SEASONS):
            values = recent.loc[recent["season"].eq(season), column].dropna()
            groups.append(values.to_numpy())
            low, high = bootstrap_mean_ci(values, 20260806 + 10 * variable_index + season_index)
            summaries.append(
                {
                    "variable": key,
                    "label": label.replace("$", ""),
                    "unit": unit.replace("$", ""),
                    "season": season,
                    "n_months": len(values),
                    "mean": values.mean(),
                    "mean_bootstrap_ci_low": low,
                    "mean_bootstrap_ci_high": high,
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                }
            )
        statistic, p_value = stats.kruskal(*groups)
        tests.append(
            {
                "variable": key,
                "test": "Kruskal-Wallis across four seasons",
                "statistic": statistic,
                "p_value": p_value,
                "n_months": sum(map(len, groups)),
            }
        )
    tests_frame = pd.DataFrame(tests)
    tests_frame["q_value_bh"] = fdr(tests_frame["p_value"])
    return pd.DataFrame(summaries), tests_frame


def prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    selected = daily[daily["selected_record"].astype(str).str.lower().eq("true")].copy()
    selected = selected[selected["report_date"].dt.year.between(2024, 2025)]
    selected["aqi"] = pd.to_numeric(selected["aqi"], errors="coerce")
    selected = selected.dropna(subset=["aqi"]).sort_values("report_date")
    selected["year"] = selected["report_date"].dt.year
    selected["month"] = selected["report_date"].dt.month
    selected["month_start"] = selected["report_date"].dt.to_period("M").dt.to_timestamp()
    selected["season"] = selected["month"].map(season_for_month)
    return selected


def daily_burden_tables(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    yearly_rows: list[dict[str, Any]] = []
    for year, group in daily.groupby("year"):
        expected = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365
        yearly_rows.append(
            {
                "year": year,
                "reported_days": len(group),
                "calendar_days": expected,
                "coverage_pct": 100 * len(group) / expected,
                "mean_aqi": group["aqi"].mean(),
                "median_aqi": group["aqi"].median(),
                "days_aqi_gt_100": int(group["aqi"].gt(100).sum()),
                "days_aqi_gt_150": int(group["aqi"].gt(150).sum()),
                "days_aqi_gt_200": int(group["aqi"].gt(200).sum()),
                "share_gt_150_pct": 100 * group["aqi"].gt(150).mean(),
                "maximum_aqi": group["aqi"].max(),
            }
        )

    monthly_rows: list[dict[str, Any]] = []
    for month, group in daily.groupby("month_start"):
        n = len(group)
        for threshold in (100, 150, 200):
            count = int(group["aqi"].gt(threshold).sum())
            low, high = proportion_confint(count, n, alpha=0.05, method="wilson")
            monthly_rows.append(
                {
                    "month_start": month,
                    "threshold": threshold,
                    "reported_days": n,
                    "days_above_threshold": count,
                    "share_pct": 100 * count / n,
                    "wilson_ci_low_pct": 100 * low,
                    "wilson_ci_high_pct": 100 * high,
                }
            )

    seasonal_rows: list[dict[str, Any]] = []
    for season in SEASONS:
        group = daily[daily["season"].eq(season)]
        seasonal_rows.append(
            {
                "season": season,
                "reported_days": len(group),
                "mean_aqi": group["aqi"].mean(),
                "median_aqi": group["aqi"].median(),
                "share_gt_100_pct": 100 * group["aqi"].gt(100).mean(),
                "share_gt_150_pct": 100 * group["aqi"].gt(150).mean(),
                "share_gt_200_pct": 100 * group["aqi"].gt(200).mean(),
            }
        )
    return {
        "daily_year": pd.DataFrame(yearly_rows),
        "monthly_exceedance": pd.DataFrame(monthly_rows),
        "daily_season": pd.DataFrame(seasonal_rows),
    }


def episode_table(daily: pd.DataFrame, threshold: float = 150) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current: list[pd.Series] = []
    previous_date: pd.Timestamp | None = None
    previous_high = False
    for _, row in daily.iterrows():
        date = row["report_date"]
        high = row["aqi"] > threshold
        consecutive = previous_date is not None and date == previous_date + pd.Timedelta(days=1)
        if high and (not previous_high or not consecutive):
            if current:
                rows.append(_summarize_episode(current, threshold))
            current = [row]
        elif high:
            current.append(row)
        elif current:
            rows.append(_summarize_episode(current, threshold))
            current = []
        previous_date = date
        previous_high = high
    if current:
        rows.append(_summarize_episode(current, threshold))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.insert(0, "episode_id", np.arange(1, len(frame) + 1))
    return frame


def _summarize_episode(rows: list[pd.Series], threshold: float) -> dict[str, Any]:
    values = np.array([row["aqi"] for row in rows], dtype=float)
    start = rows[0]["report_date"]
    end = rows[-1]["report_date"]
    return {
        "start_date": start.date(),
        "end_date": end.date(),
        "season": season_for_month(start.month),
        "duration_days": len(rows),
        "mean_aqi": values.mean(),
        "peak_aqi": values.max(),
        "cumulative_aqi_above_150": np.maximum(values - threshold, 0).sum(),
        "definition": "consecutive reported calendar days with AQI > 150; missing days break episodes",
    }


def normalize_station(value: str) -> str:
    compact = "".join(character for character in str(value).lower() if character.isalnum())
    if "barc" in compact:
        return "BARC"
    if compact in {"doe", "departmentofenvironment"}:
        return "DoE"
    if "darus" in compact:
        return "Darus-salam"
    return str(value).strip()


def station_sensitivity_series(monthly: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.copy()
    frame["report_month"] = pd.to_datetime(frame["report_month"])
    frame = frame[
        frame["report_month"].dt.year.between(2022, 2025)
        & frame["parameter"].isin(["PM2.5", "PM10"])
    ].copy()
    frame["station"] = frame["station_label_as_reported"].map(normalize_station)
    statistic = frame["statistic_as_reported"].str.lower().str.replace(" ", "", regex=False)
    averages = frame[statistic.eq("average")][
        ["report_month", "parameter", "station", "value"]
    ].rename(columns={"value": "station_average"})
    capture = frame[statistic.str.contains("datacapture", na=False)][
        ["report_month", "parameter", "station", "value"]
    ].rename(columns={"value": "capture_pct"})
    merged = averages.merge(capture, on=["report_month", "parameter", "station"], how="left")

    rows: list[dict[str, Any]] = []
    for (month, parameter), group in merged.groupby(["report_month", "parameter"]):
        valid_weights = group.dropna(subset=["capture_pct"])
        weighted = (
            np.average(valid_weights["station_average"], weights=valid_weights["capture_pct"])
            if not valid_weights.empty and valid_weights["capture_pct"].sum() > 0
            else np.nan
        )
        barc = group.loc[group["station"].eq("BARC"), "station_average"]
        rows.append(
            {
                "month_start": month,
                "variable": "pm25" if parameter == "PM2.5" else "pm10",
                "unweighted_station_mean": group["station_average"].mean(),
                "capture_weighted_mean": weighted,
                "barc_fixed_station_mean": barc.mean() if not barc.empty else np.nan,
                "station_count": group["station"].nunique(),
                "mean_capture_pct": group["capture_pct"].mean(),
            }
        )
    return pd.DataFrame(rows)


def adjusted_hac_trend(
    dates: pd.Series, values: pd.Series, start_year: int = 2022
) -> dict[str, float]:
    frame = pd.DataFrame({"date": dates, "value": values}).dropna()
    frame = frame[frame["date"].dt.year.ge(start_year)]
    frame["time_years"] = (frame["date"] - frame["date"].min()).dt.days / 365.25
    dummies = pd.get_dummies(frame["date"].dt.month.astype(str), drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([frame[["time_years"]].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1))
    model = sm.OLS(frame["value"].to_numpy(), design).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    ci = model.conf_int().loc["time_years"]
    return {
        "n_months": len(frame),
        "slope_per_year": model.params["time_years"],
        "ci_low": ci.iloc[0],
        "ci_high": ci.iloc[1],
        "p_value": model.pvalues["time_years"],
    }


def theil_sen_trend(dates: pd.Series, values: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"date": dates, "value": values}).dropna()
    frame = frame[frame["date"].dt.year.between(2022, 2025)].copy()
    frame["anomaly"] = frame["value"] - frame.groupby(frame["date"].dt.month)["value"].transform("mean")
    time_years = (frame["date"] - frame["date"].min()).dt.days / 365.25
    slope, _, low, high = stats.theilslopes(frame["anomaly"], time_years, alpha=0.95)
    return {"n_months": len(frame), "slope_per_year": slope, "ci_low": low, "ci_high": high, "p_value": np.nan}


def trend_sensitivity_table(wide: pd.DataFrame, station_series: pd.DataFrame) -> pd.DataFrame:
    recent = wide[wide["year"].between(2022, 2025)]
    rows: list[dict[str, Any]] = []
    for key, (column, _, unit) in CORE_VARIABLES.items():
        specifications = [
            ("month-adjusted HAC, 2022-2025", adjusted_hac_trend(recent["month_start"], recent[column], 2022)),
            ("month-adjusted HAC, 2023-2025", adjusted_hac_trend(recent["month_start"], recent[column], 2023)),
            ("Theil-Sen on monthly anomalies", theil_sen_trend(recent["month_start"], recent[column])),
        ]
        for specification, result in specifications:
            rows.append({"variable": key, "series": "city monthly mean", "specification": specification, "unit_per_year": unit, **result})

        if key in {"pm25", "pm10"}:
            subset = station_series[station_series["variable"].eq(key)]
            for series_column, series_label in [
                ("capture_weighted_mean", "capture-weighted stations"),
                ("barc_fixed_station_mean", "BARC fixed station"),
            ]:
                result = adjusted_hac_trend(subset["month_start"], subset[series_column], 2022)
                rows.append(
                    {
                        "variable": key,
                        "series": series_label,
                        "specification": "month-adjusted HAC, 2022-2025",
                        "unit_per_year": unit,
                        **result,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["q_value_bh"] = fdr(frame["p_value"])
    return frame


def data_quality_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, (column, _, _) in CORE_VARIABLES.items():
        available = wide.loc[wide[column].notna(), "month_start"]
        rows.append(
            {
                "variable": key,
                "available_months": len(available),
                "first_month": available.min().date(),
                "last_month": available.max().date(),
                "complete_months_2022_2025": wide.loc[wide["year"].between(2022, 2025), column].notna().sum(),
                "expected_months_2022_2025": 48,
            }
        )
    return pd.DataFrame(rows)


def source_qa_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    qa = pd.read_csv(PROCESSED / "doe_qa_issues.csv")
    manifest = pd.read_csv(PROCESSED / "doe_source_manifest.csv")
    qa_summary = (
        qa.groupby(["stage", "severity"], dropna=False).size().rename("issue_count").reset_index()
    )
    manifest_summary = (
        manifest.groupby(["source_kind", "extraction_status", "download_status"], dropna=False)
        .size()
        .rename("report_count")
        .reset_index()
    )
    return qa_summary, manifest_summary


def context_notes(population: pd.DataFrame, worldometer: pd.DataFrame, forest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "UN WUP population",
                "scope": "Bangladesh national",
                "coverage": f"{population.year.min()}-{population.year.max()}",
                "research_role": "descriptive context",
                "interpretation_limit": "Not a Dhaka exposure denominator and not entered into pollution models.",
            },
            {
                "dataset": "Worldometer population",
                "scope": "Bangladesh national",
                "coverage": f"selected years {worldometer.year.min()}-{worldometer.year.max()}",
                "research_role": "cross-check of UN-derived estimates",
                "interpretation_limit": "Sparse presentation series; not an independent population source or causal predictor.",
            },
            {
                "dataset": "Global Forest Watch tree-cover loss",
                "scope": "Bangladesh national",
                "coverage": f"{forest.year.min()}-{forest.year.max()}",
                "research_role": "descriptive environmental context",
                "interpretation_limit": "Tree-cover loss is not necessarily permanent deforestation and is spatially mismatched to Dhaka AQI.",
            },
        ]
    )


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def figure_coverage(wide: pd.DataFrame) -> None:
    matrix = np.vstack([wide[column].notna().to_numpy() for column, _, _ in CORE_VARIABLES.values()])
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), gridspec_kw={"height_ratios": [1, 1.6]}, constrained_layout=True)
    axes[0].imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_yticks(range(3), ["AQI", r"PM$_{2.5}$", r"PM$_{10}$"])
    year_positions = wide.groupby("year").head(1).index.to_numpy()
    axes[0].set_xticks(year_positions, wide.loc[year_positions, "year"].astype(str), rotation=45)
    axes[0].set_title("A. Official monthly data availability")
    axes[0].set_xlabel("Blue = reported; white = unavailable")

    recent = wide[wide["year"].between(2022, 2025)]
    for key, label, color in [("pm25", r"PM$_{2.5}$", COLORS["pm25"]), ("pm10", r"PM$_{10}$", COLORS["pm10"])]:
        axes[1].plot(recent["month_start"], recent[f"{key}_station_count"], color=color, label=f"{label} stations")
    axes[1].set_ylabel("Reporting stations")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("B. Monitoring support for the comparable 2022–2025 period")
    second = axes[1].twinx()
    second.plot(recent["month_start"], recent["pm25_mean_data_capture_pct"], color="#2166AC", alpha=0.7, linestyle="--", label=r"PM$_{2.5}$ capture")
    second.set_ylabel("Mean data capture (%)")
    second.set_ylim(0, 105)
    lines = axes[1].lines + second.lines
    axes[1].legend(lines, [line.get_label() for line in lines], loc="lower left", ncol=3, frameon=False)
    fig.suptitle("Figure 1. Coverage and monitoring support in the official DoE record", fontweight="bold")
    fig.savefig(FIGURES / "figure_1_coverage_and_monitoring.png", dpi=240)
    plt.close(fig)


def figure_daily_burden(daily: pd.DataFrame, monthly_exceedance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.1, 1]}, constrained_layout=True)
    bands = [(0, 50, "#00A651"), (50, 100, "#FFF200"), (100, 150, "#F7941D"), (150, 200, "#ED1C24"), (200, 300, "#92278F"), (300, max(500, daily.aqi.max() + 20), "#7E0023")]
    for low, high, color in bands:
        axes[0].axhspan(low, high, color=color, alpha=0.09, linewidth=0)
    axes[0].plot(daily["report_date"], daily["aqi"], color="#777777", linewidth=0.7, alpha=0.65, label="Daily AQI")
    rolling = daily.set_index("report_date")["aqi"].rolling("30D", min_periods=15).mean()
    axes[0].plot(rolling.index, rolling, color="#111111", linewidth=2, label="30-day mean")
    axes[0].axhline(150, color="#B2182B", linestyle="--", linewidth=1)
    axes[0].set_ylabel("AQI")
    axes[0].set_title("A. Daily observations and 30-day burden")
    axes[0].legend(frameon=False, ncol=2)

    burden = monthly_exceedance[monthly_exceedance["threshold"].eq(150)]
    yerr = np.maximum(
        np.vstack(
            [
                burden["share_pct"] - burden["wilson_ci_low_pct"],
                burden["wilson_ci_high_pct"] - burden["share_pct"],
            ]
        ),
        0,
    )
    axes[1].errorbar(burden["month_start"], burden["share_pct"], yerr=yerr, color=COLORS["aqi"], marker="o", markersize=3, linewidth=1.2, capsize=2)
    axes[1].set_ylabel("Reported days\nwith AQI > 150 (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("B. Monthly high-pollution frequency with 95% Wilson intervals")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle("Figure 2. Daily AQI burden in Dhaka, 2024–2025", fontweight="bold")
    fig.savefig(FIGURES / "figure_2_daily_aqi_burden.png", dpi=240)
    plt.close(fig)


def figure_seasonal(wide: pd.DataFrame) -> None:
    recent = wide[wide["year"].between(2022, 2025)]
    rng = np.random.default_rng(20260806)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for axis, (key, (column, label, unit)) in zip(axes, CORE_VARIABLES.items(), strict=True):
        values = [recent.loc[recent["season"].eq(season), column].dropna().to_numpy() for season in SEASONS]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.58, showfliers=False)
        for box in boxes["boxes"]:
            box.set(facecolor=COLORS[key], alpha=0.28, edgecolor=COLORS[key])
        for index, group in enumerate(values, start=1):
            axis.scatter(rng.normal(index, 0.045, len(group)), group, s=15, alpha=0.7, color=COLORS[key], edgecolor="white", linewidth=0.25)
        axis.set_xticks(range(1, 5), ["Winter", "Pre-\nmonsoon", "Monsoon", "Post-\nmonsoon"])
        axis.set_title(label)
        axis.set_ylabel(unit)
    fig.suptitle("Figure 3. Seasonal distributions of monthly AQI and particulate matter, 2022–2025", fontweight="bold")
    fig.savefig(FIGURES / "figure_3_seasonal_particulate_burden.png", dpi=240)
    plt.close(fig)


def figure_episodes(episodes: pd.DataFrame, daily_season: pd.DataFrame) -> None:
    top = episodes.nlargest(12, ["duration_days", "peak_aqi"]).sort_values("duration_days")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True)
    labels = [f"{row.start_date:%d %b %Y}" for row in top.itertuples()]
    scatter = axes[0].scatter(top["duration_days"], np.arange(len(top)), c=top["peak_aqi"], cmap="magma", s=60, zorder=3)
    axes[0].hlines(np.arange(len(top)), 0, top["duration_days"], color="#999999", linewidth=1)
    axes[0].set_yticks(np.arange(len(top)), labels)
    axes[0].set_xlabel("Consecutive reported days with AQI > 150")
    axes[0].set_title("A. Twelve longest high-pollution episodes")
    fig.colorbar(scatter, ax=axes[0], label="Episode peak AQI", shrink=0.75)

    x = np.arange(len(SEASONS))
    width = 0.24
    for offset, threshold, color in [(-width, 100, "#F7941D"), (0, 150, "#ED1C24"), (width, 200, "#92278F")]:
        axes[1].bar(x + offset, daily_season[f"share_gt_{threshold}_pct"], width, label=f"> {threshold}", color=color, alpha=0.85)
    axes[1].set_xticks(x, ["Winter", "Pre-\nmonsoon", "Monsoon", "Post-\nmonsoon"])
    axes[1].set_ylabel("Reported days above threshold (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("B. Seasonal threshold burden")
    axes[1].legend(title="AQI", frameon=False, ncol=3, loc="upper center")
    fig.suptitle("Figure 4. Persistence and seasonal concentration of high-AQI episodes, 2024–2025", fontweight="bold")
    fig.savefig(FIGURES / "figure_4_pollution_episodes.png", dpi=240)
    plt.close(fig)


def figure_trends(trends: pd.DataFrame) -> None:
    labels = {"aqi": "AQI", "pm25": r"PM$_{2.5}$", "pm10": r"PM$_{10}$"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.7), constrained_layout=True)
    for axis, variable in zip(axes, CORE_VARIABLES, strict=True):
        subset = trends[trends["variable"].eq(variable)].copy().reset_index(drop=True)
        subset["plot_label"] = subset["specification"].str.replace("month-adjusted HAC, ", "HAC ", regex=False)
        subset["plot_label"] = subset["plot_label"].replace(
            {"Theil-Sen on monthly anomalies": "Theil-Sen anomalies"}
        )
        subset.loc[subset["series"].ne("city monthly mean"), "plot_label"] = subset.loc[subset["series"].ne("city monthly mean"), "series"]
        subset["plot_label"] = subset["plot_label"].replace(
            {
                "capture-weighted stations": "Capture-weighted",
                "BARC fixed station": "BARC fixed station",
            }
        )
        y = np.arange(len(subset))[::-1]
        axis.axvline(0, color="#555555", linewidth=1)
        axis.errorbar(
            subset["slope_per_year"],
            y,
            xerr=np.vstack([subset["slope_per_year"] - subset["ci_low"], subset["ci_high"] - subset["slope_per_year"]]),
            fmt="o",
            color=COLORS[variable],
            capsize=3,
        )
        axis.set_yticks(y, subset["plot_label"])
        axis.set_title(labels[variable])
        axis.set_xlabel("Estimated change per year (95% CI)")
    fig.suptitle("Figure 5. Trend estimates across temporal and monitoring-sensitivity specifications", fontweight="bold")
    fig.savefig(FIGURES / "figure_5_trend_sensitivity.png", dpi=240)
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
    if sheet.max_row > 1:
        sheet.auto_filter.ref = sheet.dimensions
    for index, column in enumerate(frame.columns, start=1):
        width = min(max(len(str(column)) + 2, 12), 45)
        sheet.column_dimensions[get_column_letter(index)].width = width


def write_analysis_workbook(tables: dict[str, pd.DataFrame]) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    readme = pd.DataFrame(
        [
            ("Purpose", "Paper-ready analysis of official DoE Dhaka observations."),
            ("Primary monthly period", "2022-2025; 2026 is partial and excluded from estimates."),
            ("Primary daily period", "2024-2025 selected daily records."),
            ("Core outcomes", "AQI, PM2.5, and PM10; gases remain in the source workbook because recent units are unresolved."),
            ("Episodes", "Consecutive reported calendar days with AQI > 150; missing dates break episodes."),
            ("Trends", "Month-adjusted OLS with HAC errors, exclude-2022 sensitivity, Theil-Sen anomalies, capture weighting, and fixed-station checks."),
            ("Context", "Population and tree-cover loss are national descriptive context only; no ecological causal correlations are fitted."),
            ("Figures", "Five main figures; superseded exploratory graphics were removed."),
        ],
        columns=["item", "detail"],
    )
    write_frame(workbook, "read_me", readme)
    for name, frame in tables.items():
        write_frame(workbook, name[:31], frame)
    workbook.save(OUTPUT / "dhaka_doe_analysis.xlsx")


def build_report(
    descriptive: pd.DataFrame,
    seasonal: pd.DataFrame,
    season_tests: pd.DataFrame,
    daily_tables: dict[str, pd.DataFrame],
    episodes: pd.DataFrame,
    trends: pd.DataFrame,
    forest: pd.DataFrame,
) -> str:
    winter = seasonal[seasonal["season"].eq("Winter")].set_index("variable")
    monsoon = seasonal[seasonal["season"].eq("Monsoon")].set_index("variable")
    years = daily_tables["daily_year"].set_index("year")
    longest = episodes.sort_values(["duration_days", "peak_aqi"], ascending=False).iloc[0]
    trend_core = trends[(trends["series"].eq("city monthly mean")) & trends["specification"].eq("month-adjusted HAC, 2022-2025")].set_index("variable")
    lines = [
        "# Research findings: official Dhaka air-quality burden",
        "",
        "## Study focus",
        "",
        "The defensible contribution is an official-source assessment of burden, seasonality, episode persistence, and trend robustness. The primary monthly window is 2022–2025; the primary daily window is 2024–2025. Partial 2026 observations are retained in the dataset but excluded from estimates.",
        "",
        "## Central results",
        "",
        f"- Winter monthly mean AQI was **{winter.loc['aqi', 'mean']:.1f}**, compared with **{monsoon.loc['aqi', 'mean']:.1f}** during monsoon.",
        f"- Winter mean PM2.5 was **{winter.loc['pm25', 'mean']:.1f} µg/m³**, compared with **{monsoon.loc['pm25', 'mean']:.1f} µg/m³** during monsoon.",
        f"- AQI exceeded 150 on **{years.loc[2024, 'share_gt_150_pct']:.1f}%** of reported days in 2024 and **{years.loc[2025, 'share_gt_150_pct']:.1f}%** in 2025.",
        f"- The longest observed high-pollution episode lasted **{int(longest['duration_days'])} consecutive reported days**, from **{longest['start_date']}** through **{longest['end_date']}**, and peaked at AQI **{longest['peak_aqi']:.0f}**.",
        "",
        "## Trend robustness",
        "",
        *[
            f"- {variable.upper()}: {row.slope_per_year:.2f} per year (95% CI {row.ci_low:.2f} to {row.ci_high:.2f}; HAC month-adjusted model)."
            for variable, row in trend_core.iterrows()
        ],
        "",
        "Trend direction and magnitude should be judged across the full sensitivity table, not from one p-value. It includes an exclusion of 2022, a robust Theil-Sen estimator, capture-weighted station averages, and the near-complete BARC fixed-station series.",
        "",
        "## Statistical evidence",
        "",
        *[
            f"- {row.variable.upper()}: Kruskal–Wallis H={row.statistic:.2f}, q={row.q_value_bh:.3g}."
            for row in season_tests.itertuples()
        ],
        "",
        "## Population and forest context",
        "",
        f"Bangladesh recorded {forest.loc[forest.year.between(2022, 2024), 'tree_cover_loss_ha'].sum():,.0f} hectares of tree-cover loss across 2022–2024 in the Global Forest Watch-derived series. This is national context, not a Dhaka attribution result: tree-cover loss includes temporary and permanent stand-replacement disturbances and is not synonymous with deforestation. Worldometer population estimates are retained only as a sparse cross-check of their underlying UN series. Neither context series is entered into the air-quality models because geography and temporal resolution do not match Dhaka exposure observations.",
        "",
        "## Reproducible outputs",
        "",
        "- `analysis/dhaka_doe_analysis.xlsx`: result tables, sensitivity estimates, episodes, QA summaries, and context data.",
        "- `analysis/figures/`: five main paper figures.",
        "- `scripts/analyze_doe_dataset.py`: complete analysis code.",
        "- `data/processed/dhaka_doe_air_quality.xlsx`: source observations and provenance.",
        "",
        "## Interpretation boundary",
        "",
        "The record supports temporal description and association, not source apportionment or causal effects. Changing station composition, unresolved units for some gases, gaps in the public monthly archive, and the short comparable period remain material limitations.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    wide = pd.read_csv(PROCESSED / "doe_monthly_dataset_wide.csv", parse_dates=["month_start"])
    daily_raw = pd.read_csv(PROCESSED / "doe_daily_dhaka_aqi.csv", parse_dates=["report_date"])
    monthly = pd.read_csv(PROCESSED / "doe_monthly_dhaka.csv")
    population = pd.read_csv(CONTEXT / "bangladesh_population.csv")
    worldometer = pd.read_csv(CONTEXT / "bangladesh_population_worldometer.csv")
    forest = pd.read_csv(CONTEXT / "bangladesh_tree_cover_loss.csv")

    descriptive = descriptive_table(wide)
    seasonal, season_tests = seasonal_tables(wide)
    daily = prepare_daily(daily_raw)
    daily_tables = daily_burden_tables(daily)
    episodes = episode_table(daily)
    station_series = station_sensitivity_series(monthly)
    trends = trend_sensitivity_table(wide, station_series)
    qa_summary, manifest_summary = source_qa_tables()

    tables = {
        "descriptive_main": descriptive,
        "seasonal_main": seasonal,
        "season_tests": season_tests,
        **daily_tables,
        "episodes_gt_150": episodes,
        "trend_sensitivity": trends,
        "station_sensitivity": station_series,
        "data_quality": data_quality_table(wide),
        "qa_summary": qa_summary,
        "manifest_summary": manifest_summary,
        "context_notes": context_notes(population, worldometer, forest),
        "population_un": population,
        "population_worldometer": worldometer,
        "tree_cover_loss": forest,
    }

    figure_coverage(wide)
    figure_daily_burden(daily, daily_tables["monthly_exceedance"])
    figure_seasonal(wide)
    figure_episodes(episodes, daily_tables["daily_season"])
    figure_trends(trends)
    write_analysis_workbook(tables)
    report = build_report(descriptive, seasonal, season_tests, daily_tables, episodes, trends, forest)
    (OUTPUT / "RESEARCH_FINDINGS.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'dhaka_doe_analysis.xlsx'}")
    print(f"Wrote {OUTPUT / 'RESEARCH_FINDINGS.md'}")
    print("Wrote five focused paper figures")


if __name__ == "__main__":
    main()
