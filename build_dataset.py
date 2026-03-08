#!/usr/bin/env python3
"""
Dhaka Air Quality Data Processing Pipeline
===========================================

Reads two source Excel files, inspects their schemas, merges relevant data,
cleans inconsistencies, and outputs a final tidy dataset suitable for
statistical analysis and paper writing.

Source files
------------
1. dhaka_aqi_monthly_enriched_v2_2017_2025_filled.xlsx
   - Monthly_Enriched  : 108 rows, monthly AQI/PM2.5 + population/HDI/poverty (2017-2025)
   - Annual_Context    : 26 rows, yearly demographic/development indicators (2000-2025)
   - Web_Scraped_AQI_PM25_Evidence : 5 rows, web evidence citations
   - Methodology       : 7 rows, methodology notes

2. dhaka_observed_air_quality_dataset.xlsx
   - Daily_Observed    : 2 192 rows, daily PM2.5 & AQI (2016-01 to 2021-12)
   - Hourly_Raw        : 44 990 rows, hourly PM2.5/AQI (2016-2021)
   - Monthly_Meteo_Norm: 72 rows, monthly wind/rain normals (2016-2021)
   - Yearly_Summary    : 14 rows, yearly aggregation
   - Sources / README  : documentation sheets

Merge decisions
---------------
* File 1 Monthly_Enriched is the **primary** source for the final monthly
  dataset because it spans 2017-01 through 2025-12 and already carries
  population, HDI, and poverty context merged at the annual level.
* File 2 provides *observed* daily/hourly granularity for 2016-2021. We
  aggregate it to monthly level and use it to **validate** (not replace)
  File 1 values for the overlap period 2017-01 to 2021-12.
* Meteorological normals (norm_wind, norm_rain) from File 2 are merged into
  the final dataset for the months they cover (2016-2021).
* The source_notes column records provenance and any inferred/filled values.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FILE1 = BASE_DIR / "dhaka_aqi_monthly_enriched_v2_2017_2025_filled.xlsx"
FILE2 = BASE_DIR / "dhaka_observed_air_quality_dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"

CORE_START_YEAR = 2017
CORE_END_YEAR = 2025

REQUIRED_COLUMNS = [
    "month_start", "year", "month", "month_name",
    "aqi_mean", "aqi_median", "aqi_min", "aqi_max",
    "pm25_mean", "pm25_median", "pm25_min", "pm25_max",
    "hourly_observations", "expected_hours", "coverage_pct",
    "is_partial_month",
    "population_total", "urban_population", "rural_population",
    "urban_share_pct", "rural_share_pct",
    "hdi", "poverty_rate_pct",
    "source_notes",
]

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}


# ---------------------------------------------------------------------------
# 1. load_workbook_safely
# ---------------------------------------------------------------------------
def load_workbook_safely(path: str | Path) -> pd.ExcelFile:
    """Open an Excel file with openpyxl and return a pd.ExcelFile handle.

    Raises FileNotFoundError with a clear message if the file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected Excel file not found: {path}\n"
            f"Please place the file in {path.parent}"
        )
    return pd.ExcelFile(path, engine="openpyxl")


# ---------------------------------------------------------------------------
# 2. inspect_excel_file
# ---------------------------------------------------------------------------
def inspect_excel_file(xl: pd.ExcelFile, label: str) -> dict[str, pd.DataFrame]:
    """Read every sheet, print schema info, and return a dict of DataFrames.

    Parameters
    ----------
    xl : pd.ExcelFile
    label : str – human-readable label for console output

    Returns
    -------
    dict mapping sheet_name -> DataFrame
    """
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  Sheets: {xl.sheet_names}")
    print(f"{'=' * 72}")

    sheets: dict[str, pd.DataFrame] = {}
    for name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=name)
        sheets[name] = df
        print(f"\n--- {name} ---")
        print(f"  Rows : {len(df)}")
        print(f"  Cols : {len(df.columns)}")
        print(f"  Columns & dtypes:")
        for col in df.columns:
            missing = df[col].isnull().sum()
            print(f"    {col:40s} {str(df[col].dtype):20s} missing={missing}")
    return sheets


# ---------------------------------------------------------------------------
# 3. normalize_columns
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case and strip whitespace."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return df


# ---------------------------------------------------------------------------
# 4. clean_monthly_dataset
# ---------------------------------------------------------------------------
def clean_monthly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the primary monthly enriched dataset.

    Steps
    -----
    1. Normalize column names.
    2. Ensure month_start is datetime; coerce bad values to NaT.
    3. Filter to CORE_START_YEAR..CORE_END_YEAR.
    4. Remove exact duplicate rows.
    5. Detect impossible values and report them.
    6. Build source_notes from existing notes/context columns.
    7. Flag partial months.
    """
    df = normalize_columns(df)

    # -- datetime -----------------------------------------------------------
    df["month_start"] = pd.to_datetime(df["month_start"], errors="coerce")
    bad_dates = df["month_start"].isnull().sum()
    if bad_dates:
        print(f"  WARNING: {bad_dates} rows have unparseable month_start – dropped")
        df = df.dropna(subset=["month_start"])

    # -- core period --------------------------------------------------------
    df = df[(df["year"] >= CORE_START_YEAR) & (df["year"] <= CORE_END_YEAR)].copy()

    # -- duplicates ---------------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"  Removed {n_dupes} exact duplicate rows")

    # -- impossible-value checks --------------------------------------------
    _report_impossible(df, "pm25_mean", 0, 1000)
    _report_impossible(df, "pm25_max", 0, 2000)
    _report_impossible(df, "aqi_mean", 0, 500)
    _report_impossible(df, "aqi_max", 0, 500)
    _report_impossible(df, "coverage_pct", 0, 100)
    _report_impossible(df, "hdi", 0, 1)
    _report_impossible(df, "poverty_rate_pct", 0, 100)

    # -- source_notes -------------------------------------------------------
    # Combine existing context columns into a single source_notes field.
    note_cols = [c for c in ("notes", "web_sources", "context_note", "context_source") if c in df.columns]
    if note_cols:
        df["source_notes"] = df[note_cols].apply(
            lambda row: " | ".join(
                str(v) for v in row if pd.notna(v) and str(v).strip()
            ),
            axis=1,
        )
    else:
        df["source_notes"] = ""

    # Drop the original note columns that were merged into source_notes
    df = df.drop(columns=[c for c in note_cols if c in df.columns], errors="ignore")

    # Also drop source_city – constant for Dhaka
    df = df.drop(columns=["source_city"], errors="ignore")

    # -- ensure is_partial_month is bool ------------------------------------
    df["is_partial_month"] = df["is_partial_month"].astype(bool)

    return df.reset_index(drop=True)


def _report_impossible(df: pd.DataFrame, col: str, lo: float, hi: float) -> None:
    """Print a warning for values outside [lo, hi] without silently changing them."""
    if col not in df.columns:
        return
    mask = (df[col] < lo) | (df[col] > hi)
    n = mask.sum()
    if n:
        print(f"  ⚠ {col}: {n} value(s) outside [{lo}, {hi}]:")
        print(df.loc[mask, ["month_start", col]].to_string(index=False))


# ---------------------------------------------------------------------------
# 5. merge_context_variables
# ---------------------------------------------------------------------------
def merge_context_variables(
    monthly: pd.DataFrame,
    meteo: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge meteorological normals from File 2 into the monthly dataset.

    Merge decision:
    * Population, HDI, poverty are ALREADY in the monthly enriched file
      (from Annual_Context). We keep those as-is.
    * norm_wind and norm_rain from File 2 Monthly_Meteo_Norm are added
      for months where they exist (2016-2021 → we only keep 2017-2021).
    """
    if meteo is None or meteo.empty:
        monthly["norm_wind"] = np.nan
        monthly["norm_rain"] = np.nan
        return monthly

    meteo = normalize_columns(meteo.copy())
    # Rename 'date' to avoid collision; we join on year+month
    meteo = meteo.rename(columns={"date": "meteo_date"})
    meteo = meteo[["year", "month", "norm_wind", "norm_rain"]].copy()
    meteo = meteo[
        (meteo["year"] >= CORE_START_YEAR) & (meteo["year"] <= CORE_END_YEAR)
    ]

    monthly = monthly.merge(meteo, on=["year", "month"], how="left")
    return monthly


# ---------------------------------------------------------------------------
# 6. validate_final_dataset
# ---------------------------------------------------------------------------
def validate_final_dataset(df: pd.DataFrame) -> dict:
    """Run validation checks and return a QA info dict."""
    qa: dict = {}

    # -- one row per month --------------------------------------------------
    ym = df[["year", "month"]].drop_duplicates()
    if len(ym) != len(df):
        dupes = df[df.duplicated(subset=["year", "month"], keep=False)]
        print(f"  ⚠ Duplicate year-month combos found:\n{dupes[['year', 'month']]}")
    qa["unique_year_months"] = len(ym)
    qa["total_rows"] = len(df)

    # -- expected span ------------------------------------------------------
    expected_months = set()
    for y in range(CORE_START_YEAR, CORE_END_YEAR + 1):
        for m in range(1, 13):
            expected_months.add((y, m))
    actual_months = set(zip(df["year"], df["month"]))
    missing = sorted(expected_months - actual_months)
    qa["expected_months"] = len(expected_months)
    qa["actual_months"] = len(actual_months)
    qa["missing_months"] = missing

    # -- missingness --------------------------------------------------------
    qa["missingness"] = df.isnull().sum().to_dict()
    qa["total_columns"] = len(df.columns)

    # -- date span ----------------------------------------------------------
    qa["date_min"] = str(df["month_start"].min().date())
    qa["date_max"] = str(df["month_start"].max().date())

    # -- partial months -----------------------------------------------------
    qa["partial_months"] = int(df["is_partial_month"].sum())

    return qa


# ---------------------------------------------------------------------------
# 7. export_outputs
# ---------------------------------------------------------------------------
def export_outputs(
    final: pd.DataFrame,
    qa: dict,
    data_dict_rows: list[dict],
    output_dir: Path,
) -> None:
    """Write all output artefacts into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- final dataset CSV & Excel ------------------------------------------
    final.to_csv(output_dir / "final_dhaka_aqi_dataset.csv", index=False)
    final.to_excel(output_dir / "final_dhaka_aqi_dataset.xlsx", index=False, engine="openpyxl")

    # -- data dictionary ----------------------------------------------------
    dd = pd.DataFrame(data_dict_rows)
    dd.to_csv(output_dir / "data_dictionary.csv", index=False)

    # -- QA report ----------------------------------------------------------
    _write_qa_report(final, qa, output_dir / "qa_report.md")

    # -- analysis-ready tables ----------------------------------------------
    _write_annual_summary(final, output_dir)
    _write_seasonality_summary(final, output_dir)
    _write_correlation_ready(final, output_dir)

    print(f"\n✅ All outputs written to {output_dir}/")


# -- helpers for export_outputs ---------------------------------------------

def _write_qa_report(df: pd.DataFrame, qa: dict, path: Path) -> None:
    lines = [
        "# QA Report – Dhaka AQI Final Dataset",
        "",
        "## Overview",
        f"- **Total rows**: {qa['total_rows']}",
        f"- **Total columns**: {qa['total_columns']}",
        f"- **Date span**: {qa['date_min']} to {qa['date_max']}",
        f"- **Expected months (2017-01 to 2025-12)**: {qa['expected_months']}",
        f"- **Actual months present**: {qa['actual_months']}",
        f"- **Months with partial coverage**: {qa['partial_months']}",
        "",
    ]

    if qa["missing_months"]:
        lines.append("### Missing months")
        for y, m in qa["missing_months"]:
            lines.append(f"- {y}-{m:02d}")
        lines.append("")
    else:
        lines.append("### Missing months")
        lines.append("None – full coverage from 2017-01 to 2025-12.")
        lines.append("")

    lines.append("## Missingness by column")
    lines.append("")
    lines.append("| Column | Missing |")
    lines.append("|--------|---------|")
    for col, n in sorted(qa["missingness"].items()):
        lines.append(f"| {col} | {n} |")
    lines.append("")

    lines.append("## Source file contributions")
    lines.append("")
    lines.append(
        "| Variable group | Authoritative source | Notes |\n"
        "|----------------|----------------------|-------|\n"
        "| Monthly AQI (mean/median/min/max) | File 1 – Monthly_Enriched | Covers 2017-2025; includes inferred values for recent months |\n"
        "| Monthly PM2.5 (mean/median/min/max) | File 1 – Monthly_Enriched | Same as AQI |\n"
        "| Hourly observations / coverage | File 1 – Monthly_Enriched | Derived from hourly data; coverage_pct and is_partial_month indicate completeness |\n"
        "| Population (total, urban, rural) | File 1 – Annual_Context sheet joined to Monthly_Enriched | Yearly values repeated for each month |\n"
        "| HDI | File 1 – Annual_Context | Yearly values |\n"
        "| Poverty rate | File 1 – Annual_Context | Yearly values |\n"
        "| Meteorological normals (norm_wind, norm_rain) | File 2 – Monthly_Meteo_Norm | Available for 2017-2021 only; NaN elsewhere |\n"
        "| Observed daily PM2.5 (used for validation only) | File 2 – Daily_Observed | 2016-2021; not merged into final but used for cross-checks |"
    )
    lines.append("")

    lines.append("## Assumptions and limitations")
    lines.append("")
    lines.append(
        "1. File 1 is treated as the authoritative monthly source because it spans "
        "2017–2025 and was already enriched with population/HDI/poverty data.\n"
        "2. Observed data from File 2 was preferred for validation but not used "
        "to overwrite File 1 values, because File 1 already integrates those "
        "observations for months where they overlap.\n"
        "3. For months beyond the observed period (post-2021), AQI/PM2.5 values "
        "in File 1 may be inferred or web-scraped; the `source_notes` column "
        "documents this.\n"
        "4. HDI and poverty_rate_pct are annual values applied uniformly to all "
        "months within a year.\n"
        "5. norm_wind and norm_rain are only available for 2017-2021.\n"
        "6. December 2025 and other recent months may rely on externally filled "
        "values; see `source_notes` and `is_partial_month`."
    )
    lines.append("")

    lines.append("## Columns: derived vs directly observed")
    lines.append("")
    lines.append("| Column | Type |")
    lines.append("|--------|------|")
    derived = {
        "month_start", "year", "month", "month_name", "coverage_pct",
        "is_partial_month", "urban_share_pct", "rural_share_pct",
        "source_notes", "season",
    }
    for col in df.columns:
        kind = "Derived / computed" if col in derived else "Directly observed / reported"
        lines.append(f"| {col} | {kind} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_annual_summary(df: pd.DataFrame, out: Path) -> None:
    agg = df.groupby("year").agg(
        aqi_mean_annual=("aqi_mean", "mean"),
        aqi_max_annual=("aqi_max", "max"),
        pm25_mean_annual=("pm25_mean", "mean"),
        pm25_max_annual=("pm25_max", "max"),
        total_observations=("hourly_observations", "sum"),
        months_available=("month", "count"),
        partial_months=("is_partial_month", "sum"),
        population_total=("population_total", "first"),
        hdi=("hdi", "first"),
        poverty_rate_pct=("poverty_rate_pct", "first"),
    ).reset_index()
    agg.to_csv(out / "annual_summary.csv", index=False)


def _write_seasonality_summary(df: pd.DataFrame, out: Path) -> None:
    if "season" not in df.columns:
        df = df.copy()
        df["season"] = df["month"].map(SEASON_MAP)
    agg = df.groupby("season").agg(
        aqi_mean=("aqi_mean", "mean"),
        aqi_median=("aqi_median", "median"),
        pm25_mean=("pm25_mean", "mean"),
        pm25_median=("pm25_median", "median"),
        months_counted=("month", "count"),
    ).reset_index()
    agg.to_csv(out / "seasonality_summary.csv", index=False)


def _write_correlation_ready(df: pd.DataFrame, out: Path) -> None:
    """Export a numeric-only dataset suitable for correlation analysis."""
    numeric_cols = [
        "year", "month",
        "aqi_mean", "aqi_median", "aqi_min", "aqi_max",
        "pm25_mean", "pm25_median", "pm25_min", "pm25_max",
        "hourly_observations", "expected_hours", "coverage_pct",
        "population_total", "urban_population", "rural_population",
        "urban_share_pct", "rural_share_pct",
        "hdi", "poverty_rate_pct",
    ]
    # Include meteorological columns if present
    for col in ("norm_wind", "norm_rain"):
        if col in df.columns:
            numeric_cols.append(col)
    existing = [c for c in numeric_cols if c in df.columns]
    df[existing].to_csv(out / "correlation_ready_dataset.csv", index=False)


# ---------------------------------------------------------------------------
# build_data_dictionary
# ---------------------------------------------------------------------------
def build_data_dictionary(df: pd.DataFrame) -> list[dict]:
    """Return a list of dicts describing each column in the final dataset."""
    descriptions = {
        "month_start": "First day of the month (datetime, YYYY-MM-DD)",
        "year": "Calendar year",
        "month": "Calendar month (1-12)",
        "month_name": "English month name",
        "aqi_mean": "Monthly mean AQI",
        "aqi_median": "Monthly median AQI",
        "aqi_min": "Monthly minimum AQI",
        "aqi_max": "Monthly maximum AQI",
        "pm25_mean": "Monthly mean PM2.5 concentration (µg/m³)",
        "pm25_median": "Monthly median PM2.5 (µg/m³)",
        "pm25_min": "Monthly minimum PM2.5 (µg/m³)",
        "pm25_max": "Monthly maximum PM2.5 (µg/m³)",
        "hourly_observations": "Count of hourly observations in the month",
        "expected_hours": "Expected number of hours in the month",
        "coverage_pct": "Observation coverage percentage (0-100)",
        "is_partial_month": "True if coverage is incomplete",
        "population_total": "Total national population (annual, repeated monthly)",
        "urban_population": "Urban population (annual, repeated monthly)",
        "rural_population": "Rural population (annual, repeated monthly)",
        "urban_share_pct": "Urban share of population (%)",
        "rural_share_pct": "Rural share of population (%)",
        "hdi": "Human Development Index (annual, not rounded)",
        "poverty_rate_pct": "National poverty rate (%)",
        "source_notes": "Provenance notes; flags inferred/filled values",
        "norm_wind": "Normalised monthly wind speed (File 2, 2017-2021 only)",
        "norm_rain": "Normalised monthly rainfall (File 2, 2017-2021 only)",
        "season": "Meteorological season (Winter/Pre-monsoon/Monsoon/Post-monsoon)",
    }
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "null_count": int(df[col].isnull().sum()),
            "description": descriptions.get(col, ""),
        })
    return rows


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    print("Dhaka AQI Data Processing Pipeline")
    print("=" * 72)

    # ── Step 1: Load both workbooks ────────────────────────────────────────
    xl1 = load_workbook_safely(FILE1)
    xl2 = load_workbook_safely(FILE2)

    # ── Step 2: Inspect schemas ────────────────────────────────────────────
    sheets1 = inspect_excel_file(xl1, f"File 1: {FILE1.name}")
    sheets2 = inspect_excel_file(xl2, f"File 2: {FILE2.name}")

    # ── Step 3: Validate expected sheets exist ─────────────────────────────
    _require_sheet(sheets1, "Monthly_Enriched", FILE1.name)
    _require_sheet(sheets2, "Monthly_Meteo_Norm", FILE2.name)
    _require_sheet(sheets2, "Daily_Observed", FILE2.name)

    # ── Step 4: Clean primary monthly dataset ──────────────────────────────
    print("\n── Cleaning Monthly_Enriched ──")
    monthly = clean_monthly_dataset(sheets1["Monthly_Enriched"])
    print(f"  Rows after cleaning: {len(monthly)}")

    # ── Step 5: Merge meteorological context from File 2 ───────────────────
    print("\n── Merging meteorological normals from File 2 ──")
    monthly = merge_context_variables(monthly, sheets2.get("Monthly_Meteo_Norm"))
    print(f"  norm_wind non-null: {monthly['norm_wind'].notna().sum()}")
    print(f"  norm_rain non-null: {monthly['norm_rain'].notna().sum()}")

    # ── Step 6: Add season column ──────────────────────────────────────────
    monthly["season"] = monthly["month"].map(SEASON_MAP)

    # ── Step 7: Cross-validate with File 2 daily observed ──────────────────
    print("\n── Cross-validation with Daily_Observed ──")
    _cross_validate(monthly, sheets2["Daily_Observed"])

    # ── Step 8: Ensure all required columns are present ────────────────────
    for col in REQUIRED_COLUMNS:
        if col not in monthly.columns:
            raise ValueError(f"Required column missing from final dataset: {col}")

    # ── Step 9: Sort and final order ───────────────────────────────────────
    monthly = monthly.sort_values("month_start").reset_index(drop=True)

    # ── Step 10: Validate ──────────────────────────────────────────────────
    print("\n── Validation ──")
    qa = validate_final_dataset(monthly)
    print(f"  Total rows          : {qa['total_rows']}")
    print(f"  Total columns       : {qa['total_columns']}")
    print(f"  Date span           : {qa['date_min']} to {qa['date_max']}")
    print(f"  Missing months      : {len(qa['missing_months'])}")
    print(f"  Partial months      : {qa['partial_months']}")

    # ── Step 11: Export ────────────────────────────────────────────────────
    dd = build_data_dictionary(monthly)
    export_outputs(monthly, qa, dd, OUTPUT_DIR)

    print("\n🏁 Pipeline complete.")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _require_sheet(sheets: dict, name: str, file_label: str) -> None:
    if name not in sheets:
        raise KeyError(
            f"Expected sheet '{name}' not found in {file_label}. "
            f"Available sheets: {list(sheets.keys())}"
        )


def _cross_validate(monthly: pd.DataFrame, daily_obs: pd.DataFrame) -> None:
    """Aggregate File 2 daily data to monthly and compare with File 1."""
    daily = normalize_columns(daily_obs.copy())

    # Filter to core period
    daily = daily[(daily["year"] >= CORE_START_YEAR) & (daily["year"] <= CORE_END_YEAR)]

    if daily.empty:
        print("  No overlapping daily data for cross-validation.")
        return

    agg = daily.groupby(["year", "month"]).agg(
        obs_pm25_mean=("daily_mean_pm25", "mean"),
        obs_aqi_mean=("daily_aqi_from_pm25", "mean"),
    ).reset_index()

    merged = monthly[["year", "month", "pm25_mean", "aqi_mean"]].merge(
        agg, on=["year", "month"], how="inner",
    )

    if merged.empty:
        print("  No overlapping months for cross-validation.")
        return

    pm25_corr = merged[["pm25_mean", "obs_pm25_mean"]].dropna().corr().iloc[0, 1]
    aqi_corr = merged[["aqi_mean", "obs_aqi_mean"]].dropna().corr().iloc[0, 1]
    print(f"  Overlapping months: {len(merged)}")
    print(f"  PM2.5 mean correlation (File1 vs File2 daily agg): {pm25_corr:.4f}")
    print(f"  AQI mean correlation   (File1 vs File2 daily agg): {aqi_corr:.4f}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
