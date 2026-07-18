"""Build standardized daily and monthly products from AirNow summaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.aqi import AQI_STANDARD, AQI_VERSION, aqi_category, pm25_aqi
from src.quality.completeness import DEFAULT_DAY_COVERAGE_PCT


SOURCE_ID = "airnow_dhaka_dk1010001"
TIMEZONE = "Asia/Dhaka"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_daily(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, dtype={"station_id": "string", "full_station_id": "string"})
    required = {
        "date_local",
        "station_id",
        "station_name",
        "parameter",
        "unit",
        "value",
        "duration_hours",
        "agency",
        "latitude",
        "longitude",
        "source_url",
        "retrieval_timestamp_utc",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing raw AirNow columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("Raw AirNow input is empty")
    if raw["date_local"].duplicated().any():
        raise ValueError("Duplicate AirNow local dates")
    if set(raw["station_id"].dropna()) != {"DK1010001"}:
        raise ValueError("Unexpected station identity")
    if set(raw["unit"].dropna()) != {"UG/M3"}:
        raise ValueError("Unexpected PM2.5 unit")
    if set(raw["parameter"].dropna()) != {"PM2.5-24hr"}:
        raise ValueError("Unexpected parameter")
    if (raw["value"] < 0).any() or (~np.isfinite(raw["value"])).any():
        raise ValueError("Invalid PM2.5 concentrations")

    local_midnight = pd.to_datetime(raw["date_local"]).dt.tz_localize(TIMEZONE)
    retrieval = pd.to_datetime(raw["retrieval_timestamp_utc"], utc=True)
    if (local_midnight.dt.tz_convert("UTC") > retrieval.max()).any():
        raise ValueError("Observation date exceeds retrieval cutoff")

    daily = pd.DataFrame(
        {
            "timestamp_utc": local_midnight.dt.tz_convert("UTC"),
            "timestamp_local": local_midnight,
            "timezone": TIMEZONE,
            "date_local": local_midnight.dt.date.astype(str),
            "station_id": raw["station_id"],
            "station_name": raw["station_name"],
            "latitude": raw["latitude"].astype(float),
            "longitude": raw["longitude"].astype(float),
            "pollutant": "pm25",
            "value": raw["value"].astype(float),
            "unit": "ug/m3",
            "averaging_period": "24-hour",
            "provider": "US EPA AirNow",
            "original_provider": raw["agency"],
            "instrument": "BAM-1020 (literature-reported; not encoded in archive row)",
            "qa_flag": "AirNow valid preliminary summary; validation status unavailable",
            "validity": "valid_preliminary",
            "source_file": raw["source_url"],
            "source_version": "daily_data_v2.dat",
            "retrieval_date": retrieval.dt.date.astype(str),
            "measurement_type": "observed_ground",
            "source_id": SOURCE_ID,
            "source_reported_aqi": raw["source_aqi"].astype("Int64"),
        }
    )
    daily["aqi_standard"] = AQI_STANDARD
    daily["aqi_version"] = AQI_VERSION
    daily["pm25_subindex"] = daily["value"].map(pm25_aqi).astype("Int64")
    daily["aqi_category"] = daily["pm25_subindex"].map(aqi_category)
    daily["pm10_subindex"] = pd.Series(pd.NA, index=daily.index, dtype="Int64")
    daily["no2_subindex"] = pd.Series(pd.NA, index=daily.index, dtype="Int64")
    daily["so2_subindex"] = pd.Series(pd.NA, index=daily.index, dtype="Int64")
    daily["co_subindex"] = pd.Series(pd.NA, index=daily.index, dtype="Int64")
    daily["o3_subindex"] = pd.Series(pd.NA, index=daily.index, dtype="Int64")
    daily["dominant_pollutant"] = "pm25"
    return daily.sort_values("timestamp_local").reset_index(drop=True)


def build_monthly(daily: pd.DataFrame, threshold_pct: float = DEFAULT_DAY_COVERAGE_PCT) -> pd.DataFrame:
    work = daily.copy()
    work["month_start"] = pd.to_datetime(work["date_local"]).dt.to_period("M").dt.to_timestamp()
    grouped = work.groupby("month_start", sort=True)
    monthly = grouped.agg(
        pm25_mean=("value", "mean"),
        pm25_median=("value", "median"),
        pm25_std=("value", "std"),
        pm25_min=("value", "min"),
        pm25_max=("value", "max"),
        pm25_q25=("value", lambda values: values.quantile(0.25)),
        pm25_q75=("value", lambda values: values.quantile(0.75)),
        aqi_mean=("pm25_subindex", "mean"),
        aqi_median=("pm25_subindex", "median"),
        aqi_max=("pm25_subindex", "max"),
        valid_days=("date_local", "nunique"),
        last_valid_day=("date_local", "max"),
    ).reset_index()
    full_index = pd.DataFrame(
        {
            "month_start": pd.period_range(
                monthly["month_start"].min(), monthly["month_start"].max(), freq="M"
            ).to_timestamp()
        }
    )
    monthly = full_index.merge(monthly, on="month_start", how="left")
    monthly["valid_days"] = monthly["valid_days"].fillna(0).astype(int)
    monthly["expected_days"] = monthly["month_start"].dt.days_in_month.astype(int)
    monthly["day_coverage_pct"] = monthly["valid_days"] / monthly["expected_days"] * 100
    monthly["valid_hours"] = pd.Series(pd.NA, index=monthly.index, dtype="Int64")
    monthly["expected_hours"] = monthly["expected_days"] * 24
    monthly["hour_coverage_pct"] = np.nan
    monthly["is_complete"] = monthly["day_coverage_pct"] >= threshold_pct
    # A terminal calendar month cut short by feed cessation is never considered complete.
    terminal = monthly["month_start"] == monthly["month_start"].max()
    terminal_month_end = monthly["month_start"] + pd.offsets.MonthEnd(0)
    observed_end = pd.to_datetime(monthly["last_valid_day"], errors="coerce")
    monthly.loc[terminal & (observed_end < terminal_month_end), "is_complete"] = False
    monthly["is_partial"] = ~monthly["is_complete"]
    monthly["completeness_rule"] = (
        f">={threshold_pct:g}% valid AirNow daily summaries; terminal month must reach month end"
    )
    monthly["source_id"] = SOURCE_ID
    monthly["station_id"] = "DK1010001"
    monthly["measurement_type"] = "observed_ground"
    monthly["unit"] = "ug/m3"
    monthly["aqi_standard"] = AQI_STANDARD
    monthly["aqi_version"] = AQI_VERSION
    numeric = [column for column in monthly if column.startswith("pm25_") or column.startswith("aqi_")]
    for column in numeric:
        if pd.api.types.is_numeric_dtype(monthly[column]):
            monthly[column] = monthly[column].round(3)
    return monthly


def update_manifest(manifest_path: Path, raw_path: Path, request_log: Path) -> None:
    manifest = {
        "manifest_version": 1,
        "generated_by": "scripts/build_primary_dataset.py",
        "sources": [
            {
                "source_id": SOURCE_ID,
                "provider": "US EPA AirNow / U.S. Department of State",
                "product_version": "daily_data_v2.dat dated archive",
                "retrieval_timestamp": "per-row; see raw file and request log",
                "download_url_template": (
                    "https://files.airnowtech.org/airnow/{YYYY}/{YYYYMMDD}/daily_data_v2.dat"
                ),
                "station": {
                    "station_id": "DK1010001",
                    "full_station_id": "050DK1010001",
                    "name": "Dhaka",
                    "latitude": 23.796374,
                    "longitude": 90.424614,
                },
                "license": "U.S. federal public data; third-party terms checked at use",
                "redistribution_allowed": True,
                "artifacts": [
                    {
                        "filename": str(raw_path.relative_to(manifest_path.parents[1])),
                        "sha256": sha256_file(raw_path),
                        "file_size": raw_path.stat().st_size,
                        "description": "Exact extracted Dhaka source rows plus response digests",
                    },
                    {
                        "filename": str(request_log.relative_to(manifest_path.parents[1])),
                        "sha256": sha256_file(request_log),
                        "file_size": request_log.stat().st_size,
                        "description": "One audit record for every dated archive request",
                    },
                ],
            }
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

