"""Validation and monthly aggregation for supplied BMD surface observations.

The BMD purchase portal does not publish a stable export schema. A received
file must therefore be mapped once into the long staging contract defined
here. The mapping is deliberately separate from the provider's raw file so
that no station identifier, unit, or QA field is guessed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.quality.completeness import day_coverage_pct, expected_days, is_complete_month

STAGING_COLUMNS = [
    "date_local",
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "variable",
    "value",
    "unit",
    "qa_flag",
    "observation_basis",
    "source_file",
    "retrieval_date",
    "notes",
]

MONTHLY_COLUMNS = [
    "month_start",
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "variable",
    "value",
    "unit",
    "aggregation",
    "valid_days",
    "expected_days",
    "day_coverage_pct",
    "is_complete",
    "provider",
    "source_id",
    "measurement_type",
    "source_files",
    "retrieval_date",
    "notes",
]

CANONICAL_UNITS = {
    "rainfall": "mm",
    "dry_bulb_temperature": "degC",
    "maximum_temperature": "degC",
    "minimum_temperature": "degC",
    "dew_point_temperature": "degC",
    "relative_humidity": "percent",
    "wind_speed": "m/s",
    "wind_direction": "degree",
    "mean_sea_level_pressure": "hPa",
    "station_level_pressure": "hPa",
    "sunshine_hour": "hour",
}


def empty_monthly() -> pd.DataFrame:
    """Return the committed empty product used until BMD data are supplied."""
    return pd.DataFrame(columns=MONTHLY_COLUMNS)


def validate_staging(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a manually mapped BMD daily file without filling missing metadata."""
    missing = set(STAGING_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"BMD staging file is missing columns: {sorted(missing)}")

    work = frame[STAGING_COLUMNS].copy()
    work["date_local"] = pd.to_datetime(work["date_local"], format="%Y-%m-%d", errors="raise")
    work["retrieval_date"] = pd.to_datetime(
        work["retrieval_date"], format="%Y-%m-%d", errors="raise"
    )
    work["value"] = pd.to_numeric(work["value"], errors="raise")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="raise")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="raise")

    required_text = [
        "station_id",
        "station_name",
        "variable",
        "unit",
        "observation_basis",
        "source_file",
    ]
    for column in required_text:
        if work[column].isna().any() or work[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"BMD staging column {column!r} contains a blank value")

    if set(work["station_name"]) != {"Dhaka"}:
        raise ValueError("BMD staging data must contain only the requested Dhaka station")
    if work["station_id"].nunique() != 1:
        raise ValueError("BMD staging data must contain one documented station identifier")
    if work["latitude"].nunique() != 1 or work["longitude"].nunique() != 1:
        raise ValueError("BMD staging station coordinates must be constant")
    if set(work["observation_basis"].str.lower()) != {"daily"}:
        raise ValueError("BMD staging data must be daily observations")
    if (work["retrieval_date"] < work["date_local"]).any():
        raise ValueError("BMD retrieval_date cannot precede the observation date")
    if not np.isfinite(work[["value", "latitude", "longitude"]].to_numpy()).all():
        raise ValueError("BMD staging numeric fields must be finite")
    if work.duplicated(["date_local", "station_id", "variable"]).any():
        raise ValueError("BMD staging data contain duplicate date/station/variable rows")

    unknown = set(work["variable"]) - set(CANONICAL_UNITS)
    if unknown:
        raise ValueError(f"Unknown canonical BMD variables: {sorted(unknown)}")
    bad_units = work[
        work.apply(lambda row: CANONICAL_UNITS[row["variable"]] != row["unit"], axis=1)
    ]
    if not bad_units.empty:
        pairs = sorted(set(zip(bad_units["variable"], bad_units["unit"], strict=True)))
        raise ValueError(f"Unexpected BMD variable/unit pairs: {pairs}")
    if (work.loc[work["variable"] == "rainfall", "value"] < 0).any():
        raise ValueError("BMD rainfall cannot be negative")
    if not work.loc[work["variable"] == "relative_humidity", "value"].between(0, 100).all():
        raise ValueError("BMD relative humidity must be between 0 and 100 percent")
    if (work.loc[work["variable"] == "wind_speed", "value"] < 0).any():
        raise ValueError("BMD wind speed cannot be negative")
    if not work.loc[work["variable"] == "wind_direction", "value"].between(
        0, 360, inclusive="left"
    ).all():
        raise ValueError("BMD wind direction must be in [0, 360) degrees")
    return work.sort_values(["date_local", "variable"]).reset_index(drop=True)


def _circular_mean_degrees(values: pd.Series) -> float:
    radians = np.deg2rad(values.to_numpy(dtype=float))
    angle = math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean()))
    normalized = float(angle % 360)
    return 0.0 if math.isclose(normalized, 360.0, abs_tol=1e-12) else normalized


def aggregate_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validated daily BMD values to a long monthly product."""
    work = validate_staging(frame)
    work["month_start"] = work["date_local"].dt.to_period("M").dt.to_timestamp()
    rows: list[dict[str, object]] = []
    for (month_start, variable), group in work.groupby(["month_start", "variable"], sort=True):
        if variable == "rainfall":
            value = float(group["value"].sum())
            aggregation = "sum_of_daily_values"
        elif variable == "wind_direction":
            value = _circular_mean_degrees(group["value"])
            aggregation = "circular_mean_of_daily_values"
        else:
            value = float(group["value"].mean())
            aggregation = "mean_of_daily_values"
        valid_days = int(group["date_local"].nunique())
        year = int(month_start.year)
        month = int(month_start.month)
        rows.append(
            {
                "month_start": month_start.strftime("%Y-%m-%d"),
                "station_id": group["station_id"].iloc[0],
                "station_name": group["station_name"].iloc[0],
                "latitude": float(group["latitude"].iloc[0]),
                "longitude": float(group["longitude"].iloc[0]),
                "variable": variable,
                "value": value,
                "unit": group["unit"].iloc[0],
                "aggregation": aggregation,
                "valid_days": valid_days,
                "expected_days": expected_days(year, month),
                "day_coverage_pct": day_coverage_pct(valid_days, year, month),
                "is_complete": is_complete_month(valid_days, year, month),
                "provider": "Bangladesh Meteorological Department",
                "source_id": "bmd_dhaka_surface_meteorology",
                "measurement_type": "observed_ground_meteorology",
                "source_files": ";".join(sorted(group["source_file"].astype(str).unique())),
                "retrieval_date": group["retrieval_date"].max().strftime("%Y-%m-%d"),
                "notes": "; ".join(
                    sorted(
                        note
                        for note in group["notes"].dropna().astype(str).str.strip().unique()
                        if note
                    )
                ),
            }
        )
    return pd.DataFrame(rows, columns=MONTHLY_COLUMNS)
