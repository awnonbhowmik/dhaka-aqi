#!/usr/bin/env python3
"""Create explicit empty validation schema until DoE machine-readable data are obtained."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "month_start",
    "station_id",
    "station_name",
    "pollutant",
    "value",
    "unit",
    "valid_days",
    "expected_days",
    "day_coverage_pct",
    "provider",
    "source_id",
    "measurement_type",
    "notes",
]


def main() -> None:
    output = ROOT / "data/processed/validation_observed_monthly.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)
    for name, columns in {
        "cams_modeled_monthly.csv": [
            "month_start",
            "grid_latitude",
            "grid_longitude",
            "pollutant",
            "value",
            "unit",
            "product_version",
            "measurement_type",
            "notes",
        ],
        "context_annual.csv": [
            "year",
            "variable",
            "value",
            "unit",
            "provider",
            "measurement_type",
            "notes",
        ],
    }.items():
        pd.DataFrame(columns=columns).to_csv(output.parent / name, index=False)


if __name__ == "__main__":
    main()
