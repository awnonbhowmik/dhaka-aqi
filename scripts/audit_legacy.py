#!/usr/bin/env python3
"""Create a row-level classification ledger for every legacy scalar value."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/provenance/legacy_observation_provenance.csv"
FIELDS = [
    "record_id",
    "date_or_month",
    "variable",
    "value",
    "unit",
    "legacy_file",
    "legacy_column",
    "claimed_source",
    "verified_source",
    "station_id",
    "measurement_type",
    "temporal_resolution",
    "aggregation_method",
    "coverage",
    "qa_flag",
    "source_version",
    "retrieval_date",
    "verification_status",
    "exclusion_reason",
    "notes",
]


def record_id(file: str, row_number: int, column: str) -> str:
    return hashlib.sha256(f"{file}|{row_number}|{column}".encode()).hexdigest()[:20]


def classify_daily(column: str, date: str, source: str) -> tuple[str, str, str]:
    if column == "aqi":
        if source == "monthly_avg":
            return "monthly_value_repeated_daily", "monthly", "Monthly AQI repeated by script"
        return "scraped_index", "daily", "Daily index scraped from aqi.in"
    if column in {"pm25", "pm10", "no2", "so2", "co", "o3"}:
        if date <= "2022-12-31":
            return "modeled_reanalysis", "daily", "CAMS overwrote pollutant column"
        if column in {"co", "o3"}:
            return "unknown", "daily", "Missing after CAMS coverage"
        return "monthly_value_repeated_daily", "monthly", "Monthly mean repeated by script"
    return "unknown", "daily", "Not an analytical observation"


def emit(writer: csv.DictWriter, **kwargs: str) -> None:
    row = {field: "" for field in FIELDS}
    row.update(kwargs)
    writer.writerow(row)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        path = ROOT / "data/daily_dhaka_aqi_dataset.csv"
        with path.open(encoding="utf-8", newline="") as source_handle:
            for row_number, row in enumerate(csv.DictReader(source_handle), start=2):
                for column in ("aqi", "pm25", "pm10", "no2", "so2", "co", "o3"):
                    if not row.get(column):
                        continue
                    kind, resolution, note = classify_daily(column, row["date"], row["source"])
                    counts[kind] = counts.get(kind, 0) + 1
                    emit(
                        writer,
                        record_id=record_id(str(path.relative_to(ROOT)), row_number, column),
                        date_or_month=row["date"],
                        variable=column,
                        value=row[column],
                        unit="unknown" if column == "aqi" else "claimed ug/m3",
                        legacy_file=str(path.relative_to(ROOT)),
                        legacy_column=column,
                        claimed_source=row["source"],
                        verified_source="legacy script/Git history",
                        station_id="unknown",
                        measurement_type=kind,
                        temporal_resolution=resolution,
                        aggregation_method="legacy merge/overwrite",
                        qa_flag="not preserved",
                        source_version="deb9b0d",
                        retrieval_date="unknown",
                        verification_status="classified_excluded",
                        exclusion_reason="not a homogeneous ground-observed concentration series",
                        notes=note,
                    )

        path = ROOT / "data/cams_dhaka_pollutants.csv"
        with path.open(encoding="utf-8", newline="") as source_handle:
            for row_number, row in enumerate(csv.DictReader(source_handle), start=2):
                for column in ("pm25", "pm10", "no2", "so2", "co", "o3"):
                    if not row.get(column):
                        continue
                    counts["modeled_reanalysis"] = counts.get("modeled_reanalysis", 0) + 1
                    emit(
                        writer,
                        record_id=record_id(str(path.relative_to(ROOT)), row_number, column),
                        date_or_month=row["date"],
                        variable=column,
                        value=row[column],
                        unit="claimed ug/m3",
                        legacy_file=str(path.relative_to(ROOT)),
                        legacy_column=column,
                        claimed_source="CAMS EAC4",
                        verified_source="legacy CAMS extraction; raw NetCDF absent",
                        station_id="grid_cell_not_station",
                        measurement_type="modeled_reanalysis",
                        temporal_resolution="daily",
                        aggregation_method="3-hourly modeled values averaged daily",
                        qa_flag="source metadata not preserved",
                        source_version="legacy EAC4 extraction",
                        retrieval_date="unknown",
                        verification_status="classified_excluded",
                        exclusion_reason="modeled product cannot enter observed dataset",
                        notes="Gas conversion used fixed air density; reprocess from official source before use",
                    )

        path = ROOT / "data/final_dhaka_aqi_dataset_clean.csv"
        with path.open(encoding="utf-8", newline="") as source_handle:
            for row_number, row in enumerate(csv.DictReader(source_handle), start=2):
                for column, value in row.items():
                    if value in {"", None} or column in {"month_start", "year", "month", "season"}:
                        continue
                    if column.startswith(("population", "urban_", "hdi", "poverty")):
                        kind = "contextual_annual"
                        reason = "annual contextual value repeated monthly; exclude from monthly inference"
                    elif column == "norm_rain":
                        kind = "unknown"
                        reason = "normalized index is not physical rainfall"
                    elif column.startswith("aqi_"):
                        kind = "unknown"
                        reason = "AQI origin/standard not verified by record"
                    else:
                        kind = "unknown"
                        reason = "station/provider/method not verified by record"
                    counts[kind] = counts.get(kind, 0) + 1
                    emit(
                        writer,
                        record_id=record_id(str(path.relative_to(ROOT)), row_number, column),
                        date_or_month=row["month_start"],
                        variable=column,
                        value=value,
                        unit="legacy README claim only",
                        legacy_file=str(path.relative_to(ROOT)),
                        legacy_column=column,
                        claimed_source="Dhaka continuous ambient air-quality monitoring station",
                        verified_source="unresolved deleted enriched workbook",
                        station_id="unknown",
                        measurement_type=kind,
                        temporal_resolution="monthly" if kind != "contextual_annual" else "annual repeated monthly",
                        aggregation_method="unknown",
                        qa_flag="not preserved",
                        source_version="69aee4b/c766118",
                        retrieval_date="unknown",
                        verification_status="unknown_excluded",
                        exclusion_reason=reason,
                        notes="December 2025 source workbook cited snippets rather than an observation series",
                    )

    report = ROOT / "reports/legacy_values_excluded.md"
    lines = [
        "# Legacy values excluded",
        "",
        "Every analytical scalar in the three legacy CSVs is represented in the provenance ledger.",
        "None enters the revised observed series.",
        "",
        "## Counts by classification",
        "",
    ]
    lines.extend(f"- `{key}`: {value:,}" for key, value in sorted(counts.items()))
    lines.extend(
        [
            "",
            "## Reasons",
            "",
            "- CAMS rows are modeled reanalysis and remain separate from observations.",
            "- Monthly means repeated over days are not independent daily observations.",
            "- Scraped AQI values are indexes, not physical pollutant concentrations.",
            "- Monthly station, instrument, units, provider transitions, and QA flags are unresolved.",
            "- December 2025 is not accepted as observed; web snippets do not establish a monthly series.",
            "- Annual socioeconomic values repeated monthly are contextual only.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

