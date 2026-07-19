#!/usr/bin/env python3
"""Download official DoE reports and build the fresh Dhaka AQI workbook."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.doe import (  # noqa: E402
    DAILY_ARCHIVE_URL,
    MONTHLY_YEAR_PAGES,
    ArchiveRecord,
    build_monthly_dataset,
    discover_daily,
    discover_monthly,
    extract_daily,
    extract_monthly,
    extract_monthly_report_aqi,
    fetch_bytes,
    make_ssl_context,
    save_attachment,
    select_daily_records,
    write_workbook,
)

RAW = ROOT / "data/raw/doe"
PROCESSED = ROOT / "data/processed"
CERTIFICATE = ROOT / "config/SectigoPublicServerAuthenticationCADVR36.pem"
HDI = ROOT / "data/context/bangladesh_hdi.csv"
POPULATION = ROOT / "data/context/bangladesh_population.csv"

DAILY_COLUMNS = [
    "report_date", "document_aqi_date", "published_date", "city", "city_as_reported",
    "aqi_as_reported", "aqi", "responsible_pollutant", "aqi_category_as_reported",
    "comments", "is_missing", "source_category_scheme", "dhaka_basis_note",
    "extraction_method", "source_url", "source_sha256", "source_file_type",
    "selected_record", "duplicate_date", "qa_status",
]
MONTHLY_COLUMNS = [
    "report_month", "station_label_as_reported", "parameter", "parameter_as_reported",
    "statistic_as_reported", "value_as_reported", "value", "unit", "is_missing",
    "page_number", "table_number", "extraction_method", "source_url", "source_sha256",
]
MONTHLY_REPORT_AQI_COLUMNS = [
    "report_month", "aqi_date", "city", "aqi_as_reported", "aqi", "is_missing",
    "extraction_method", "source_url", "source_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--daily-limit", type=int)
    parser.add_argument("--monthly-limit", type=int)
    parser.add_argument(
        "--workbook-only",
        action="store_true",
        help="Rebuild the XLSX from existing processed CSVs without network access",
    )
    return parser.parse_args()


def download_all(
    records: list[ArchiveRecord], context: Any, workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(save_attachment, RAW, record, context): record for record in records}
        completed = 0
        for future in as_completed(futures):
            record = futures[future]
            completed += 1
            try:
                result = future.result()
                result["local_path"] = Path(result["local_path"]).relative_to(ROOT).as_posix()
                manifest.append(result)
            except Exception as error:
                issues.append(
                    {
                        "stage": "download",
                        "period": record.period,
                        "source_url": record.url,
                        "severity": "error",
                        "issue": str(error),
                    }
                )
            if completed % 100 == 0 or completed == len(records):
                print(f"downloaded/checked {completed:,}/{len(records):,}", flush=True)
    return manifest, issues


def monthly_aqi_from_cached_reports(manifest_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reports = manifest_frame[manifest_frame["source_kind"].eq("monthly_report")]
    for source in reports.to_dict(orient="records"):
        path = Path(source["local_path"])
        if not path.is_absolute():
            path = ROOT / path
        rows.extend(extract_monthly_report_aqi(path, source))
    return pd.DataFrame(rows, columns=MONTHLY_REPORT_AQI_COLUMNS).sort_values(
        ["report_month", "aqi_date"]
    )


def write_derived_outputs(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    monthly_report_aqi: pd.DataFrame,
    manifest_frame: pd.DataFrame,
    qa: pd.DataFrame,
) -> pd.DataFrame:
    population = pd.read_csv(POPULATION)
    hdi = pd.read_csv(HDI)
    monthly_dataset = build_monthly_dataset(monthly, daily, monthly_report_aqi)
    monthly_report_aqi.to_csv(PROCESSED / "doe_monthly_report_dhaka_aqi.csv", index=False)
    monthly_dataset.to_csv(PROCESSED / "doe_monthly_dataset_wide.csv", index=False)
    workbook = PROCESSED / "dhaka_doe_air_quality.xlsx"
    write_workbook(
        workbook,
        monthly_dataset,
        daily,
        monthly_report_aqi,
        monthly,
        population,
        hdi,
        manifest_frame,
        qa,
    )
    return monthly_dataset


def main() -> None:
    args = parse_args()
    if args.workbook_only:
        daily = pd.read_csv(PROCESSED / "doe_daily_dhaka_aqi.csv")
        monthly = pd.read_csv(PROCESSED / "doe_monthly_dhaka.csv")
        manifest_frame = pd.read_csv(PROCESSED / "doe_source_manifest.csv")
        qa = pd.read_csv(PROCESSED / "doe_qa_issues.csv")
        monthly_report_aqi = monthly_aqi_from_cached_reports(manifest_frame)
        monthly_dataset = write_derived_outputs(
            daily, monthly, monthly_report_aqi, manifest_frame, qa
        )
        summary_path = PROCESSED / "doe_build_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "monthly_report_aqi_rows": len(monthly_report_aqi),
                "monthly_wide_rows": len(monthly_dataset),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        workbook = PROCESSED / "dhaka_doe_air_quality.xlsx"
        print(
            f"rebuilt {workbook.relative_to(ROOT)} and {len(monthly_dataset):,} monthly rows "
            "from cached official sources"
        )
        return

    context = make_ssl_context(CERTIFICATE)
    retrieval_time = datetime.now(UTC).replace(microsecond=0).isoformat()
    daily_html = fetch_bytes(DAILY_ARCHIVE_URL, context).decode("utf-8", errors="replace")
    daily_records = discover_daily(daily_html)
    if args.daily_limit:
        daily_records = daily_records[-args.daily_limit :]

    monthly_records: list[ArchiveRecord] = []
    archive_issues: list[dict[str, Any]] = []
    for year, page in MONTHLY_YEAR_PAGES.items():
        try:
            html = fetch_bytes(page, context).decode("utf-8", errors="replace")
            monthly_records.extend(discover_monthly(year, page, html))
        except Exception as error:
            archive_issues.append(
                {
                    "stage": "archive_discovery",
                    "period": str(year),
                    "source_url": page,
                    "severity": "error",
                    "issue": str(error),
                }
            )
    if args.monthly_limit:
        monthly_records = sorted(monthly_records, key=lambda record: record.period)[
            -args.monthly_limit :
        ]
    records = daily_records + monthly_records
    print(
        f"discovered {len(daily_records):,} daily attachments and "
        f"{len(monthly_records):,} monthly attachments",
        flush=True,
    )
    manifest, issues = download_all(records, context, args.workers)
    issues.extend(archive_issues)
    for row in manifest:
        row["retrieval_timestamp_utc"] = retrieval_time

    daily_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    monthly_report_aqi_rows: list[dict[str, Any]] = []
    for index, source in enumerate(manifest, start=1):
        path = ROOT / source["local_path"] if not Path(source["local_path"]).is_absolute() else Path(source["local_path"])
        try:
            if source["source_kind"] == "daily_aqi":
                daily_rows.append(extract_daily(path, source))
            else:
                extracted = extract_monthly(path, source)
                if not extracted:
                    raise ValueError("No Dhaka pollutant summary rows extracted")
                monthly_rows.extend(extracted)
                expected = {"PM2.5", "PM10", "SO2", "NO2", "CO", "O3"}
                missing = sorted(expected - {row["parameter"] for row in extracted})
                if missing:
                    source["extraction_status"] = "partial"
                    issues.append(
                        {
                            "stage": "monthly_record_validation",
                            "period": source["period"],
                            "source_url": source["source_url"],
                            "severity": "warning",
                            "issue": f"Pollutant blocks not extracted: {', '.join(missing)}",
                        }
                    )
                else:
                    source["extraction_status"] = "ok"
                try:
                    monthly_report_aqi_rows.extend(extract_monthly_report_aqi(path, source))
                except Exception as error:
                    issues.append(
                        {
                            "stage": "monthly_aqi_extraction",
                            "period": source["period"],
                            "source_url": source["source_url"],
                            "severity": "warning",
                            "issue": str(error),
                        }
                    )
            if source["source_kind"] == "daily_aqi":
                source["extraction_status"] = "ok"
        except Exception as error:
            source["extraction_status"] = "failed"
            issues.append(
                {
                    "stage": "extraction",
                    "period": source["period"],
                    "source_url": source["source_url"],
                    "severity": "error",
                    "issue": str(error),
                }
            )
        if index % 100 == 0 or index == len(manifest):
            print(f"extracted {index:,}/{len(manifest):,}", flush=True)

    daily_rows = select_daily_records(daily_rows)
    for row in daily_rows:
        if row["qa_status"] != "ok":
            issues.append(
                {
                    "stage": "daily_record_validation",
                    "period": row["report_date"],
                    "source_url": row["source_url"],
                    "severity": "warning",
                    "issue": row["qa_status"],
                }
            )
    daily = pd.DataFrame(daily_rows, columns=DAILY_COLUMNS).sort_values(["report_date", "source_url"])
    monthly = pd.DataFrame(monthly_rows, columns=MONTHLY_COLUMNS).sort_values(
        ["report_month", "station_label_as_reported", "parameter_as_reported", "statistic_as_reported"]
    )
    manifest_frame = pd.DataFrame(manifest).sort_values(["source_kind", "period", "source_url"])
    qa = pd.DataFrame(issues, columns=["stage", "period", "source_url", "severity", "issue"])
    monthly_report_aqi = pd.DataFrame(
        monthly_report_aqi_rows, columns=MONTHLY_REPORT_AQI_COLUMNS
    ).sort_values(["report_month", "aqi_date"])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    daily.to_csv(PROCESSED / "doe_daily_dhaka_aqi.csv", index=False)
    monthly.to_csv(PROCESSED / "doe_monthly_dhaka.csv", index=False)
    manifest_frame.to_csv(PROCESSED / "doe_source_manifest.csv", index=False)
    qa.to_csv(PROCESSED / "doe_qa_issues.csv", index=False)
    monthly_dataset = write_derived_outputs(
        daily, monthly, monthly_report_aqi, manifest_frame, qa
    )
    workbook = PROCESSED / "dhaka_doe_air_quality.xlsx"
    summary = {
        "retrieval_timestamp_utc": retrieval_time,
        "daily_archive_records": len(daily_records),
        "daily_extracted_rows": len(daily),
        "daily_selected_rows": int(daily["selected_record"].sum()) if not daily.empty else 0,
        "daily_start": daily["report_date"].min() if not daily.empty else None,
        "daily_end": daily["report_date"].max() if not daily.empty else None,
        "monthly_archive_records": len(monthly_records),
        "monthly_extracted_rows": len(monthly),
        "monthly_report_aqi_rows": len(monthly_report_aqi),
        "monthly_wide_rows": len(monthly_dataset),
        "qa_issues": len(qa),
        "workbook": workbook.relative_to(ROOT).as_posix(),
    }
    (PROCESSED / "doe_build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
