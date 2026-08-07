#!/usr/bin/env python3
"""Download official DoE reports and build the fresh Dhaka AQI workbook."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.doe import (  # noqa: E402
    DAILY_ARCHIVE_URL,
    MONTHLY_ARCHIVE_URL,
    MONTHLY_YEAR_PAGES,
    ArchiveRecord,
    build_monthly_dataset,
    discover_daily,
    discover_monthly,
    discover_monthly_year_pages,
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
POPULATION_WORLDOMETER = ROOT / "data/context/bangladesh_population_worldometer.csv"
TREE_COVER_LOSS = ROOT / "data/context/bangladesh_tree_cover_loss.csv"

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
        "--incremental",
        action="store_true",
        help="Reuse extracted rows for unchanged manifest entries and parse only changed reports",
    )
    parser.add_argument(
        "--discard-raw",
        action="store_true",
        help="Delete downloaded report files after all processed outputs are written successfully",
    )
    parser.add_argument(
        "--workbook-only",
        action="store_true",
        help="Rebuild the XLSX from existing processed CSVs without network access",
    )
    args = parser.parse_args()
    if args.incremental and (args.daily_limit or args.monthly_limit):
        parser.error("--incremental cannot be combined with archive limit options")
    if args.workbook_only and (args.incremental or args.discard_raw):
        parser.error("--workbook-only cannot be combined with download/update options")
    return args


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


def extract_source(source: dict[str, Any]) -> dict[str, Any]:
    """Extract one cached attachment in an isolated worker process."""
    path = Path(source["local_path"])
    if not path.is_absolute():
        path = ROOT / path
    daily_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    monthly_report_aqi_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    status = "ok"
    try:
        if source["source_kind"] == "daily_aqi":
            daily_rows.append(extract_daily(path, source))
        else:
            monthly_rows = extract_monthly(path, source)
            if not monthly_rows:
                raise ValueError("No Dhaka pollutant summary rows extracted")
            expected = {"PM2.5", "PM10", "SO2", "NO2", "CO", "O3"}
            missing = sorted(expected - {row["parameter"] for row in monthly_rows})
            if missing:
                status = "partial"
                issues.append(
                    {
                        "stage": "monthly_record_validation",
                        "period": source["period"],
                        "source_url": source["source_url"],
                        "severity": "warning",
                        "issue": f"Pollutant blocks not extracted: {', '.join(missing)}",
                    }
                )
            try:
                monthly_report_aqi_rows = extract_monthly_report_aqi(path, source)
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
    except Exception as error:
        status = "failed"
        issues.append(
            {
                "stage": "extraction",
                "period": source["period"],
                "source_url": source["source_url"],
                "severity": "error",
                "issue": str(error),
            }
        )
    return {
        "daily_rows": daily_rows,
        "monthly_rows": monthly_rows,
        "monthly_report_aqi_rows": monthly_report_aqi_rows,
        "issues": issues,
        "extraction_status": status,
    }


def extract_all(
    manifest: list[dict[str, Any]], workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract independent attachments concurrently while updating their manifest status."""
    daily_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    monthly_report_aqi_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if not manifest:
        return daily_rows, monthly_rows, monthly_report_aqi_rows, issues

    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(extract_source, source): index
            for index, source in enumerate(manifest)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            source = manifest[index]
            completed += 1
            try:
                result = future.result()
                source["extraction_status"] = result["extraction_status"]
                daily_rows.extend(result["daily_rows"])
                monthly_rows.extend(result["monthly_rows"])
                monthly_report_aqi_rows.extend(result["monthly_report_aqi_rows"])
                issues.extend(result["issues"])
            except Exception as error:
                source["extraction_status"] = "failed"
                issues.append(
                    {
                        "stage": "extraction_worker",
                        "period": source["period"],
                        "source_url": source["source_url"],
                        "severity": "error",
                        "issue": str(error),
                    }
                )
            if completed % 100 == 0 or completed == len(manifest):
                print(f"extracted {completed:,}/{len(manifest):,}", flush=True)
    return daily_rows, monthly_rows, monthly_report_aqi_rows, issues


def _record_key(record: ArchiveRecord) -> tuple[str, str, str]:
    return record.source_kind, record.period, record.url


def _source_key(source: dict[str, Any]) -> tuple[str, str, str]:
    return str(source["source_kind"]), str(source["period"]), str(source["source_url"])


def _filter_rows(
    frame: pd.DataFrame,
    source_kind: str,
    period_column: str,
    reusable_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    keep = [
        (source_kind, str(period), str(url)) in reusable_keys
        for period, url in zip(frame[period_column], frame["source_url"], strict=True)
    ]
    return frame.loc[keep].to_dict(orient="records")


def load_incremental_state(
    records: list[ArchiveRecord],
) -> tuple[
    list[dict[str, Any]],
    list[ArchiveRecord],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Reuse source-level extracted rows when the archive inventory entry is unchanged."""
    paths = {
        "manifest": PROCESSED / "doe_source_manifest.csv",
        "daily": PROCESSED / "doe_daily_dhaka_aqi.csv",
        "monthly": PROCESSED / "doe_monthly_dhaka.csv",
        "monthly_aqi": PROCESSED / "doe_monthly_report_dhaka_aqi.csv",
        "qa": PROCESSED / "doe_qa_issues.csv",
    }
    if not all(path.exists() for path in paths.values()):
        return [], records, [], [], [], []

    old_manifest = pd.read_csv(paths["manifest"], dtype={"period": str}, keep_default_na=False)
    current_keys = {_record_key(record) for record in records}
    reusable_manifest = [
        source
        for source in old_manifest.to_dict(orient="records")
        if _source_key(source) in current_keys
        and source.get("extraction_status") in {"ok", "partial"}
    ]
    reusable_keys = {_source_key(source) for source in reusable_manifest}
    pending = [record for record in records if _record_key(record) not in reusable_keys]

    daily_rows = _filter_rows(
        pd.read_csv(paths["daily"]), "daily_aqi", "report_date", reusable_keys
    )
    monthly_rows = _filter_rows(
        pd.read_csv(paths["monthly"]), "monthly_report", "report_month", reusable_keys
    )
    monthly_aqi_rows = _filter_rows(
        pd.read_csv(paths["monthly_aqi"]),
        "monthly_report",
        "report_month",
        reusable_keys,
    )

    reusable_monthly_period_urls = {
        (period, url)
        for kind, period, url in reusable_keys
        if kind == "monthly_report"
    }
    old_qa = pd.read_csv(paths["qa"], dtype={"period": str}, keep_default_na=False)
    preserved_stages = {"monthly_aqi_extraction", "monthly_record_validation"}
    keep_qa = [
        stage in preserved_stages and (str(period), str(url)) in reusable_monthly_period_urls
        for stage, period, url in zip(
            old_qa["stage"], old_qa["period"], old_qa["source_url"], strict=True
        )
    ]
    preserved_issues = old_qa.loc[keep_qa].to_dict(orient="records")
    return (
        reusable_manifest,
        pending,
        daily_rows,
        monthly_rows,
        monthly_aqi_rows,
        preserved_issues,
    )


def reset_daily_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recalculate duplicate selection after incremental rows are merged."""
    for row in rows:
        row["selected_record"] = False
        row["duplicate_date"] = False
        row["qa_status"] = (
            "ok"
            if str(row["document_aqi_date"]) == str(row["report_date"])
            else "document_date_mismatch"
        )
    return select_daily_records(rows)


def discard_raw_files(local_paths: list[str]) -> tuple[int, int]:
    """Delete only validated manifest paths below the dedicated DoE raw directory."""
    raw_root = RAW.resolve()
    deleted_files = 0
    deleted_bytes = 0
    for value in sorted(set(filter(None, local_paths))):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_symlink():
            raise RuntimeError(f"Refusing to delete symlinked raw report: {candidate}")
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(raw_root):
            raise RuntimeError(f"Refusing to delete raw report outside {raw_root}: {resolved}")
        if not resolved.exists():
            continue
        if not resolved.is_file():
            raise RuntimeError(f"Refusing to delete non-file raw report: {resolved}")
        deleted_bytes += resolved.stat().st_size
        resolved.unlink()
        deleted_files += 1
    for directory in [RAW / "daily_aqi", RAW / "monthly_report", RAW]:
        if directory.exists() and not directory.is_symlink() and not any(directory.iterdir()):
            directory.rmdir()
    return deleted_files, deleted_bytes


def write_derived_outputs(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    monthly_report_aqi: pd.DataFrame,
    manifest_frame: pd.DataFrame,
    qa: pd.DataFrame,
) -> pd.DataFrame:
    population = pd.read_csv(POPULATION)
    population_worldometer = pd.read_csv(POPULATION_WORLDOMETER)
    tree_cover_loss = pd.read_csv(TREE_COVER_LOSS)
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
        population_worldometer,
        tree_cover_loss,
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
        monthly_report_aqi = pd.read_csv(
            PROCESSED / "doe_monthly_report_dhaka_aqi.csv"
        )
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
            "from processed official-source tables"
        )
        return

    context = make_ssl_context(CERTIFICATE)
    retrieval_time = datetime.now(UTC).replace(microsecond=0).isoformat()
    daily_html = fetch_bytes(DAILY_ARCHIVE_URL, context).decode("utf-8", errors="replace")
    daily_records = discover_daily(daily_html)
    if args.daily_limit:
        daily_records = daily_records[-args.daily_limit :]

    archive_issues: list[dict[str, Any]] = []
    monthly_pages = dict(MONTHLY_YEAR_PAGES)
    try:
        monthly_master_html = fetch_bytes(MONTHLY_ARCHIVE_URL, context).decode(
            "utf-8", errors="replace"
        )
        monthly_pages.update(discover_monthly_year_pages(monthly_master_html))
    except Exception as error:
        archive_issues.append(
            {
                "stage": "archive_discovery",
                "period": "monthly_master",
                "source_url": MONTHLY_ARCHIVE_URL,
                "severity": "warning",
                "issue": f"Using known year pages because master-page discovery failed: {error}",
            }
        )

    monthly_records: list[ArchiveRecord] = []
    for year, page in sorted(monthly_pages.items(), reverse=True):
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
    if args.incremental and any(issue["severity"] == "error" for issue in archive_issues):
        raise RuntimeError(
            "Incremental update aborted because an archive page could not be read; "
            "existing processed rows were left unchanged"
        )
    print(
        f"discovered {len(daily_records):,} daily attachments and "
        f"{len(monthly_records):,} monthly attachments",
        flush=True,
    )
    reused_manifest: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    monthly_report_aqi_rows: list[dict[str, Any]] = []
    preserved_issues: list[dict[str, Any]] = []
    pending_records = records
    if args.incremental:
        (
            reused_manifest,
            pending_records,
            daily_rows,
            monthly_rows,
            monthly_report_aqi_rows,
            preserved_issues,
        ) = load_incremental_state(records)
        print(
            f"reusing {len(reused_manifest):,} extracted attachments; "
            f"downloading/extracting {len(pending_records):,}",
            flush=True,
        )

    downloaded_manifest, issues = download_all(pending_records, context, args.workers)
    if args.incremental and issues:
        raise RuntimeError(
            "Incremental update aborted because one or more new reports could not be downloaded; "
            "existing processed rows were left unchanged"
        )
    issues.extend(archive_issues)
    issues.extend(preserved_issues)
    for row in downloaded_manifest:
        row["retrieval_timestamp_utc"] = retrieval_time

    (
        extracted_daily_rows,
        extracted_monthly_rows,
        extracted_monthly_report_aqi_rows,
        extraction_issues,
    ) = extract_all(downloaded_manifest, args.workers)
    daily_rows.extend(extracted_daily_rows)
    monthly_rows.extend(extracted_monthly_rows)
    monthly_report_aqi_rows.extend(extracted_monthly_report_aqi_rows)
    issues.extend(extraction_issues)
    if args.incremental and any(
        source.get("extraction_status") == "failed" for source in downloaded_manifest
    ):
        raise RuntimeError(
            "Incremental update aborted because a new report could not be extracted; "
            "existing processed rows were left unchanged and the raw report was retained"
        )
    manifest = reused_manifest + downloaded_manifest

    raw_paths = [str(source.get("local_path", "")) for source in manifest]
    if args.discard_raw:
        for source in manifest:
            source["local_path"] = ""
            source["download_status"] = "processed_then_deleted"

    daily_rows = reset_daily_selection(daily_rows)
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
    qa = pd.DataFrame(
        issues, columns=["stage", "period", "source_url", "severity", "issue"]
    ).sort_values(["stage", "period", "source_url", "severity", "issue"])
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
        "raw_reports_retained": not args.discard_raw,
        "workbook": workbook.relative_to(ROOT).as_posix(),
    }
    (PROCESSED / "doe_build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.discard_raw:
        deleted_files, deleted_bytes = discard_raw_files(raw_paths)
        print(
            f"deleted {deleted_files:,} processed raw reports "
            f"({deleted_bytes / (1024**2):,.1f} MiB)",
            flush=True,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
