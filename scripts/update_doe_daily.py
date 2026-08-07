#!/usr/bin/env python3
"""Check the official DoE archives and rebuild outputs only when their inventory changes."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.doe import (  # noqa: E402
    DAILY_ARCHIVE_URL,
    MONTHLY_ARCHIVE_URL,
    MONTHLY_YEAR_PAGES,
    ArchiveRecord,
    discover_daily,
    discover_monthly,
    discover_monthly_year_pages,
    fetch_bytes,
    make_ssl_context,
)

CERTIFICATE = ROOT / "config/SectigoPublicServerAuthenticationCADVR36.pem"
MANIFEST = ROOT / "data/processed/doe_source_manifest.csv"


def log(message: str) -> None:
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    print(f"{timestamp} {message}", flush=True)


def discover_archive_records() -> list[ArchiveRecord]:
    context = make_ssl_context(CERTIFICATE)
    daily_html = fetch_bytes(DAILY_ARCHIVE_URL, context).decode("utf-8", errors="replace")
    daily_records = discover_daily(daily_html)

    master_html = fetch_bytes(MONTHLY_ARCHIVE_URL, context).decode(
        "utf-8", errors="replace"
    )
    monthly_pages = dict(MONTHLY_YEAR_PAGES)
    monthly_pages.update(discover_monthly_year_pages(master_html))
    monthly_records: list[ArchiveRecord] = []
    for year, page in sorted(monthly_pages.items(), reverse=True):
        html = fetch_bytes(page, context).decode("utf-8", errors="replace")
        monthly_records.extend(discover_monthly(year, page, html))
    return daily_records + monthly_records


def record_keys(records: list[ArchiveRecord]) -> set[tuple[str, str, str]]:
    return {(record.source_kind, record.period, record.url) for record in records}


def manifest_keys() -> set[tuple[str, str, str]]:
    if not MANIFEST.exists():
        return set()
    manifest = pd.read_csv(MANIFEST, usecols=["source_kind", "period", "source_url"], dtype=str)
    return set(manifest.itertuples(index=False, name=None))


def run(command: list[str]) -> None:
    log(f"running: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    log("checking official DoE daily and monthly archives")
    discovered = record_keys(discover_archive_records())
    existing = manifest_keys()
    added = sorted(discovered - existing)
    removed = sorted(existing - discovered)
    if not added and not removed:
        log(f"no archive changes ({len(discovered):,} attachments); nothing to rebuild")
        return

    log(f"archive changed: {len(added)} added, {len(removed)} removed")
    for source_kind, period, _ in (added + removed)[:20]:
        log(f"changed inventory item: {source_kind} {period}")
    if len(added) + len(removed) > 20:
        log(f"{len(added) + len(removed) - 20} additional inventory changes not listed")

    python = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    run(
        [
            python,
            "scripts/build_doe_workbook.py",
            "--workers",
            "10",
            "--incremental",
            "--discard-raw",
        ]
    )
    run([python, "scripts/analyze_doe_dataset.py"])
    run([pytest])
    log("dataset, workbook, analysis tables, and figures updated successfully")


if __name__ == "__main__":
    main()
