"""Verified-HTTPS adapter for official dated AirNow daily summary files."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import logging
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://files.airnowtech.org/airnow/{year}/{ymd}/daily_data_v2.dat"
TARGET_SITE = "DK1010001"
TARGET_PARAMETER = "PM2.5-24hr"
EXPECTED_AGENCY = "U.S. Department of State Bangladesh - Dhaka"


@dataclass(frozen=True)
class AirNowRecord:
    date_local: str
    station_id: str
    station_name: str
    parameter: str
    unit: str
    value: float
    duration_hours: int
    agency: str
    source_aqi: int | None
    source_category_number: int | None
    latitude: float
    longitude: float
    full_station_id: str
    source_url: str
    source_line: str
    retrieval_timestamp_utc: str
    response_sha256: str


def iter_dates(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def parse_target_line(text: str, source_url: str, retrieved: str, digest: str) -> AirNowRecord | None:
    """Parse the single Dhaka PM2.5 record from an AirNow daily file."""
    for raw_line in text.splitlines():
        if f"|{TARGET_SITE}|" not in raw_line or f"|{TARGET_PARAMETER}|" not in raw_line:
            continue
        fields = next(csv.reader([raw_line], delimiter="|"))
        if len(fields) not in {13, 14}:
            raise ValueError(f"Unexpected AirNow field count {len(fields)} at {source_url}")
        (
            date_text,
            station_id,
            station_name,
            parameter,
            unit,
            value,
            duration,
            agency,
            source_aqi,
            category_number,
            latitude,
            longitude,
            full_station_id,
        ) = fields[:13]
        # Some files include a trailing delimiter; reject material extra data.
        if any(item.strip() for item in fields[13:]):
            raise ValueError(f"Unexpected trailing AirNow content at {source_url}")
        parsed_date = dt.datetime.strptime(date_text, "%m/%d/%y").date().isoformat()
        if agency != EXPECTED_AGENCY:
            raise ValueError(f"Unexpected agency for {TARGET_SITE}: {agency}")
        return AirNowRecord(
            date_local=parsed_date,
            station_id=station_id,
            station_name=station_name,
            parameter=parameter,
            unit=unit,
            value=float(value),
            duration_hours=int(duration),
            agency=agency,
            source_aqi=None if source_aqi == "-999" else int(source_aqi),
            source_category_number=None if category_number == "-999" else int(category_number),
            latitude=float(latitude),
            longitude=float(longitude),
            full_station_id=full_station_id,
            source_url=source_url,
            source_line=raw_line,
            retrieval_timestamp_utc=retrieved,
            response_sha256=digest,
        )
    return None


def _fetch_one(day: dt.date, retries: int = 3) -> tuple[AirNowRecord | None, dict[str, object]]:
    url = BASE_URL.format(year=day.year, ymd=day.strftime("%Y%m%d"))
    context = ssl.create_default_context()
    retrieved = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    request = urllib.request.Request(url, headers={"User-Agent": "dhaka-aqi-research/0.2"})
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, context=context, timeout=45) as response:
                payload = response.read()
                digest = hashlib.sha256(payload).hexdigest()
                record = parse_target_line(
                    payload.decode("utf-8", errors="strict"), url, retrieved, digest
                )
                return record, {
                    "date_requested": day.isoformat(),
                    "source_url": url,
                    "http_status": response.status,
                    "bytes": len(payload),
                    "response_sha256": digest,
                    "target_record_found": record is not None,
                    "retrieval_timestamp_utc": retrieved,
                    "error": "",
                }
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None, {
                    "date_requested": day.isoformat(),
                    "source_url": url,
                    "http_status": 404,
                    "bytes": 0,
                    "response_sha256": "",
                    "target_record_found": False,
                    "retrieval_timestamp_utc": retrieved,
                    "error": "archive file not found",
                }
            last_error = f"HTTP {exc.code}: {exc.reason}"
        except (OSError, UnicodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(0.5 * (2 ** (attempt - 1)))
    return None, {
        "date_requested": day.isoformat(),
        "source_url": url,
        "http_status": "",
        "bytes": 0,
        "response_sha256": "",
        "target_record_found": False,
        "retrieval_timestamp_utc": retrieved,
        "error": last_error,
    }


def download_range(
    start: dt.date,
    end: dt.date,
    raw_output: Path,
    request_log: Path,
    workers: int = 16,
) -> tuple[int, int]:
    """Download a date range and preserve exact target lines plus request audit."""
    if end < start:
        raise ValueError("end date precedes start date")
    records: list[AirNowRecord] = []
    audits: list[dict[str, object]] = []
    days = list(iter_dates(start, end))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, day): day for day in days}
        for index, future in enumerate(as_completed(futures), start=1):
            record, audit = future.result()
            audits.append(audit)
            if record is not None:
                records.append(record)
            if index % 250 == 0:
                LOGGER.info("downloaded %s/%s dates; %s target records", index, len(days), len(records))

    records.sort(key=lambda row: row.date_local)
    audits.sort(key=lambda row: str(row["date_requested"]))
    if len({row.date_local for row in records}) != len(records):
        raise ValueError("Duplicate AirNow local dates found")

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    request_log.parent.mkdir(parents=True, exist_ok=True)
    record_fields = list(asdict(records[0]).keys()) if records else list(AirNowRecord.__annotations__)
    with raw_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=record_fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)
    with request_log.open("w", encoding="utf-8", newline="") as handle:
        fields = list(audits[0]) if audits else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audits)
    return len(records), len(audits)
