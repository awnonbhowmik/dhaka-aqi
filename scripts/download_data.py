#!/usr/bin/env python3
"""Download authoritative raw inputs without storing credentials."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_sources.airnow import download_range  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["airnow"], default="airnow")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2019, 1, 1))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2025, 4, 30))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count, requests = download_range(
        args.start,
        args.end,
        ROOT / "data/raw/airnow/dhaka_pm25_daily_data_v2.csv",
        ROOT / "data/provenance/airnow_request_log.csv",
        workers=args.workers,
    )
    print(f"preserved {count} Dhaka PM2.5 records from {requests} dated archive requests")


if __name__ == "__main__":
    main()
