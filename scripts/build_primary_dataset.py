#!/usr/bin/env python3
"""Build audited observed daily/monthly PM2.5 products."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aggregation.primary import build_daily, build_monthly, update_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-threshold", type=float, default=75.0)
    args = parser.parse_args()
    raw = ROOT / "data/raw/airnow/dhaka_pm25_daily_data_v2.csv"
    request_log = ROOT / "data/provenance/airnow_request_log.csv"
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    daily = build_daily(raw)
    monthly = build_monthly(daily, args.coverage_threshold)
    cutoff = monthly.loc[monthly["is_complete"], "month_start"].max()
    if cutoff is None:
        raise RuntimeError("No complete month under configured rule")
    analysis_monthly = monthly[(monthly["month_start"] <= cutoff) & monthly["is_complete"]].copy()
    daily.to_parquet(processed / "primary_observed_daily.parquet", index=False)
    monthly.to_csv(processed / "primary_observed_monthly.csv", index=False)
    analysis_monthly.to_csv(processed / "analysis_monthly.csv", index=False)
    update_manifest(ROOT / "data/source_manifest.yml", raw, request_log)
    print(
        f"daily={len(daily)} monthly={len(monthly)} complete={len(analysis_monthly)} "
        f"cutoff={cutoff.date()}"
    )


if __name__ == "__main__":
    main()

