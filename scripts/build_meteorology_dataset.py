#!/usr/bin/env python3
"""Build the BMD monthly meteorology product from an authorized daily delivery."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_sources.bmd import aggregate_monthly, empty_monthly  # noqa: E402

STAGING = ROOT / "data/staging/bmd_dhaka_daily.csv"
OUTPUT = ROOT / "data/processed/meteorology_monthly.csv"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not STAGING.exists():
        empty_monthly().to_csv(OUTPUT, index=False)
        print(
            "BMD data not supplied: wrote an empty meteorology product. "
            "See docs/bmd_data_request.md."
        )
        return
    monthly = aggregate_monthly(pd.read_csv(STAGING))
    monthly.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(monthly):,} BMD station-variable months to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
