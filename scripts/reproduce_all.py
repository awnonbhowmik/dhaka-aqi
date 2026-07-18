#!/usr/bin/env python3
"""Run every deterministic build step from the committed raw extraction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "scripts/build_primary_dataset.py",
    "scripts/build_validation_dataset.py",
    "scripts/build_meteorology_dataset.py",
    "scripts/audit_legacy.py",
    "scripts/run_analysis.py",
    "scripts/run_forecasting.py",
    "scripts/build_manuscript.py",
]


def main() -> None:
    for step in STEPS:
        print(f"==> {step}", flush=True)
        subprocess.run([sys.executable, step], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
