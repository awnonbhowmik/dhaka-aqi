"""Completeness rules for daily-to-monthly aggregation."""

from __future__ import annotations

import calendar

DEFAULT_DAY_COVERAGE_PCT = 75.0


def expected_days(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def day_coverage_pct(valid_days: int, year: int, month: int) -> float:
    if valid_days < 0:
        raise ValueError("valid_days must be non-negative")
    expected = expected_days(year, month)
    if valid_days > expected:
        raise ValueError("valid_days exceeds calendar days")
    return valid_days / expected * 100.0


def is_complete_month(
    valid_days: int,
    year: int,
    month: int,
    threshold_pct: float = DEFAULT_DAY_COVERAGE_PCT,
) -> bool:
    return day_coverage_pct(valid_days, year, month) >= threshold_pct

