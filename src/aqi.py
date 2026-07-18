"""AQI calculations with explicit standards and breakpoint versions."""

from __future__ import annotations

import math

AQI_STANDARD = "US EPA AQI"
AQI_VERSION = "2024 PM2.5 breakpoints (effective 2024-05-06)"

# Concentration endpoints are for the truncated 24-hour PM2.5 concentration.
PM25_BREAKPOINTS = (
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
)


def pm25_aqi(concentration_ug_m3: float) -> int:
    """Calculate U.S. EPA 2024 PM2.5 AQI after truncating to 0.1 ug/m3."""
    value = float(concentration_ug_m3)
    if not math.isfinite(value) or value < 0:
        raise ValueError("PM2.5 concentration must be finite and non-negative")
    truncated = math.floor(value * 10) / 10
    if truncated > 325.4:
        return 500
    for c_low, c_high, i_low, i_high in PM25_BREAKPOINTS:
        if c_low <= truncated <= c_high:
            index = (i_high - i_low) / (c_high - c_low) * (truncated - c_low) + i_low
            return int(round(index))
    raise ValueError(f"No AQI breakpoint for concentration {truncated}")


def aqi_category(index: int) -> str:
    """Return the U.S. EPA category for an integer AQI."""
    if index <= 50:
        return "Good"
    if index <= 100:
        return "Moderate"
    if index <= 150:
        return "Unhealthy for Sensitive Groups"
    if index <= 200:
        return "Unhealthy"
    if index <= 300:
        return "Very Unhealthy"
    return "Hazardous"

