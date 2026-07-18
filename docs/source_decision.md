# Formal source and scope decision

## Decision: Path B — PM2.5-focused salvage study

The primary observational series is the U.S. Department of State monitor at
the U.S. Embassy/mission site in Dhaka, distributed by the official AirNow
dated archive:

- station ID: `DK1010001` (full AirNow code `050DK1010001`)
- station name: Dhaka
- coordinates: 23.796374 N, 90.424614 E
- provider: U.S. Department of State Bangladesh - Dhaka
- parameter: 24-hour PM2.5 concentration
- unit: micrograms per cubic metre (`UG/M3` in source)
- measurement: ground monitor; BAM-1020 is reported in the literature
- data status: preliminary archive summaries, not regulatory validated
- accessible coverage: determined by the downloader, ending in March 2025

The Bangladesh Department of Environment network/report is the separate
validation source. Its results are never appended to AirNow records. OpenAQ can
be used only as a distribution/metadata cross-check because an OpenAQ record
from the same embassy monitor is not independent.

## Why Path A is rejected

No homogeneous, station-identified, reproducibly downloadable observational
source was verified for PM10, NO2, SO2, CO, and O3 across the study period. The
legacy multi-pollutant table combines unknown monthly values, repeated values,
scraped indexes, and CAMS modeled fields. Retaining a multi-pollutant title or
analysis would force unsupported observations.

## Analysis scope

The revised empirical scope is:

- observed daily PM2.5 at one identified Dhaka monitor;
- complete monthly PM2.5 summaries based on at least 75% valid daily values;
- AQI recalculated consistently with U.S. EPA 2024 PM2.5 breakpoints;
- seasonal description, robust trend analysis, season-matched COVID analysis,
  guideline comparisons at their correct averaging periods, and rolling-origin
  short-horizon forecasts;
- no citywide exposure inference, multi-pollutant correlation, EKC, source
  apportionment, precise health burden, or PM10-ratio forecast.

The end date is the last month satisfying the completeness rule. A later
calendar date is never filled or called observed.

