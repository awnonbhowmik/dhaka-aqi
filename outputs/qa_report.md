# QA Report – Dhaka AQI Final Dataset

## Overview
- **Total rows**: 108
- **Total columns**: 27
- **Date span**: 2017-01-01 to 2025-12-01
- **Expected months (2017-01 to 2025-12)**: 108
- **Actual months present**: 108
- **Months with partial coverage**: 2

### Missing months
None – full coverage from 2017-01 to 2025-12.

## Missingness by column

| Column | Missing |
|--------|---------|
| aqi_max | 0 |
| aqi_mean | 0 |
| aqi_median | 0 |
| aqi_min | 0 |
| coverage_pct | 0 |
| expected_hours | 0 |
| hdi | 0 |
| hourly_observations | 0 |
| is_partial_month | 0 |
| month | 0 |
| month_name | 0 |
| month_start | 0 |
| norm_rain | 48 |
| norm_wind | 48 |
| pm25_max | 0 |
| pm25_mean | 0 |
| pm25_median | 0 |
| pm25_min | 0 |
| population_total | 0 |
| poverty_rate_pct | 0 |
| rural_population | 0 |
| rural_share_pct | 0 |
| season | 0 |
| source_notes | 0 |
| urban_population | 0 |
| urban_share_pct | 0 |
| year | 0 |

## Source file contributions

| Variable group | Authoritative source | Notes |
|----------------|----------------------|-------|
| Monthly AQI (mean/median/min/max) | File 1 – Monthly_Enriched | Covers 2017-2025; includes inferred values for recent months |
| Monthly PM2.5 (mean/median/min/max) | File 1 – Monthly_Enriched | Same as AQI |
| Hourly observations / coverage | File 1 – Monthly_Enriched | Derived from hourly data; coverage_pct and is_partial_month indicate completeness |
| Population (total, urban, rural) | File 1 – Annual_Context sheet joined to Monthly_Enriched | Yearly values repeated for each month |
| HDI | File 1 – Annual_Context | Yearly values |
| Poverty rate | File 1 – Annual_Context | Yearly values |
| Meteorological normals (norm_wind, norm_rain) | File 2 – Monthly_Meteo_Norm | Available for 2017-2021 only; NaN elsewhere |
| Observed daily PM2.5 (used for validation only) | File 2 – Daily_Observed | 2016-2021; not merged into final but used for cross-checks |

## Assumptions and limitations

1. File 1 is treated as the authoritative monthly source because it spans 2017–2025 and was already enriched with population/HDI/poverty data.
2. Observed data from File 2 was preferred for validation but not used to overwrite File 1 values, because File 1 already integrates those observations for months where they overlap.
3. For months beyond the observed period (post-2021), AQI/PM2.5 values in File 1 may be inferred or web-scraped; the `source_notes` column documents this.
4. HDI and poverty_rate_pct are annual values applied uniformly to all months within a year.
5. norm_wind and norm_rain are only available for 2017-2021.
6. December 2025 and other recent months may rely on externally filled values; see `source_notes` and `is_partial_month`.

## Columns: derived vs directly observed

| Column | Type |
|--------|------|
| month_start | Derived / computed |
| year | Derived / computed |
| month | Derived / computed |
| month_name | Derived / computed |
| hourly_observations | Directly observed / reported |
| expected_hours | Directly observed / reported |
| coverage_pct | Derived / computed |
| pm25_mean | Directly observed / reported |
| pm25_median | Directly observed / reported |
| pm25_min | Directly observed / reported |
| pm25_max | Directly observed / reported |
| aqi_mean | Directly observed / reported |
| aqi_median | Directly observed / reported |
| aqi_min | Directly observed / reported |
| aqi_max | Directly observed / reported |
| is_partial_month | Derived / computed |
| population_total | Directly observed / reported |
| urban_population | Directly observed / reported |
| rural_population | Directly observed / reported |
| urban_share_pct | Derived / computed |
| rural_share_pct | Derived / computed |
| hdi | Directly observed / reported |
| poverty_rate_pct | Directly observed / reported |
| source_notes | Derived / computed |
| norm_wind | Directly observed / reported |
| norm_rain | Directly observed / reported |
| season | Derived / computed |
