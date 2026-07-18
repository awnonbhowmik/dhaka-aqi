---
title: "Observed PM2.5 at an Identified U.S. Department of State Monitor in Dhaka, 2019-February 2025"
subtitle: "A provenance audit, robust trend assessment, and short-horizon forecast"
date: "2026-07-17"
---

# Abstract

Air-quality studies can be invalidated when modeled concentrations, scraped
indexes, changing stations, and repeated monthly values are treated as one
observational series. We audited the data lineage of a Dhaka air-quality
repository and rebuilt the analysis from official U.S. EPA AirNow archive
records for the U.S. Department of State monitor `DK1010001` (23.796374 N,
90.424614 E). The archive supplied 2,209 valid preliminary 24-hour
PM2.5 summaries from 2019-01-01 to 2025-03-24.
Using a >=75% valid-day rule and excluding the truncated terminal month yielded
71 complete months from January 2019 through
February 2025. The mean of complete monthly means was
98.65 ug/m3 (median 79.66).
Seasonal Mann-Kendall indicated an increase (p=0.0046), and
month-adjusted HAC regression estimated 3.14
ug/m3/year (95% CI 1.42 to 4.86); however,
trend-free prewhitening was not significant (p=0.18), so trend
inference is sensitive to serial dependence. A season-matched March-August 2020
contrast was -7.32 ug/m3 (95% bootstrap CI
-42.90 to 28.65); the design does not support a
causal lockdown claim. SARIMA achieved the lowest rolling-origin MASE
(0.85) and was used for a 24-month forecast; 2030 values are shown
only as conditional targets/benchmarks. The audit reverses the legacy declining
trend claim and removes unsupported multi-pollutant, source-apportionment,
health-burden, and citywide-exposure conclusions.

**Keywords:** PM2.5; Dhaka; AirNow; data provenance; seasonal Mann-Kendall;
rolling-origin forecasting; reproducibility

# 1. Introduction

Dhaka's air pollution is an important public-health and policy concern, but a
credible time-series study requires consistent measurement lineage. The legacy
repository described a 2017-2025 multi-pollutant monitoring record. Audit of its
code and deleted workbooks instead found CAMS reanalysis values, scraped AQI,
monthly values repeated across days, unresolved station identities, and annual
contextual variables repeated monthly. This revision asks: (1) what seasonal
and temporal patterns are supported by one identified ground monitor; (2) how
sensitive is trend inference to seasonality and serial dependence; (3) what
non-causal association is visible during the 2020 restriction period; and (4)
which short-horizon forecast model outperforms seasonal naive in rolling tests?

# 2. Data and methods

## 2.1 Monitor and provenance

The primary source is the official dated AirNow `daily_data_v2.dat` archive.
The original provider is U.S. Department of State Bangladesh - Dhaka; AirNow is
the distribution system. Each source row reports station `DK1010001` (full code
`050DK1010001`), site name Dhaka, latitude 23.796374, longitude 90.424614,
parameter PM2.5-24hr, unit UG/M3, duration 24 hours, concentration, AQI, and
category. The archive row does not encode instrument metadata, so the
instrument remains unresolved. AirNow values are preliminary and subject to
revision.

The downloader used verified HTTPS and preserved every URL, exact source line,
response SHA-256, and retrieval timestamp. Bangladesh Department of Environment
reports were selected as a separate validation source, but no reproducible raw
machine-readable DoE series was found; no numeric series was extracted from
charts or spliced into AirNow.

## 2.2 QA, aggregation, and cutoff

The pipeline rejected wrong station, agency, pollutant, unit, duration,
duplicates, negative values, and dates beyond retrieval. Asia/Dhaka defines
local days. Monthly summaries require >=75% valid daily AirNow summaries.
Hourly completeness is unknown and remains null. A terminal month must reach
month-end, so March 2025 (24/31 days) is
partial and the cutoff is February 2025. No missing observation was imputed.

AQI was recalculated from physical PM2.5 using U.S. EPA breakpoints effective
6 May 2024. Source-reported AQI was retained separately because historical
values used contemporaneous breakpoints. AQI is a mathematical transform here,
not an independent environmental outcome.

## 2.3 Statistical analysis

Descriptive results include valid counts, mean, median, standard deviation, and
IQR. Monthly climatology includes n and t-based 95% confidence intervals.
Seasonal differences use Kruskal-Wallis, epsilon-squared, and Holm-corrected
pairwise Mann-Whitney tests.

Trend methods comprise seasonal Mann-Kendall with tie correction, same-month
Sen slope with a seeded year-block bootstrap interval, trend-free prewhitening,
and elapsed-time regression with calendar-month fixed effects and 12-lag
Newey-West covariance. Pettitt's test is a change diagnostic only.

COVID analyses compare March-August 2020 with the same months of 2019 and fit a
month-adjusted interrupted association. Meteorology was unavailable; language
is associational rather than causal. Annual means are compared with annual
guidelines, and daily observations with the specified 24-hour forms.

## 2.4 Forecasting and scenarios

Seasonal-naive, naive, drift, ETS, SARIMA, and time/month regression models were
evaluated at three expanding origins with 12-month test horizons. Metrics were
MAE, RMSE, MASE, sMAPE, and 95% interval coverage. Positive-support statistical
models used log1p concentration. The lowest mean MASE selected the model. The
empirical horizon is 24 months. 2030 lines are deterministic benchmarks, not
forecasts; no probabilistic interval is fabricated for policy implementation.

# 3. Results

## 3.1 Completeness and distribution

The observed archive contained 2,209 daily summaries. Of
75 calendar months, 71 met
the primary rule. Complete-month PM2.5 ranged from
25.97 to 234.06 ug/m3; the
IQR was 98.82 ug/m3.

![Observed monthly PM2.5. Open symbols are excluded partial months. Source: U.S. Department of State via AirNow.](figures/observed_pm25_time_series.png){ width=90% }

## 3.2 Seasonality

Calendar-month climatology peaked in month 1 at
200.53 ug/m3 and was lowest in month 7 at
31.33 ug/m3. Seasonal differences were large (Kruskal-Wallis
epsilon-squared 0.86, p<0.001).

![Monthly climatology with 95% confidence intervals; complete months only.](figures/monthly_climatology.png){ width=80% }

## 3.3 Trends and break diagnostic

Seasonal Mann-Kendall yielded z=2.83, p=0.0046. The
seasonal Sen estimate was 2.80 ug/m3/year (95%
bootstrap CI -0.00 to 4.88). HAC regression
estimated 3.14 (95% CI 1.42 to
4.86), but prewhitening yielded p=0.18 with
lag-1 autocorrelation 0.80. We therefore interpret
the direction as suggestive of an increase but not robust across dependence
adjustments. Pettitt's test did not identify a significant single change
(p=0.48).

## 3.4 2020 restriction-period association

The season-matched difference was -7.32 ug/m3 (95% CI
-42.90 to 28.65, p=0.93). The
month-adjusted interrupted coefficient was -10.62 ug/m3
(95% CI -14.88 to -6.35). The discrepancy and lack
of meteorological adjustment preclude a causal conclusion.

## 3.5 Guideline comparisons

All complete annual monitor means (2019-2024) exceeded WHO 2021 (5 ug/m3),
Bangladesh 2022 (35 ug/m3), and U.S. EPA 2024 (9 ug/m3) annual levels. EPA
comparisons are descriptive because formal NAAQS determinations use design-value
procedures and regulatory data.

| Year | PM2.5 mean (ug/m3) | valid-day coverage (%) |
| --- | --- | --- |
| 2019 | 86.4 | 98.4 |
| 2020 | 84.6 | 96.2 |
| 2021 | 98.2 | 94.5 |
| 2022 | 96.3 | 94.8 |
| 2023 | 103.6 | 99.2 |
| 2024 | 98.2 | 98.9 |

![Annual monitor means with annual reference levels; these are not interchangeable with 24-hour standards.](figures/annual_guideline_comparison.png){ width=80% }

## 3.6 Forecast validation

The selected model was sarima (mean MAE 13.64, RMSE
18.05, MASE 0.85, sMAPE 14.8%).
Seasonal naive had MASE
1.03.

![Rolling-origin 12-month backtesting.](figures/forecast_backtesting.png){ width=80% }

![Selected 24-month forecast with 95% prediction intervals.](figures/forecast_intervals.png){ width=90% }

The 2030 benchmarks end at 98.19 ug/m3 for no
additional change, 35 ug/m3 for the Bangladesh annual standard, and 5 ug/m3 for
the WHO AQG. These values express assumptions/targets, not expected outcomes.

![Conditional 2030 benchmark paths, not forecasts.](figures/scenarios_2030.png){ width=80% }

# 4. Discussion

Rebuilding from one identified monitor changes both scope and inference. The
legacy analysis reported a strong PM2.5 decline using a source-mixed 2017-2025
series. The audited 2019-2025 monitor record instead suggests an increase under
seasonal MK and HAC regression, with loss of significance after prewhitening.
The correct conclusion is sensitivity, not a definitive monotonic trajectory.

Seasonality is much more stable: winter concentrations are high and monsoon
concentrations low. This pattern is consistent with meteorological dispersion,
wet removal, seasonal emissions, and regional transport, but the present data
do not apportion sources. PM2.5 cannot be summed with PM10 for source
apportionment, and no chemical speciation, PMF/CMB, inventory, or transport
attribution is implemented.

The season-matched COVID comparison is uncertain. Although the interrupted
coefficient is negative, a single-station ecological time series without
meteorology or mobility controls cannot isolate lockdown effects. Likewise,
AQI-PM2.5 correlation would be definitional and is not presented as discovery.

# 5. Limitations

One monitor does not measure citywide spatial exposure. Values are preliminary
daily summaries; per-hour completeness and archive-row instrument metadata are
unavailable. Three internal months with 66-71% coverage are excluded from trend
inference but retained explicitly in forecasting to preserve the calendar
without imputation. No independent numeric DoE validation or meteorological
series was available. The feed cessation limits the empirical period. Forecast
validation has only three annual rolling origins, and the 24-month outlook
cannot establish 2030 conditions. Trend conclusions depend on serial-correlation
handling. The missing original manuscript also prevents a line-by-line reference
audit and genuine tracked changes.

# 6. Conclusion

The defensible result is a narrow, station-specific PM2.5 study. Pollution at
this monitor remained far above annual health and regulatory reference levels,
with strong seasonality and method-sensitive trend evidence. The data do not
support multi-pollutant trends, citywide exposure, source apportionment, an EKC,
precise attributable mortality, or a causal lockdown claim. A 24-month forecast
can be reported with rolling-origin validation; 2030 values remain conditional
benchmarks.

# Data and code availability

Exact extracted AirNow rows, request URLs, response checksums, standardized
products, source manifest, provenance ledger, code, tests, and generated outputs
are in the repository. Large or terms-restricted CAMS, BMD, OpenAQ, or future
DoE raw files must be downloaded separately under provider terms. The paper was
generated from repository commit `7d8d48e66bfe2d7acb291dd48614957dcd28f963`. Raw retrieval occurred on
2026-07-18 UTC (2026-07-17 America/New_York).

# References

1. World Health Organization. *WHO global air quality guidelines: particulate matter, ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*. 2021. ISBN 978-92-4-003422-8. https://www.who.int/publications/i/item/9789240034228
2. Bangladesh Department of Environment. *Ambient Air Quality in Bangladesh (2018-2023)*. Published 17 June 2025. https://doe.gov.bd/pages/publications/ambient-air-quality-in-bangladesh-2018-2023-7b0bcb-6922da5381fc96cef9eb5f62
3. U.S. Environmental Protection Agency. *Daily Data File Fact Sheet*. AirNow. Accessed 17 July 2026. https://docs.airnowapi.org/docs/DailyDataFactSheet.pdf
4. U.S. Environmental Protection Agency. *Final Updates to the Air Quality Index for Particulate Matter*. 2024. https://www.epa.gov/system/files/documents/2024-02/pm-naaqs-air-quality-index-fact-sheet.pdf
5. U.S. Environmental Protection Agency. *Final Reconsideration of the National Ambient Air Quality Standards for Particulate Matter*. Effective 6 May 2024. https://www.epa.gov/pm-pollution/final-reconsideration-national-ambient-air-quality-standards-particulate-matter-pm
