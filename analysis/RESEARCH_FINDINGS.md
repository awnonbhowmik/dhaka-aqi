# Research findings from the official DoE Dhaka air-quality dataset

## Scope and defensible study design

The analysis-ready file contains 163 calendar-month rows from January 2013 through July 2026. Pollutant summaries begin in 2013, but the DoE master archive has no linked 2020–2021 monthly reports. Numeric monthly AQI is available from January 2022. Consequently, **2022–2025 is the strongest common comparison window**; 2026 is partial.

The original manuscript's complete 2017–2025 panel and 2030 forecasting design should not be retained unchanged. It would conceal a two-year pollutant-report gap and treat pre-2022 AQI as observed when no comparable numeric series exists in these official reports.

## Descriptive results, 2022–2025

| Variable | Months | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| AQI (index) | 48 | 167.0 | 149.0 | 81.7 | 338.0 |
| PM2.5 (µg/m³) | 47 | 92.4 | 72.7 | 27.5 | 236.7 |
| PM10 (µg/m³) | 47 | 146.0 | 124.6 | 47.0 | 312.9 |
| NO2 (ppb / unresolved in some reports) | 47 | 15.2 | 13.9 | 2.5 | 39.7 |
| SO2 (ppb / unresolved in some reports) | 47 | 6.8 | 5.9 | 0.7 | 17.3 |
| CO (8-hour) (ppm / unresolved in some reports) | 47 | 2.0 | 1.6 | 0.4 | 6.3 |
| O3 (8-hour) (ppb / unresolved in some reports) | 47 | 6.9 | 6.3 | 1.9 | 19.6 |

Annual means show why the overall trend requires nuance:

| Year | AQI | PM2.5 | PM10 | O3 |
|---:|---:|---:|---:|---:|
| 2022 | 197.5 | 107.4 | 153.1 | 4.9 |
| 2023 | 160.6 | 92.4 | 153.8 | 5.9 |
| 2024 | 150.2 | 85.3 | 138.9 | 5.8 |
| 2025 | 159.5 | 83.9 | 137.4 | 11.5 |
AQI fell sharply from the unusually high 2022 average, but rebounded from 150.2 in 2024 to 159.5 in 2025. PM2.5 and PM10 annual means continued downward, while O3 increased sharply in 2025; the O3 result is provisional because source-unit visibility and station coverage vary.

## Seasonality

- **AQI:** highest in Winter (245.3) and lowest in Monsoon (109.8); ratio 2.23. Kruskal–Wallis H=38.71, q=0.0000, ε²=0.81.
- **PM2.5:** highest in Winter (167.7) and lowest in Monsoon (43.5); ratio 3.86. Kruskal–Wallis H=38.92, q=0.0000, ε²=0.84.
- **PM10:** highest in Winter (246.5) and lowest in Monsoon (74.7); ratio 3.30. Kruskal–Wallis H=38.52, q=0.0000, ε²=0.83.
- **NO2:** highest in Winter (19.3) and lowest in Monsoon (11.8); ratio 1.64. Kruskal–Wallis H=3.46, q=0.3798, ε²=0.01.
- **SO2:** highest in Pre-monsoon (8.5) and lowest in Monsoon (5.7); ratio 1.50. Kruskal–Wallis H=2.29, q=0.5148, ε²=0.00.
- **CO (8-hour):** highest in Post-monsoon (3.0) and lowest in Pre-monsoon (1.4); ratio 2.12. Kruskal–Wallis H=5.57, q=0.2353, ε²=0.06.
- **O3 (8-hour):** highest in Pre-monsoon (7.6) and lowest in Winter (6.1); ratio 1.23. Kruskal–Wallis H=4.68, q=0.2755, ε²=0.04.
The median monthly PM10:PM2.5 ratio was 1.66. Its mean was highest in Pre-monsoon (1.84) and lowest in Post-monsoon (1.47). This is descriptive only: the numerator and denominator are separate station-summary aggregates, not paired source-apportionment measurements.

## Season-adjusted temporal trends

- **AQI:** decreasing by 12.43 index per year (95% CI -20.55 to -4.32; q=0.0062; FDR-significant).
- **PM2.5:** decreasing by 8.66 µg/m³ per year (95% CI -11.63 to -5.69; q=0.0000; FDR-significant).
- **PM10:** decreasing by 6.96 µg/m³ per year (95% CI -15.09 to 1.16; q=0.1628; not FDR-significant).
- **NO2:** increasing by 1.32 ppb / unresolved in some reports per year (95% CI -1.84 to 4.47; q=0.4137; not FDR-significant).
- **SO2:** increasing by 0.74 ppb / unresolved in some reports per year (95% CI -1.03 to 2.51; q=0.4137; not FDR-significant).
- **CO (8-hour):** decreasing by 0.37 ppm / unresolved in some reports per year (95% CI -0.87 to 0.12; q=0.1896; not FDR-significant).
- **O3 (8-hour):** increasing by 1.96 ppb / unresolved in some reports per year (95% CI 0.91 to 3.02; q=0.0009; FDR-significant).
These are month-adjusted descriptive trends, not emission-source or causal effects. Station composition, data capture, and unresolved source units can influence apparent changes. The decline in AQI is strongly influenced by the high 2022 baseline and does not imply uninterrupted improvement.

## Pollutant relationships

**Raw monthly correlations (five strongest):**

- aqi–pm25: Spearman ρ=0.98, q=0.0000, n=47.
- pm25–pm10: Spearman ρ=0.95, q=0.0000, n=47.
- aqi–pm10: Spearman ρ=0.93, q=0.0000, n=47.
- no2–so2: Spearman ρ=0.63, q=0.0000, n=47.
- co–o3: Spearman ρ=-0.38, q=0.0318, n=47.

**After removing month-of-year means correlations (five strongest):**

- aqi–pm25: Spearman ρ=0.86, q=0.0000, n=47.
- no2–so2: Spearman ρ=0.59, q=0.0001, n=47.
- pm25–pm10: Spearman ρ=0.48, q=0.0043, n=47.
- no2–co: Spearman ρ=0.46, q=0.0045, n=47.
- pm10–co: Spearman ρ=0.46, q=0.0045, n=47.

The deseasonalized matrix is the better evidence for co-movement beyond the shared winter–monsoon cycle. AQI correlations are still not independent causal tests because AQI is calculated from pollutant sub-indices and DoE identifies the controlling pollutant.

## Daily AQI findings

The selected daily archive contains 1161 unambiguous reports. Across complete calendar years 2024–2025, mean AQI was 155.1. Winter daily AQI averaged 214.6, compared with 108.5 in monsoon. 97.4% of reported winter days exceeded 150, versus 8.2% in monsoon.

DoE named PM2.5 as the responsible pollutant on 99.91% of selected daily reports. This supports PM2.5's dominance in the published AQI product, but it does **not** make PM10, NO2, SO2, CO, or O3 unimportant exposure indicators.

The largest selected daily AQI was 367 on 22 January 2025. Category counts must be reported separately by `source_category_scheme` because DoE labels changed during the archive.

## Data quality and provenance diagnostics

Recent pollutant coverage is 97.9%–97.9% by variable. Only 26 of 47 reported months have a fully explicit parsed unit for each pollutant; unresolved months are retained and flagged rather than silently converted. Mean reporting-station counts range from 2.09 to 2.74.
QA records include 64 document-date mismatches, 2 conflicting daily duplicate dates, and 8 partially extracted monthly reports. Conflicting daily duplicates are excluded from selected-record analysis; all source files and hashes remain in the manifest.

## Population and HDI context

- no2 versus urban_population: ρ=-0.67, q=0.1865, n=11 annual observations.
- no2 versus urban_share_fraction: ρ=-0.67, q=0.1865, n=11 annual observations.
- no2 versus total_population: ρ=-0.67, q=0.1865, n=11 annual observations.
- o3 versus hdi_undp_same_year: ρ=-0.53, q=0.7520, n=9 annual observations.
- no2 versus hdi_undp_same_year: ρ=-0.50, q=0.7520, n=9 annual observations.
These ecological correlations should not be presented as effects of population or development on Dhaka air pollution. Population and HDI are national annual measures, while air-quality observations represent changing Dhaka monitoring stations; both sets of variables also change with calendar time.

## Forecasting readiness

- **AQI:** best 2025 diagnostic was seasonal_naive (MAE=12.87, RMSE=16.78, 12 validation months).
- **PM2.5:** best 2025 diagnostic was additive_ETS (MAE=11.25, RMSE=14.47, 11 validation months).
- **PM10:** best 2025 diagnostic was seasonal_naive (MAE=16.34, RMSE=21.40, 11 validation months).
- **NO2:** best 2025 diagnostic was linear_trend_plus_harmonics (MAE=7.34, RMSE=10.27, 11 validation months).
- **SO2:** best 2025 diagnostic was additive_ETS (MAE=2.13, RMSE=3.03, 11 validation months).
- **CO (8-hour):** best 2025 diagnostic was seasonal_naive (MAE=0.93, RMSE=1.09, 11 validation months).
- **O3 (8-hour):** best 2025 diagnostic was additive_ETS (MAE=3.45, RMSE=4.81, 11 validation months).
Only 36 complete common training months (2022–2024) precede this validation year. A forecast through 2030 would extend five times farther than the one-year validation horizon and would be highly sensitive to station changes and unusual future conditions. The evidence supports seasonal description and short-horizon monitoring benchmarks, not a publication-grade 2030 projection yet.

## Suggested figure selection

For the main paper, prioritize the monthly climatology, daily AQI seasonality, temporal-trend estimates, high-AQI frequency, and particulate matter–AQI relationship figures. Together they cover the core seasonal, temporal, public-health-threshold, and pollutant-association results.

Use the correlation and seasonal-fingerprint heatmaps to compare indicators. The full time series, annual means, PM10:PM2.5 ratio, data availability, and monitoring-support figures are best suited to supplementary material or the data-quality section.

## Recommended paper structure and claims

1. Reframe the study as an **official-source exploratory and temporal analysis**, with forecasting readiness assessed rather than assumed.
2. Use 2022–2025 for the primary multivariate AQI–pollutant analysis; present 2013–2019 pollutant history as a separate legacy-report era.
3. Lead with the winter–monsoon contrast, PM2.5's dominance of published AQI, the persistence of high daily values, and the distinct behavior of gaseous pollutants.
4. Report station counts, data capture, explicit-unit status, AQI coverage, and source-basis fields alongside results.
5. Do not claim a COVID-period effect: the official linked pollutant archive is absent for 2020–2021.
6. Do not interpret national population or HDI correlations causally or as Dhaka-specific demographic effects.
7. Do not fill pollutant medians or pre-2022 numeric AQI. Those values are unavailable, not zero.

## Core limitations

- Monthly pollutant values are aggregates across a changing set of reporting stations, not a fixed-site city mean.
- The wide-table mean is an unweighted mean of station monthly averages; stations with different capture rates receive equal weight.
- Source summary tables sometimes omit or ambiguously expose pollutant units; values are not converted.
- Pollutant medians cannot be recovered from the published summary statistics.
- AQI and pollutant concentrations have different meanings and must not be merged as one measurement scale.
- Daily AQI uses archive report date because some document-internal dates conflict; QA flags remain available.
- Multiple comparisons are controlled with Benjamini–Hochberg q-values, but observational dependence remains.

## Reproducible outputs

- `analysis/dhaka_doe_analysis.xlsx`: all result tables.
- `analysis/figures/`: thirteen paper-ready diagnostic figures.
- `scripts/analyze_doe_dataset.py`: complete analysis code.
- `data/processed/dhaka_doe_air_quality.xlsx`: source dataset and provenance.
