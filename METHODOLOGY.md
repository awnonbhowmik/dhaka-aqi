# Methodology

## Acquisition and QA

The downloader uses the system CA store with hostname verification, bounded
retries, concurrent dated requests, an explicit user agent, and no credentials.
It preserves the exact matching source line and the SHA-256 digest of each full
response. Schema, station, agency, pollutant, unit, duration, coordinates,
duplicates, negative values, and future dates are validated.

AirNow files contain local-standard-time daily summaries and only valid data.
Monthly completeness requires at least 75% valid daily summaries. Hourly
coverage is unknown and remains null. The terminal month must reach month-end;
therefore March 2025 is partial despite 77.4% numerical day coverage.

## AQI

AQI is calculated from the physical 24-hour PM2.5 concentration using U.S. EPA
breakpoints effective 2024-05-06. Concentration is truncated to 0.1 µg/m3 before
linear interpolation. The source-reported AQI is retained separately because it
was produced under historically applicable breakpoints.

## Inference

- Seasonal Mann-Kendall uses within-calendar-month comparisons and tie correction.
- Seasonal Sen slope uses same-calendar-month pairs and a seeded year-block bootstrap CI.
- Trend-free prewhitening is the serial-correlation sensitivity analysis.
- OLS uses elapsed years, calendar-month fixed effects, and HAC/Newey-West covariance with 12 lags.
- Pettitt's test is a single-change diagnostic, not proof of a source change.
- Seasonality uses Kruskal-Wallis, epsilon-squared, pairwise Mann-Whitney tests, and Holm correction.
- COVID results include a March-August 2020 versus the same 2019 months contrast, an April-June sensitivity, and a month-adjusted interrupted association. No causal claim is made.
- Annual means are compared only with annual thresholds. Daily observations are evaluated against 24-hour forms, including WHO percentile and Bangladesh allowed-crossing language. No monthly value is compared with a daily standard.

## Forecasting

Six models are tested: seasonal naive, naive, drift, ETS, SARIMA, and transparent
time/month regression. Expanding-window origins use 12-month test horizons.
Metrics are MAE, RMSE, MASE, sMAPE, and 95% interval coverage. Positive-support
statistical models use log1p concentration. Model selection is lowest mean MASE.
The empirical forecast is 24 months. 2030 lines are deterministic policy/health
benchmarks and are never called forecasts.

