# Original versus revised results

| Item | Original repository | Revised audit |
|---|---|---|
| Study period | 2017-2025/2030; README and CSV dates conflicted | Observed 2019-01-01 to 2025-03-24; complete analysis through 2025-02 |
| Source | Claimed one Dhaka CAMS station | AirNow/DoS station DK1010001; exact rows/checksums |
| Stations | Unidentified | One, 23.796374 N 90.424614 E |
| Observations | 108 monthly plus 3,407 pseudo-daily rows | 2,209 observed daily summaries; 71 complete months |
| PM2.5 level | README approximately 130 ug/m3 | Mean of complete monthly means 98.65 ug/m3; daily-weighted 2024 mean 98.19 |
| Trend | MK decreasing; simple OLS -13.57 ug/m3/year | Seasonal Sen +2.80; HAC +3.14; prewhitening p=0.18; method-sensitive |
| Seasonal peak | Winter, based mixed source | January climatology 200.53 ug/m3; strong seasonal effect |
| Rainfall | Undocumented normalized index | Removed; physical meteorology unavailable |
| COVID | Whole pre-period versus March-August; causal framing | Same-month contrast -7.32 (CI -42.90, 28.65); association only |
| Guidelines | Monthly exceedances against mismatched periods | Separate annual and daily/form-specific tables |
| Model evaluation | One 2024-2025 split; unweighted ensemble MAE 22.13 | Three rolling origins; sarima MASE 0.85, seasonal naive 1.03 |
| Forecast horizon | 2026-2030 forecast | 24 months through 2027-02 |
| 2030 | Precise forecasts including ratio-derived PM10 | Deterministic PM2.5 benchmarks only; no uncertainty invented |
| Claims removed | None | Multi-pollutant trends, source apportionment, EKC, mortality count, citywide exposure, PM10-ratio forecast |
