# Research findings: official Dhaka air-quality burden

## Study focus

The defensible contribution is an official-source assessment of burden, seasonality, episode persistence, and trend robustness. The primary monthly window is 2022–2025; the primary daily window is 2024–2025. Partial 2026 observations are retained in the dataset but excluded from estimates.

## Central results

- Winter monthly mean AQI was **245.3**, compared with **109.8** during monsoon.
- Winter mean PM2.5 was **167.7 µg/m³**, compared with **43.5 µg/m³** during monsoon.
- AQI exceeded 150 on **48.1%** of reported days in 2024 and **57.4%** in 2025.
- The longest observed high-pollution episode lasted **141 consecutive reported days**, from **2024-11-12** through **2025-04-01**, and peaked at AQI **367**.

## Trend robustness

- AQI: -12.43 per year (95% CI -21.80 to -3.07; HAC month-adjusted model).
- PM25: -8.66 per year (95% CI -12.82 to -4.50; HAC month-adjusted model).
- PM10: -6.96 per year (95% CI -15.26 to 1.34; HAC month-adjusted model).

Trend direction and magnitude should be judged across the full sensitivity table, not from one p-value. It includes an exclusion of 2022, a robust Theil-Sen estimator, capture-weighted station averages, and the near-complete BARC fixed-station series.

## Statistical evidence

- AQI: Kruskal–Wallis H=38.71, q=2.19e-08.
- PM25: Kruskal–Wallis H=38.92, q=2.19e-08.
- PM10: Kruskal–Wallis H=38.52, q=2.19e-08.

## Population and forest context

Bangladesh recorded 47,470 hectares of tree-cover loss across 2022–2024 in the Global Forest Watch-derived series. This is national context, not a Dhaka attribution result: tree-cover loss includes temporary and permanent stand-replacement disturbances and is not synonymous with deforestation. Worldometer population estimates are retained only as a sparse cross-check of their underlying UN series. Neither context series is entered into the air-quality models because geography and temporal resolution do not match Dhaka exposure observations.

## Reproducible outputs

- `analysis/dhaka_doe_analysis.xlsx`: result tables, sensitivity estimates, episodes, QA summaries, and context data.
- `analysis/figures/`: five main paper figures.
- `scripts/analyze_doe_dataset.py`: complete analysis code.
- `data/processed/dhaka_doe_air_quality.xlsx`: source observations and provenance.

## Interpretation boundary

The record supports temporal description and association, not source apportionment or causal effects. Changing station composition, unresolved units for some gases, gaps in the public monthly archive, and the short comparable period remain material limitations.
