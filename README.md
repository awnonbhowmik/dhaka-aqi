# Dhaka City Air Quality Analysis (2017–2025) and Projections (2026–2030)

A fully reproducible, publication-ready analysis of monthly air quality data
from the Dhaka continuous ambient air-quality monitoring station. Covers
descriptive statistics, statistical trend testing, seasonal analysis,
meteorological drivers, COVID-19 impact assessment, health burden estimation,
and multi-model ensemble forecasting through 2030.

---

## Repository structure

```
dhaka-aqi/
├── main.ipynb                         # Complete analysis notebook (run this)
├── README.md
├── .gitignore
│
├── src/                               # Python modules (imported by notebook)
│   ├── __init__.py
│   ├── config.py                      # Colour palettes, WHO/EPA limits, rcParams
│   ├── data_loader.py                 # Load & prepare the monthly dataset
│   ├── analysis.py                    # Statistical functions (MK trend, ADF, EKC, …)
│   ├── forecasting.py                 # OLS / ETS / SARIMA / Prophet / Ensemble
│   ├── viz.py                         # Seaborn-first publication figures
│   └── map_utils.py                   # Study area map (GeoPandas)
│
├── data/
│   ├── final_dhaka_aqi_dataset_clean.csv   # Primary monthly dataset (108 rows)
│   └── shapefiles/                    # Bangladesh admin boundaries (ADM2 + ADM3)
│       ├── bgd_admbnda_adm2_bbs_20201113.*  # District level (64 districts)
│       └── bgd_admbnda_adm3_bbs_20201113.*  # Upazila level (507 sub-districts)
│
└── figures/                           # Auto-generated PNG outputs (300 DPI)
```

---

## Dataset

**File**: `data/final_dhaka_aqi_dataset_clean.csv`  
**Rows**: 108 monthly observations (January 2017 – September 2025)  
**Source**: Dhaka continuous ambient air-quality monitoring station

### Column reference

| Column | Description | Unit |
|---|---|---|
| `month_start` | First day of the month (ISO 8601) | date |
| `year`, `month` | Calendar year and month number | — |
| `season` | Meteorological season (Winter / Pre-monsoon / Monsoon / Post-monsoon) | — |
| `pm25_mean/median/min/max` | Monthly PM₂.₅ statistics | µg/m³ |
| `pm10_mean/median/min/max` | Monthly PM₁₀ statistics | µg/m³ |
| `no2_mean/median/min/max` | Monthly NO₂ statistics | µg/m³ |
| `so2_mean/median/min/max` | Monthly SO₂ statistics | µg/m³ |
| `aqi_mean/median/min/max` | Monthly AQI (US EPA scale) | — |
| `hourly_observations` | Number of valid hourly readings | count |
| `expected_hours` | Total hours in the month | count |
| `coverage_pct` | Data completeness | % |
| `is_partial_month` | Flag if < 80 % hourly coverage | bool |
| `population_total` | Total Bangladesh population | persons |
| `urban_population` | Urban population | persons |
| `urban_share_pct` | Urban share of total population | % |
| `hdi` | Human Development Index (national) | 0–1 |
| `poverty_rate_pct` | Poverty headcount ratio (national) | % |
| `norm_rain` | Normalised monthly rainfall (2017–2021 only) | 0–1 |

### Seasons

| Season | Months | Typical meteorology |
|---|---|---|
| **Winter** | December–February | Cool and dry; NE trade winds; peak PM season |
| **Pre-monsoon** | March–May | Hot and dry; dust events; pre-monsoon thunderstorms |
| **Monsoon** | June–September | Heavy rainfall; monsoon washout of particulates |
| **Post-monsoon** | October–November | Transitional; PM begins rising again |

### Key statistics (2017–2025 mean)

| Variable | Annual mean | WHO annual limit | × WHO limit |
|---|---|---|---|
| PM₂.₅ | ~130 µg/m³ | 5 µg/m³ | ~26× |
| PM₁₀ | ~280 µg/m³ | 15 µg/m³ | ~19× |
| NO₂ | ~35 µg/m³ | 10 µg/m³ | ~3.5× |
| SO₂ | ~28 µg/m³ | 40 µg/m³ | < 1× |

---

## Shapefiles

Source: Bangladesh Bureau of Statistics (BBS), released November 2020.  
Only the two levels needed for the study area map are retained:

| File | Level | Features | Description |
|---|---|---|---|
| `bgd_admbnda_adm2_*` | ADM2 | 64 | District boundaries |
| `bgd_admbnda_adm3_*` | ADM3 | 507 | Upazila (sub-district) boundaries |

Filtered to **Dhaka district** (`ADM2_EN = "Dhaka"`, `ADM2_PCODE = "BD3026"`)
which contains **46 upazilas** ranging from dense urban core to peri-urban
and rural zones.

---

## Analysis sections

| § | Section | Key outputs |
|---|---|---|
| 0 | **Study Area Map** | `fig00_study_area.png` |
| 1 | **Setup & Data Loading** | Dataset summary, column types |
| 2 | **Descriptive Statistics & Data Quality** | `fig01_distributions.png`, `fig02_cv_analysis.png`, `fig03_aqi_category_pie.png` |
| 3 | **Long-Term Temporal Trends** | Mann-Kendall table, OLS table, `fig04_monthly_time_series.png`, `fig05_annual_trends.png` |
| 4 | **Seasonal & Monthly Climatology** | `fig06_monthly_climatology.png`, `fig07_aqi_heatmap.png`, `fig08_seasonal_distributions.png` |
| 5 | **Time-Series Diagnostics** | ADF table, `fig09_acf_pacf.png`, `fig10_stl_decomposition.png`, `fig11_anomaly_detection.png`, `fig12_pm_ratio.png` |
| 6 | **Pollutant Inter-relationships** | `fig13_correlation_matrix.png`, `fig14_pairwise_scatter.png`, `fig15_source_apportionment.png` |
| 7 | **Statistical Testing — Seasonal** | Kruskal-Wallis table, Dunn's post-hoc tables |
| 8 | **COVID-19 Impact Assessment** | Cohen's d table, `fig16_covid_impact.png` |
| 9 | **Meteorological Drivers** | `fig17_rainfall_scatter.png`, `fig18_rainfall_dual_axis.png`, `fig19_rolling_correlation.png` |
| 10 | **Socioeconomic Context & EKC** | EKC test output, `fig20_socioeconomic.png`, `fig21_per_capita.png` |
| 11 | **WHO & EPA Guideline Exceedance** | Exceedance table, `fig22_exceedance.png` |
| 12 | **Health Burden Estimation** | IHME GBD CRF table, `fig23_health_burden.png` |
| 13 | **Regional Comparison** | `fig24_regional_comparison.png` |
| 14 | **Predictive Modelling — Evaluation** | MAE/RMSE/MAPE table, `fig25_model_evaluation.png` |
| 15 | **Multi-Model Forecasts to 2030** | Ensemble summary table, `fig26_forecasts_2030.png` |
| 16 | **Policy Scenario Projections** | Scenario table, `fig27_scenario_projections.png` |
| 17 | **Conclusions & Limitations** | Summary, limitations table |

---

## Statistical methods

| Method | Purpose |
|---|---|
| Mann-Kendall test | Non-parametric monotonic trend |
| OLS regression | Annual trend slope and R² |
| Augmented Dickey-Fuller | Unit root / stationarity testing |
| Shapiro-Wilk | Normality testing of pollutant distributions |
| Kruskal-Wallis | Seasonal group differences |
| Dunn's post-hoc (Bonferroni) | Pairwise seasonal comparisons |
| Cohen's d | COVID-19 lockdown effect size |
| Mann-Whitney U | Non-parametric group significance |
| Spearman ρ | Pollutant and socioeconomic correlations |
| STL decomposition | Trend / seasonal / residual decomposition |
| IHME GBD CRF | PM₂.₅-attributable health burden |
| EKC quadratic OLS | Environmental Kuznets Curve test |

---

## Predictive models

| Model | Library | Configuration |
|---|---|---|
| OLS | `sklearn` | Year + month dummy variables |
| ETS (Holt-Winters) | `statsmodels` | Additive trend + additive seasonal (period=12) |
| SARIMA | `pmdarima.auto_arima` | AIC order selection; m=12; max(p,q)=3 |
| Prophet | `prophet` | Multiplicative yearly seasonality |
| Ensemble | — | Simple mean of all four model forecasts |

**Train**: January 2017 – December 2023  
**Test**: January 2024 – September 2025  
**Forecast**: October 2025 – December 2030

---

## Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn \
            scikit-posthocs pmdarima prophet geopandas
```

Tested with Python 3.10+, geopandas ≥ 1.0, pandas ≥ 2.0.

---

## Running the analysis

1. Clone the repository and navigate to the project root.
2. Install dependencies (see Prerequisites above).
3. Open `main.ipynb` in Jupyter Lab or VS Code.
4. Run all cells in order (§ 0 through § 17).
5. All figures are saved automatically to `figures/` at 300 DPI.

> **Note**: § 14 and § 15 fit SARIMA and Prophet models. These cells may take
> 3–10 minutes depending on hardware.

---

## Limitations

- **Single monitoring station** — may not represent spatial variability across 1,463 km².
- **Monthly temporal resolution** — short-term episodes and diurnal variation not captured.
- **Rainfall data gap** — `norm_rain` available only for 2017–2021.
- **National socioeconomic data** — HDI/poverty figures are national, not Dhaka-specific.
- **Health burden** — order-of-magnitude estimates using national CMR.
- **Forecast uncertainty** — structural stationarity assumed; policy shocks not modelled.
