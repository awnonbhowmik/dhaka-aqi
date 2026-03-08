# Dhaka Air Quality Index (AQI) — Reproducible Data Pipeline

A fully reproducible data pipeline for building, validating, and analyzing a monthly Dhaka AQI dataset (2017–2025), suitable for paper-quality analysis.

## Repository Structure

```
dhaka-aqi/
├── build_dataset.py                  # Entry point — runs the full pipeline
├── src/
│   ├── __init__.py
│   └── pipeline.py                   # Modular pipeline functions
├── data/
│   ├── raw/                          # Source Excel files (immutable)
│   │   ├── dhaka_aqi_monthly_enriched_v2_2017_2025_filled.xlsx
│   │   └── dhaka_observed_air_quality_dataset.xlsx
│   └── processed/                    # Final cleaned dataset
│       ├── final_dhaka_aqi_dataset.csv
│       └── final_dhaka_aqi_dataset.xlsx
├── outputs/
│   ├── annual_summary.csv
│   ├── seasonality_summary.csv
│   ├── correlation_matrix.csv
│   ├── correlation_ready_dataset.csv
│   ├── data_dictionary.csv
│   ├── qa_report.md
│   └── figures/                      # Publication-ready figures (300 DPI)
│       ├── monthly_aqi_trend.png
│       ├── monthly_pm25_trend.png
│       ├── aqi_seasonality.png
│       ├── pm25_seasonality.png
│       ├── aqi_yearly_boxplot.png
│       ├── pm25_yearly_boxplot.png
│       └── correlation_heatmap.png
├── notebooks/
│   ├── paper_analysis.ipynb          # Publication-oriented analysis notebook
│   └── dhaka_aqi_insights.ipynb      # Exploratory data insights notebook
└── README.md
```

## Data Sources

| Source File | Description |
|-------------|-------------|
| `dhaka_aqi_monthly_enriched_v2_2017_2025_filled.xlsx` | **Primary source.** Monthly AQI/PM2.5 statistics (2017–2025) enriched with population, HDI, and poverty data from annual context tables. |
| `dhaka_observed_air_quality_dataset.xlsx` | **Validation source.** Daily/hourly observed PM2.5 and AQI from the US Embassy monitor in Dhaka (2016–2021), plus monthly meteorological normals. |

## Processing Workflow

1. **Load & inspect** both Excel workbooks (all sheets, columns, dtypes, missingness).
2. **Clean** the primary monthly dataset: normalize column names to snake_case, parse dates, filter to 2017–2025, remove exact duplicates, flag impossible values.
3. **Merge** meteorological normals (wind speed, rainfall) from the observed dataset for months where available (2017–2021).
4. **Cross-validate** File 1 monthly values against File 2 daily aggregations for the overlap period.
5. **Validate** the final dataset: one row per month, 108 expected months, no duplicates, missingness report.
6. **Export** the final dataset, QA report, data dictionary, summary tables, correlation matrix, and 7 publication-ready figures.

## Assumptions

- File 1 (`Monthly_Enriched` sheet) is the authoritative monthly source, spanning 2017-01 to 2025-12.
- Observed data from File 2 is used for **validation only** — it does not overwrite File 1 values.
- For months beyond the observed period (post-2021), AQI/PM2.5 values may be inferred or web-scraped; the `source_notes` column documents provenance.
- HDI and poverty rate are annual values applied uniformly to all months within a year.
- Meteorological normals (`norm_wind`, `norm_rain`) are only available for 2017–2021; they are `NaN` for later years.
- No data is fabricated. Merge decisions and conflicts are logged to the console.

## Limitations

- The dataset relies on a single monitoring station (US Embassy Dhaka). Spatial representativeness is limited.
- Post-2021 AQI/PM2.5 values may include web-scraped or interpolated data (see `source_notes`).
- Population, HDI, and poverty data are national-level annual estimates, not Dhaka-specific monthly values.
- Coverage gaps are flagged via `is_partial_month` and `coverage_pct` but not imputed.

## How to Reproduce

### Prerequisites

```bash
pip install pandas openpyxl matplotlib seaborn scipy numpy
```

### Run the Pipeline

```bash
python build_dataset.py
```

This will:
- Read raw Excel files from `data/raw/`
- Print schema inspection, cleaning logs, and validation results to the console
- Write the final dataset to `data/processed/`
- Write summary tables, QA report, and figures to `outputs/`

### Explore the Analysis

Open the notebooks in Jupyter:

```bash
jupyter notebook notebooks/paper_analysis.ipynb    # Publication-oriented analysis
jupyter notebook notebooks/dhaka_aqi_insights.ipynb # Exploratory insights
```

## Key Pipeline Functions (`src/pipeline.py`)

| Function | Purpose |
|----------|---------|
| `load_workbook_safely()` | Open an Excel file with clear error messages |
| `inspect_excel_file()` | Read every sheet, print schema, return DataFrames |
| `normalize_columns()` | Standardize column names to snake_case |
| `clean_monthly_dataset()` | Clean and validate the primary monthly dataset |
| `merge_context_variables()` | Merge meteorological normals from File 2 |
| `validate_final_dataset()` | Run QA checks (uniqueness, span, missingness) |
| `build_data_dictionary()` | Generate column-level metadata |
| `generate_figures()` | Create 7 publication-ready PNG figures |
| `export_outputs()` | Write all output artefacts |

## Output Files

| File | Description |
|------|-------------|
| `data/processed/final_dhaka_aqi_dataset.csv` | Final 108-row monthly dataset |
| `data/processed/final_dhaka_aqi_dataset.xlsx` | Same dataset in Excel format |
| `outputs/qa_report.md` | Quality assurance report with missingness and validation results |
| `outputs/data_dictionary.csv` | Column-level metadata (name, dtype, nulls, description) |
| `outputs/annual_summary.csv` | Year-level AQI/PM2.5/population/HDI aggregations |
| `outputs/seasonality_summary.csv` | Season-level AQI/PM2.5 aggregations |
| `outputs/correlation_matrix.csv` | Pearson correlation matrix of numeric variables |
| `outputs/figures/*.png` | 7 publication-ready figures at 300 DPI |