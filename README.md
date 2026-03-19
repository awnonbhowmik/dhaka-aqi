# Dhaka Air Quality Index (AQI) — Reproducible Data Pipeline

A fully reproducible data pipeline for building, validating, and analyzing a monthly Dhaka AQI dataset (2017–2025), suitable for paper-quality analysis.

## Repository Structure (2026)

```
dhaka-aqi/
├── build_dataset.py                  # Entry point — runs the full pipeline
├── src/
│   ├── __init__.py
│   └── pipeline.py                   # Modular pipeline functions
├── data/
│   ├── final_dhaka_aqi_dataset.csv
│   ├── final_dhaka_aqi_dataset_clean.csv
│   └── final_dhaka_aqi_dataset.xlsx
├── notebooks/
│   └── dhaka_aqi_complete_merged_figures.ipynb   # Main analysis & figures notebook
└── README.md
```

## Data Files

| File                                     | Description                                                                                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `data/final_dhaka_aqi_dataset_clean.csv` | Main cleaned monthly dataset (2017–2025) with AQI, PM2.5, PM10, NO2, SO2, population, HDI, poverty, rainfall, and season columns. |
| `data/final_dhaka_aqi_dataset.csv`       | Alternate/legacy version of the dataset.                                                                                          |
| `data/final_dhaka_aqi_dataset.xlsx`      | Excel version of the dataset.                                                                                                     |

## Main Analysis Notebook

The primary analysis and all publication-ready figures are in:

- `notebooks/dhaka_aqi_complete_merged_figures.ipynb`

This notebook:

- Loads the cleaned dataset
- Performs descriptive statistics, missing value analysis, and visualizations
- Computes trends, seasonal patterns, and statistical tests
- Generates all figures for publication (titles/suptitles removed for journal formatting)
- Contains extensive code comments and section headers for reproducibility

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

## How to Use

### Prerequisites

```bash
pip install pandas openpyxl matplotlib seaborn scipy numpy scikit-learn pmdarima prophet
```

### Run the Data Pipeline (if needed)

```bash
python build_dataset.py
```

### Explore the Analysis

Open the main notebook in Jupyter:

```bash
jupyter notebook notebooks/dhaka_aqi_complete_merged_figures.ipynb
```

This notebook is self-contained and will reproduce all figures and tables for the paper.

## Key Pipeline Functions (`src/pipeline.py`)

| Function                  | Purpose                                                  |
| ------------------------- | -------------------------------------------------------- |
| `main()`                  | Entry point for the data pipeline (see build_dataset.py) |
| `clean_monthly_dataset()` | Clean and validate the primary monthly dataset           |
| ...                       | ... (see code for details)                               |

## Output Files

| File                                     | Description                  |
| ---------------------------------------- | ---------------------------- |
| `data/final_dhaka_aqi_dataset_clean.csv` | Main cleaned dataset (CSV)   |
| `data/final_dhaka_aqi_dataset.xlsx`      | Main cleaned dataset (Excel) |

All figures and tables are generated directly in the notebook.

## Recent Changes (2026)

- All figure titles and suptitles have been removed for journal submission formatting.
- Imports in the notebook have been fixed and streamlined.
- Additional code comments and section headers have been added for clarity.
- Only one main notebook is now maintained: `dhaka_aqi_complete_merged_figures.ipynb`.
- Data and code are fully reproducible from the notebook and pipeline script.
