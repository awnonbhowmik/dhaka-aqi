# Dhaka air quality from official Bangladesh DoE reports

This repository is a fresh, source-first extraction of the Bangladesh Department of Environment (DoE) public air-quality archives. It replaces the earlier mixed AirNow/CAMS workflow.

Paper-ready findings are in [`analysis/RESEARCH_FINDINGS.md`](analysis/RESEARCH_FINDINGS.md), with full tables in [`analysis/dhaka_doe_analysis.xlsx`](analysis/dhaka_doe_analysis.xlsx) and thirteen figures in [`analysis/figures/`](analysis/figures/).

The main deliverable is [`data/processed/dhaka_doe_air_quality.xlsx`](data/processed/dhaka_doe_air_quality.xlsx). Its sheets are:

- `read_me` — interpretation and cautions
- `monthly_dataset` — original-style wide monthly table with pollutant and AQI summary columns
- `daily_dhaka_aqi` — DoE-published Dhaka city AQI, category, and responsible pollutant
- `monthly_report_aqi` — Dhaka daily AQI extracted from Table 6 of newer monthly reports
- `monthly_dhaka` — reported Dhaka-station statistics for PM2.5, PM10, SO2, NO2, CO, and O3
- `population` — annual Bangladesh total, urban, and rural population for 2013–2025
- `hdi` — retained paper HDI and official UNDP same-year verification series
- `source_manifest` — archive page, source URL, local cache path, SHA-256, and extraction status
- `qa_issues` — document-date mismatches, conflicting duplicates, and partial extractions

## Coverage

- Wide monthly dataset: January 2013 through the latest available pollutant or AQI month.
- Numeric AQI in the wide table: January 2022 onward from monthly-report Table 6, with the standalone daily archive used when it provides more reported days.
- Standalone daily archive: 13 February 2023 through the latest DoE attachment available when built.
- Monthly archive: the DoE-linked reports for 2013–2019 and 2022 onward. The public master page does not link 2020 or 2021 year pages.
- The raw PDF/DOCX cache is reproducible but Git-ignored because it is large.

The 2013 start date applies to monthly pollutant statistics, not to numeric AQI. Older reports include AQI category charts but not a recoverable daily numeric series, so pre-2022 `aqi_*` cells remain blank. Pollutant `*_median` columns are retained to resemble the original dataset but remain blank because DoE publishes monthly averages, minima, and maxima—not the underlying daily series needed to calculate a median.

For each pollutant, `monthly_dataset` takes the unweighted mean of the monthly averages reported by Dhaka stations. Its minimum and maximum are the extreme station-level monthly values. CO uses the reported 8-hour block. Station counts, capture rates, units, unit-resolution flags, AQI day counts, coverage, and source basis are included beside the familiar analysis columns.

The `report_date` in the daily table is the date listed on the DoE archive page. `document_aqi_date` is the date printed inside the report. They differ in some source files; those rows are flagged rather than silently altered.

## PM10, SO2, and NO2

They have not been removed. All six reported criteria pollutants are retained in `monthly_dhaka`. The daily reports publish a city AQI and its controlling `responsible_pollutant`; they do not publish a daily contribution from every pollutant. In this archive, DoE identifies PM2.5 as the responsible pollutant for almost every Dhaka daily AQI row. That does not make PM10, SO2, or NO2 scientifically irrelevant—it means the official daily product does not provide a driver decomposition.

Do not combine the daily AQI values and monthly concentration statistics as though they were the same measurement. AQI is a dimensionless index; the monthly table contains reported concentrations, exceedance counts, and capture rates.

## Population and HDI convention

The separate `population` sheet uses the complete annual 2013–2025 Bangladesh national series from UN World Urbanization Prospects 2025. `rural_population` is calculated as total minus urban and checked against the rural values independently reported in the same official workbook. This is national context, not Dhaka-city population: Worldometer's Dhaka figure is an urban-area estimate, so subtracting it from an incompatible total to create “rural Dhaka” would be misleading.

The `hdi` sheet begins in 2013 so it aligns with the DoE pollutant record. The retained 2013–2024 values match `AIDS_BD_2000_2024.xlsx`; official UNDP observation-year values remain separate in `hdi_undp_same_year`. The source workbook's 2023–2024 values are flagged as apparent forward-fills, and the paper's 2025 context value is UNDP's 2023 observation—not a 2025 observation.

## Rebuild

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,analysis]'
.venv/bin/python scripts/build_doe_workbook.py
.venv/bin/python scripts/analyze_doe_dataset.py
.venv/bin/pytest
```

After changing only the retained context CSV or workbook documentation, regenerate the XLSX without downloading or re-extracting the DoE archive:

```bash
.venv/bin/python scripts/build_doe_workbook.py --workbook-only
```

The monthly AQI Table 6 extractor also requires Poppler's `pdftotext` executable. The DoE web server currently omits an issuing intermediate certificate. The pipeline adds the pinned public Sectigo intermediate in `config/` to the normal operating-system trust store; hostname verification and certificate verification remain enabled.

See [`DATA_SOURCES.md`](DATA_SOURCES.md) and [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) before analysis.
