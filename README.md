# Dhaka air quality from official Bangladesh DoE reports

This repository builds an auditable Dhaka air-quality dataset from Bangladesh Department of Environment (DoE) reports and produces an integrated study of pollution burden, seasonality, episode persistence, monitoring-network heterogeneity, multi-pollutant structure, trend robustness, and carefully bounded national context.

The project uses official DoE observations as its air-quality evidence. National population, HDI, and tree-cover-loss series are retained only as descriptive context; they are not treated as causes of Dhaka pollution.

## Main outputs

- [`data/processed/dhaka_doe_air_quality.xlsx`](data/processed/dhaka_doe_air_quality.xlsx): source observations, context data, manifest, and QA records.
- [`analysis/dhaka_doe_analysis.xlsx`](analysis/dhaka_doe_analysis.xlsx): publication tables for unified daily AQI, source concordance, episodes, stations, historical gases, correlations, trends, and national context.
- [`analysis/figures/`](analysis/figures/): eight numbered, reproducible paper figures.
- [`paper/draft/manuscript.md`](paper/draft/manuscript.md): authoritative manuscript source; matching DOCX and PDF files are in the same directory.
- [`paper/original/original_manifest.yml`](paper/original/original_manifest.yml): checksum and preservation record for the supplied original manuscript.

The analysis uses complementary windows rather than forcing all outcomes into the shortest series. Long-term particulate and resolved-unit gaseous-pollutant context is evaluated for 2013–2019; particulate trends are also evaluated separately for 2022–2025. The joint $\text{AQI}$–particulate analysis uses 2022–2025. A unified daily series selects either monthly-report Table 6 or the standalone archive for each month according to the same completeness rule used by the monthly dataset, yielding 1,457 reported days during 2022–2025. No trend is fitted across the 2020–2021 archive gap. Partial 2026 observations remain available for provenance but are excluded from estimates. The longest observed $\text{AQI}>150$ episode lasted 145 days, from 26 October 2022 through 19 March 2023.

## Data coverage and interpretation

- Monthly pollutant summaries begin in January 2013. The public archive links reports for 2013–2019 and 2022 onward, leaving a documented 2020–2021 gap.
- Historical particulate trends are estimated within 2013–2019 and checked again after excluding partial 2019. Recent trends are estimated within 2022–2025 and checked after excluding 2022. The eras are not pooled because the reporting gap and changing network prevent a defensible continuous trend.
- Numeric monthly $\text{AQI}$ begins in January 2022. Each month uses DoE monthly-report Table 6 or the standalone daily archive, whichever supplies more valid days; ties prefer Table 6.
- The unified daily analysis applies that month-level rule to the underlying daily records. It covers all 365 days in 2022, 363 in 2023, all 366 in 2024, and 363 in 2025.
- On 997 overlapping nonmissing days, monthly-report and standalone $\text{AQI}$ values agree exactly on 93.5% of days and have Pearson correlation $r=0.994$; the largest discrepancies remain available for audit in the analysis workbook.
- The standalone daily table begins on 13 February 2023 and extends through the latest verified DoE attachment recorded in `doe_build_summary.json`.
- $\text{AQI}$ is a dimensionless composite index. It must not be analyzed as though it were a pollutant concentration.
- DoE’s `responsible_pollutant` identifies the pollutant controlling the published daily $\text{AQI}$; it is not a complete source or pollutant-contribution decomposition.
- Monthly city pollutant values are unweighted means of reporting-station monthly averages. Station counts, capture percentages, units, and unit-resolution status accompany the measurements.
- Pollutant median columns are intentionally blank because the reports publish averages and extrema, not the underlying daily station observations required to calculate medians.
- Source-reported `DNA`, `NA`, and blank values remain missing. The pipeline does not fabricate historical $\text{AQI}$ or infer unresolved units.

## Workbook dictionary

The primary workbook contains these sheets:

- `read_me`: interpretation rules and limitations.
- `monthly_dataset`: one row per month with $\text{AQI}$, $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$, monitoring support, and source-basis fields.
- `daily_dhaka_aqi`: published city $\text{AQI}$, archive and document dates, responsible pollutant, category, duplicate selection, QA status, URL, and hash.
- `monthly_report_aqi`: daily Dhaka $\text{AQI}$ extracted from Table 6 of monthly reports.
- `monthly_dhaka`: source-level station, pollutant, statistic, value, unit, page, table, extraction method, URL, and hash.
- `population`: annual UN World Urbanization Prospects national totals and urban/rural values for 2013–2025.
- `population_worldometer`: selected Worldometer rows retained as a sparse UN-derived cross-check.
- `tree_cover_loss`: annual Global Forest Watch national tree-cover loss from all causes for 2001–2024.
- `hdi`: retained Bangladesh HDI context alongside verified same-year UNDP observations.
- `source_manifest`: one record per report attachment, including archive page, source URL, file type, SHA-256, retrieval time, download status, and extraction status.
- `qa_issues`: document-date mismatches, duplicate conflicts, and partial pollutant extractions.

Important field conventions:

- `report_date` is the date shown on the DoE archive; `document_aqi_date` is the date printed inside the report. Differences are flagged, not silently changed.
- `selected_record` identifies the single usable attachment for an unambiguous daily date. Conflicting duplicates remain unselected.
- `{pollutant}_mean` averages reporting-station monthly averages; `{pollutant}_min` and `{pollutant}_max` are the extreme station-level reported minima and maxima.
- `{pollutant}_station_count` and `{pollutant}_mean_data_capture_pct` describe monitoring support.
- `aqi_days_reported`, `aqi_calendar_days`, and `aqi_coverage_pct` describe the monthly $\text{AQI}$ denominator.
- `local_path` is blank and `download_status` is `processed_then_deleted` when a source report was used temporarily and discarded successfully.

## Sources and provenance

Air-quality sources:

- DoE daily AQI archive: <https://doe.gov.bd/pages/static-pages/6922dfba933eb65569e23b0a>
- DoE monthly reports: <https://doe.gov.bd/pages/static-pages/6922de32933eb65569e18f46>
- DoE air-quality monitoring page: <https://doe.gov.bd/pages/static-pages/6922e141933eb65569e2b272>

The extractor reads the rendered archive tables, follows only validated HTTPS report links, verifies PDF/DOCX signatures, and records a SHA-256 digest for every attachment. Raw reports are temporary inputs: after successful extraction and output validation they are deleted, while their URLs, hashes, and extraction status remain in the manifest.

National context sources:

- Population: UN DESA, *World Urbanization Prospects 2025*, `WUP2025-F14-National-Definitions_Pop_by_category.xlsx`. <https://population.un.org/wup/downloads>
- Population cross-check: Worldometer’s selected Bangladesh table, sourced by Worldometer from UN population estimates. <https://www.worldometers.info/world-population/bangladesh-population/>
- HDI retained-series source: `AIDS_BD_2000_2024.xlsx`. <https://github.com/awnonbhowmik/AIDS_BD-Data-Analysis/blob/main/data/AIDS_BD_2000_2024.xlsx>
- HDI verification: UNDP Human Development Report 2025 time series. <https://hdr.undp.org/data-center/documentation-and-downloads>
- Tree-cover loss: Global Forest Watch (2025), distributed by Our World in Data. <https://ourworldindata.org/grapher/tree-cover-loss>
- Original forest-driver dataset: World Resources Institute. <https://datasets.wri.org/datasets/dominant-drivers-of-tree-cover-loss-at-1km>

Verified source-file hashes:

- UN population workbook: `f359eb5677a9a92f6ef8b098e50320064876c131fe7585c09b956c7cf6a7011f`
- Retained HDI workbook: `7154d167ba5e78304f381a4eae325ab6fb637639efec255bb29901d44701fda2`
- UNDP HDR 2025 CSV: `61ed82e5b66c88dfca8ff9fac775c63981ecab6a254862af97acacc41c143117`
- Global Forest Watch-derived CSV downloaded on 6 August 2026: `30944f32b9b9829bcc7d666b46f2db6b141c743b0742e447d514ef39f3f13440`

The UN population series is the complete annual source used in the workbook. Worldometer is a sparse presentation check, not an independent series. Rural population is calculated as $P_{\text{rural}}=P_{\text{total}}-P_{\text{urban}}$ and matches the rural values reported independently in the same UN workbook. The Worldometer Dhaka urban-area estimate is not combined with an incompatible national total.

The retained HDI workbook repeats `0.670` for 2023 and 2024 although its cited secondary page ends in 2022; those entries are flagged as apparent forward-fills. The paper’s 2025 context value is UNDP’s observation for 2023, not a 2025 observation.

Tree-cover loss means stand-replacement disturbance detected in 30 m pixels from all causes. It is not automatically equivalent to permanent deforestation. Population, HDI, and tree-cover loss are national annual measures and are not entered into the Dhaka air-quality models because their geography and time resolution do not match the monitoring outcomes.

Bangladesh Meteorological Department data are not included. Meteorology should be added only as a separately sourced and traceable table.

## Rebuild and test

The monthly AQI extractor requires Poppler’s `pdftotext`. The pinned public Sectigo intermediate in `config/` addresses the incomplete certificate chain currently served by the DoE host while retaining hostname and certificate verification.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,analysis]'
.venv/bin/python scripts/build_doe_workbook.py
.venv/bin/python scripts/analyze_doe_dataset.py
.venv/bin/pytest
```

The full build uses ten workers by default for concurrent downloads and process-parallel extraction. A routine update reuses previously extracted observations, downloads only new or replaced reports, and deletes temporary source files after outputs are written successfully:

```bash
.venv/bin/python scripts/build_doe_workbook.py --incremental --discard-raw
```

To rebuild only the primary XLSX from processed CSV files and context tables:

```bash
.venv/bin/python scripts/build_doe_workbook.py --workbook-only
```

## Daily automation

`scripts/update_doe_daily.py` compares the live DoE archive inventory with `data/processed/doe_source_manifest.csv`. If nothing changed, it exits without rewriting outputs. When the inventory changes, it runs the incremental dataset build, regenerates the integrated analysis workbook and eight figures, and runs the acceptance tests. Failures stop the update without replacing the existing processed observations.

The tracked cron schedule checks every day at 06:15 local time and uses `flock` to prevent overlapping runs. Install it with:

```bash
crontab config/dhaka-aqi.crontab
```

Runtime logs and locks are kept in the ignored `.cron/` directory.

## License

The code is released under the MIT License. Source data retain their providers’ terms and attribution requirements.
