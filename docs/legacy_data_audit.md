# Legacy data audit

## Finding

No committed air-quality dataset is suitable as a primary observational series.
The README's claim of a single Dhaka continuous-monitoring-station dataset is
contradicted by code, Git history, the deleted source workbooks, and the latest
commit message.

## Dataset inventory

| Dataset | Classification | Coverage | Claimed/known lineage | Decision |
|---|---|---|---|---|
| `data/final_dhaka_aqi_dataset_clean.csv` | unknown, scraped, derived, reconstructed, and contextual values mixed in one monthly table | 2017-01 to 2025-12 | Copied from a deleted enriched workbook. That workbook names another local workbook as its base, contains web-snippet evidence, flags recent values as possibly inferred, and does not identify a stable station/instrument. | Exclude from primary analysis. |
| `data/daily_dhaka_aqi_dataset.csv` | modeled reanalysis + scraped AQI + repeated monthly values | 2017-01-01 to 2026-05-02 | Built by `fetch_daily_aqi.py`, then CAMS values overwrite pollutant columns for 2017–2022. AQI comes from monthly values or aqi.in. Pollutants after CAMS coverage are monthly means repeated daily. | Exclude in full. Preserve only for audit. |
| `data/cams_dhaka_pollutants.csv` | modeled reanalysis | 2017-01-01 to 2022-12-31 | CAMS EAC4 nearest grid point; script labels output as raw although it is extracted, converted, and daily-aggregated. Gases use a fixed air-density constant. | Keep separate as legacy modeled data; do not use until product metadata and conversions are rebuilt. |
| `data/dhaka_aqi_consolidated.xlsx` | derived workbook | same as component CSVs | Consolidates daily, monthly, annual contextual, and CAMS tables. | Exclude from analysis. |
| `data/shapefiles/*` | contextual boundaries | BBS 2020 release | Filename asserts BBS origin, but no source URL, license, or checksum was recorded. | Mapping only after license/source verification. |
| `main.ipynb` | generated analytical outputs | monthly legacy table | Contains unsupported EKC, health-burden, source-apportionment, guideline, COVID, and long-horizon forecast analyses. | Archived snapshot only. |
| `analysis.ipynb` | generated analytical outputs | pseudo-daily mixed table | Treats incompatible measurement classes as a daily series. | Archived snapshot only. |

## Git-history reconstruction

The deleted `dhaka_aqi_monthly_enriched_v2_2017_2025_filled.xlsx` described its
2017–2025 monthly table as primary solely because it had the longest coverage.
Its own methodology said post-2021 values might be inferred or web-scraped and
that December 2025 might be externally filled. Its evidence sheet contains
AQI.in and IQAir search/news snippets, explicitly not a full observation series.

The deleted `dhaka_observed_air_quality_dataset.xlsx` had much stronger lineage:
daily PM2.5 for 2016–2021 from an EPA ScienceHub supplement and hourly embassy
files mirrored by TSGreen. It identified calculated AQI separately. However,
the old pipeline used it only for correlation-based validation and retained the
less reliable enriched workbook as primary.

Commit `deb9b0d` then created a pseudo-daily dataset. The implementation:

- disables TLS hostname and certificate validation;
- expands monthly AQI and pollutant means across every calendar day;
- scrapes daily index values from aqi.in;
- describes WAQI `iaqi` values as concentrations despite endpoint semantics;
- overwrites pollutant columns with CAMS modeled values for 2017–2022;
- leaves a source flag describing AQI origin, not pollutant origin;
- converts CAMS gases using one assumed air-density constant;
- retains physical concentrations, calculated/scraped AQI, modeled values, and
  repeated monthly values on the same rows.

## Quantified conflicts

- The monthly table has 108 rows and extends through December 2025, contrary to
  the README statement that it ended in September 2025.
- All 108 months report complete nominal coverage in the deleted enriched
  workbook, even though the source notes admit inferred/web-scraped values.
- In the pseudo-daily table, 41 of 113 months have each pollutant constant on
  every day. These are monthly means mislabeled as daily observations.
- The pseudo-daily file has 3,407 rows: 1,266 marked `monthly_avg` and 2,141
  marked `daily_aqi`. That flag does not reveal that pollutants may be CAMS or
  repeated monthly values.
- PM2.5, PM10, NO2, SO2, CO, and O3 for 2017–2022 are modeled CAMS values but
  coexist with scraped or repeated AQI.
- From 2023 onward, PM2.5, PM10, NO2, and SO2 are monthly values repeated across
  days. CO and O3 are missing.
- December 2025 PM2.5 (92.75), PM10 (402.8), NO2 (79.6), and SO2 (80.7) are
  repeated on each day in the pseudo-daily table; the deleted source workbook
  cites only web snippets as December evidence.
- The current monthly CSV omits the original `hourly_observations`,
  `expected_hours`, `coverage_pct`, `is_partial_month`, and `source_notes`
  fields while the README still documents them.
- National population, HDI, and poverty values were repeated across twelve
  monthly rows and used in inferential correlations/EKC models.

## Station identity and units

The legacy monthly table says only “Dhaka” and does not preserve a station ID,
coordinates, instrument, provider version, or QA flag. Its PM2.5 values cannot
be assigned to the U.S. Embassy or a Bangladesh DoE station record by record.
PM10, NO2, and SO2 provenance is still less specific. The current README's unit
labels therefore do not establish measurement units.

The legacy CAMS table uses µg/m3 after conversion, but the source NetCDF units
and conversion metadata were not preserved. The gases were converted from mass
mixing ratio with a constant density; those values require reprocessing before
scientific use.

## Conclusion

All unresolved legacy values remain `unknown` or their explicit non-observed
class in the row-level provenance ledger. None enters the revised primary
observational analysis. The only salvageable study design is PM2.5-focused and
must begin again from a station-specific official archive.

