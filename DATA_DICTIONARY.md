# Data dictionary

## `monthly_dataset`

One analysis row per calendar month, shaped to resemble the original paper dataset while using official DoE observations only.

- `month_start`, `year`, `month`, `month_name`, `season`: calendar identifiers and Bangladesh seasonal grouping.
- `{pollutant}_mean`: unweighted mean of the monthly averages reported by Dhaka stations. Pollutants are `pm25`, `pm10`, `no2`, `so2`, `co`, and `o3`; CO uses the 8-hour block.
- `{pollutant}_min`, `{pollutant}_max`: minimum and maximum across the reported station-level monthly minima/maxima.
- `{pollutant}_median`: deliberately blank. DoE does not publish pollutant medians or the underlying daily station series in these summary tables.
- `{pollutant}_station_count`: number of nonmissing station averages contributing to the monthly mean.
- `{pollutant}_mean_data_capture_pct`: mean of available station capture percentages.
- `{pollutant}_unit_as_reported`, `{pollutant}_unit_status`: source unit and whether it was explicit in the parsed summary table. No unit conversion is performed.
- `aqi_mean`, `aqi_median`, `aqi_min`, `aqi_max`: calculated from official daily numeric Dhaka AQI values. They remain blank before 2022.
- `aqi_days_reported`, `aqi_calendar_days`, `aqi_coverage_pct`: numeric observations and coverage denominator.
- `aqi_source_basis`: either monthly-report Table 6 or the standalone daily archive, whichever supplies more valid days for that month; ties prefer Table 6.
- `pollutant_aggregation_basis`, `geography`: repeated interpretation fields that should travel with extracts.

## `daily_dhaka_aqi`

- `report_date`: date listed in the DoE archive table.
- `document_aqi_date`: AQI date printed inside the attachment; falls back to `report_date` only when absent.
- `published_date`: publication date printed inside the attachment, when extractable.
- `city`, `city_as_reported`: normalized and source city labels.
- `aqi_as_reported`, `aqi`: source text and numeric AQI.
- `responsible_pollutant`: pollutant DoE identifies as controlling that AQI.
- `aqi_category_as_reported`: category label preserved verbatim.
- `source_category_scheme`: detected legacy/current DoE category scheme; no cross-scheme relabeling is performed.
- `dhaka_basis_note`: source note describing how DoE combined Dhaka stations.
- `selected_record`: one unambiguous archive record selected for the date.
- `duplicate_date`: more than one attachment exists for the archive date.
- `qa_status`: `ok`, document-date mismatch, or conflicting duplicate.
- `source_url`, `source_sha256`, `source_file_type`, `extraction_method`: lineage fields.

## `monthly_dhaka`

- `report_month`: month assigned by the official year archive page and row label.
- `station_label_as_reported`: DoE station/header label. Legacy stable labels include the documented CAMS identifier and location.
- `parameter`: normalized pollutant: PM2.5, PM10, SO2, NO2, CO, or O3.
- `parameter_as_reported`: source parameter/averaging-period label. CO can have separate 1-hour and 8-hour blocks.
- `statistic_as_reported`: Average, Max, Min, exceedance, or data-capture statistic.
- `value_as_reported`, `value`: original text and parsed numeric value.
- `unit`: source unit, or statistic unit (`days`, `hours`, `percent`). `not_resolved` is used instead of guessing.
- `is_missing`: true for DNA/NA/blank nonnumeric source values.
- `page_number`, `table_number`, `extraction_method`: extraction trace.
- `source_url`, `source_sha256`: attachment lineage.

## `monthly_report_aqi`

- `report_month`: official monthly report period.
- `aqi_date`, `aqi`: date and numeric Dhaka value in the report's daily AQI table.
- `aqi_as_reported`, `is_missing`: source token and explicit DNA/NA status.
- `extraction_method`: Poppler layout-text extraction of Table 6, needed because several tables are rotated in the PDFs.
- `source_url`, `source_sha256`: exact report lineage.

## `population`

Annual Bangladesh national values; they are contextual and are not Dhaka-city measurements.

- `total_population`, `urban_population`: official UN World Urbanization Prospects 2025 annual national-definition counts, converted from thousands to persons.
- `rural_population`: derived as total minus urban.
- `rural_population_un_reported`: rural series independently published in the same official workbook.
- `urban_share_fraction`: urban divided by total.
- `geography`, `geographic_scope`, `definition_basis`: guardrails against treating these as Dhaka-city estimates.
- `rural_derivation`, `validation_status`: formula and equality check against the official rural series.
- `source`, `source_url`, `source_file`, `source_sha256`, `source_note`: exact lineage.

## `hdi`

National Bangladesh values retained by request and kept separate from population.

- `hdi_retained_from_paper`: retained paper/context value. Values for 2013–2024 exactly match the linked AIDS workbook; its 2023–2024 values are apparent forward-fills from 2022. The 2025 value is UNDP's 2023 observation.
- `hdi_undp_same_year`: verified UNDP HDR 2025 observation for the calendar year; available through 2023 only.
- `retained_source`, `retained_source_url`: immediate source of the retained value.
- `retained_reference_url`: secondary HDI reference recorded inside the source workbook, where applicable.
- `undp_verification_url`: official source for `hdi_undp_same_year`.
- `hdi_observation_year_for_retained_value`, `verification_status`: observation-year and lineage checks; a blank observation year avoids presenting a forward-fill as an observation.
- `note`: row-specific differences and interpretation. All source and verification fields must travel with any reuse.

## `source_manifest`

One row per attachment: source kind, period, official archive page, attachment URL, cached path, file type/size, SHA-256, retrieval timestamp, and extraction status (`ok`, `partial`, or `failed`).

## `qa_issues`

Machine-readable exceptions. Warnings preserve questionable source dates or incomplete pollutant blocks; errors identify attachments that could not be downloaded or parsed.
