# Data dictionary

## `primary_observed_daily.parquet`

| Field | Meaning |
|---|---|
| `timestamp_utc`, `timestamp_local`, `timezone` | Start of source local day, represented in UTC and Asia/Dhaka |
| `station_id`, `station_name`, `latitude`, `longitude` | AirNow station identity and coordinates |
| `pollutant`, `value`, `unit`, `averaging_period` | PM2.5 physical concentration and 24-hour averaging period |
| `provider`, `original_provider` | Distribution system and monitor operator |
| `instrument` | Unresolved because the AirNow archive row does not encode instrument metadata |
| `qa_flag`, `validity` | AirNow valid preliminary summary status |
| `source_file`, `source_version`, `retrieval_date`, `source_id` | Row lineage |
| `measurement_type` | Always `observed_ground` |
| `source_reported_aqi` | AirNow-reported historical AQI, retained but not analyzed as the harmonized AQI |
| `pm25_subindex`, `aqi_category` | Recalculated EPA-2024 AQI and category |
| other pollutant subindices | Null because simultaneous physical concentrations are unavailable |
| `dominant_pollutant` | PM2.5; it is the only calculated subindex |

## `primary_observed_monthly.csv`

Monthly PM2.5 mean, median, standard deviation, range, quartiles, calculated-AQI
summaries, valid/expected days, day coverage percentage, null hourly coverage,
completeness flags/rule, source/station identity, units, and AQI version.

`analysis_monthly.csv` contains only complete months through the reproducible
cutoff. Empty validation, modeled, meteorology, and contextual schemas prevent
measurement classes from being silently combined.

## Legacy provenance ledger

`legacy_observation_provenance.csv` contains one row for every analytical scalar
in the three legacy CSVs, with classification, verification status, exclusion
reason, claimed/verified source, and aggregation notes. `unknown` never enters
the primary analysis.
