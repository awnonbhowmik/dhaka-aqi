# Data-source audit and decision matrix

Scores are 0 (unacceptable) to 3 (strong). “Preservable” means the retrieved
artifact and request can be checksummed or deterministically repeated. The
matrix scores fitness for this study, not the provider's general scientific
quality.

| Candidate | Authority | Ground | Station-specific | Instrument/QA | Units | Coverage | Stable method | Complete | Reproducible | Licensing | Preservable | Trend suitability | Total / 36 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AirNow/DoS Dhaka PM2.5 | 3 | 3 | 3 | 2 | 3 | 2 | 2 | 2 | 3 | 3 | 3 | 2 | 31 |
| Bangladesh DoE raw CAMS (not openly located) | 3 | 3 | 3 | 2 | 3 | 3 | 2 | 0 | 0 | 1 | 0 | 2 | 22 |
| Bangladesh DoE published reports | 3 | 3 | 2 | 2 | 3 | 2 | 2 | 1 | 2 | 3 | 3 | 1 | 27 |
| OpenAQ v3 | 2 | 3 | 3 | 1 | 3 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 25 |
| Mendeley `9j447cynb9` v2 | 1 | 0 | 0 | 0 | 1 | 3 | 0 | 2 | 2 | 3 | 3 | 0 | 15 |
| CAMS EAC4 | 3 | 0 | 0 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 3 | 1 | 27 |
| Legacy monthly CSV | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 3 | 1 | 0 | 1 | 0 | 7 |
| Legacy pseudo-daily CSV | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 3 | 1 | 0 | 1 | 0 | 7 |

## Candidate assessments

### Bangladesh Department of Environment

DoE is the national authority and reports operation of 31 CAMS in the 2018–
2023 publication. Its 2025 and 2026 monthly-report archives are current. The
2018–2023 report documents national standards and station summaries, making it
the strongest separate validation source. No open raw station-level API or
machine-readable archive with stable IDs, instruments, QA flags, and full Dhaka
history was found. Published values will not be silently extracted into the
AirNow series.

### AirNow / U.S. Department of State

The dated official archive provides a Dhaka record with station code
`DK1010001`, full site code `050DK1010001`, coordinates 23.796374 N,
90.424614 E, agency `U.S. Department of State Bangladesh - Dhaka`, parameter
`PM2.5-24hr`, unit `UG/M3`, duration 24, concentration, AQI, and category. The
daily fact sheet says only valid data are included. Values are preliminary
AirNow summaries rather than regulatory validated data. The archive row does
not identify the instrument, so that field remains unresolved. Public
Department of State transmission ended in March 2025, so the study cannot
honestly extend through 2026.

### OpenAQ

API v3 returns physical units and provider/sensor metadata, but requires an API
key and is an aggregation layer. Versions 1 and 2 were retired on 2025-01-31.
It can distribute or cross-check the same underlying station, but cannot serve
as an independent validation network and is not needed to obtain the selected
AirNow data directly.

### Mendeley dataset

Version 2 was published 2026-01-21 and claims 1,048,551 hourly records across
103 city labels from 2000–2025 under CC BY 4.0. Its public metadata does not
identify original providers, station/sensor IDs, instrumentation, per-record QA,
method transitions, or the origin of pre-November-2020 values. A DOI and large
record count do not establish measurement lineage. It is excluded from both
primary and validation analysis.

### CAMS EAC4

EAC4 is an approximately 80-km, 0.75-degree atmospheric-composition reanalysis
using 4D-Var and 60 model levels. The current documentation covers 2003 through
August 2025. It is scientifically useful as a separate regional modeled series,
not as a monitor replacement. The repository's current CAMS extraction lacks
source-file metadata and uses an unjustified fixed-density gas conversion, so
it is quarantined pending reprocessing.

### Meteorology

BMD is preferred for Dhaka station rainfall, temperature, humidity, and wind,
but its historical portal supplies data for a fee and restricts reuse to the
declared purpose rather than offering an anonymous open download. Its public
metadata lists a Dhaka surface station from 1953 and the needed rainfall,
temperature, relative-humidity, wind, and pressure variables. The free climate
pages are normals or summaries, not date-specific 2019-2025 covariates. The
exact lawful request and prepared ingestion contract are documented in
`docs/bmd_data_request.md`; the processed table remains empty until a delivery
is supplied and verified. ERA5-Land is the documented fallback (0.1-degree
distributed grid, native ~9 km, 1950-present). No normalized rainfall index is
accepted as physical rainfall.

## Standards

- WHO 2021 PM2.5: 5 micrograms/m3 annual and 15 micrograms/m3 24-hour,
  with the 24-hour value expressed as the 99th percentile.
- Bangladesh 2022 PM2.5: 35 micrograms/m3 annual and 65 micrograms/m3
  24-hour; the short-term mean is allowed no more than one exceedance per year.
- U.S. EPA 2024: primary annual PM2.5 NAAQS 9.0 micrograms/m3; 24-hour
  standard retained at 35 micrograms/m3 and interpreted through its regulatory
  design value. EPA's 2024 AQI PM2.5 breakpoints begin 0.0–9.0 (AQI 0–50),
  9.1–35.4 (51–100), 35.5–55.4 (101–150), 55.5–125.4 (151–200),
  125.5–225.4 (201–300), and 225.5–325.4 (301–500).
