# Data sources

The selected primary series is the U.S. Department of State Dhaka monitor in
the official AirNow dated archive. The exact station record, request URL,
response digest, and retrieval timestamp are preserved for each day. See
`data/source_manifest.yml` and `docs/data_source_inventory.csv`.

Bangladesh Department of Environment publications are the separate validation
source. No open raw machine-readable DoE station archive was found, so the
validation table remains intentionally empty rather than copying report charts
or splicing stations.

OpenAQ is a possible distribution/metadata layer but requires an API key and is
not independent when it republishes the same embassy monitor. CAMS EAC4 is a
modeled reanalysis and can only enter `cams_modeled_monthly.csv`; the current
table is empty because the legacy conversion cannot be verified from raw files.

The Mendeley dataset `10.17632/9j447cynb9.2` is excluded: its public metadata
does not identify original providers, station/sensor IDs, instruments, QA, or
method continuity sufficient for trend inference.

Authoritative documentation:

- AirNow daily format: https://docs.airnowapi.org/docs/DailyDataFactSheet.pdf
- Bangladesh DoE 2018-2023 report: https://doe.gov.bd/pages/publications/ambient-air-quality-in-bangladesh-2018-2023-7b0bcb-6922da5381fc96cef9eb5f62
- OpenAQ v3: https://docs.openaq.org/about/about
- CAMS EAC4: https://confluence.ecmwf.int/spaces/CKB/pages/83395896/CAMS+Reanalysis+data+documentation
- WHO 2021 AQG: https://www.who.int/publications/i/item/9789240034228
- EPA 2024 AQI update: https://www.epa.gov/system/files/documents/2024-02/pm-naaqs-air-quality-index-fact-sheet.pdf

