# Data acquisition plan

## Automated, no credential

`python3 scripts/download_data.py --source airnow` downloads dated AirNow
`daily_data_v2.dat` products over verified HTTPS, extracts the exact Dhaka
PM2.5 record, records every request URL/status, and preserves the source line.
It retries transient failures without disabling certificate verification.

## Credentialed/terms-dependent candidates

- OpenAQ: set `OPENAQ_API_KEY` and use only for provider/sensor metadata or
  cross-checking; never commit the key.
- CAMS/ERA5: accept the Copernicus terms and configure the official CDS client.
  Reanalysis remains a separate modeled product.
- BMD: request/purchase daily Dhaka surface meteorology under BMD's stated
  purpose and redistribution restrictions. The exact seven-variable request,
  indicative price, staging contract, and no-imputation workflow are in
  `docs/bmd_data_request.md`.

## Manual request

Request Bangladesh DoE station-level, quality-controlled records with station
IDs, coordinates, instruments, calibration/QA flags, units, and method-change
history. Any received files must be checksummed in `data/source_manifest.yml`
and analyzed separately before a panel or multi-pollutant scope is considered.
