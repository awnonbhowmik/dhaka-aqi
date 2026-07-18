# Legacy values excluded

Every analytical scalar in the three legacy CSVs is represented in the provenance ledger.
None enters the revised observed series.

## Counts by classification

- `contextual_annual`: 540
- `modeled_reanalysis`: 26,292
- `monthly_value_repeated_daily`: 5,650
- `scraped_index`: 2,141
- `unknown`: 2,220

## Reasons

- CAMS rows are modeled reanalysis and remain separate from observations.
- Monthly means repeated over days are not independent daily observations.
- Scraped AQI values are indexes, not physical pollutant concentrations.
- Monthly station, instrument, units, provider transitions, and QA flags are unresolved.
- December 2025 is not accepted as observed; web snippets do not establish a monthly series.
- Annual socioeconomic values repeated monthly are contextual only.
