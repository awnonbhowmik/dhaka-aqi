# BMD meteorology request and ingestion status

## Decision

Bangladesh Meteorological Department (BMD) observations are the preferred
meteorological covariates for this study. The official purchase portal lists a
`Dhaka` surface station with records beginning in 1953 and lists all of the
core variables needed for weather adjustment. Historical observations are not
an anonymous open download: BMD supplies them for a fee and limits use to the
purpose declared in the request.

No purchase was submitted and no BMD observation has been inserted into the
analysis. The manuscript must continue to say that meteorology is unavailable
until BMD delivers a file and its station, units, QA fields, and use permission
have been verified.

## Exact portal request

Use the [BMD Climate Data Portal](https://dataportal.bmd.gov.bd/web/) with:

- Data type: `Surface Meteorological Data`
- Basis: `Daily` (preferred; permits independent completeness checks)
- Period: `01/01/2019` through `31/03/2025`
- Station: `Dhaka` (portal metadata states start year 1953)
- Purpose: `Research`
- Variables:
  - `Rainfall`
  - `Dry bulb temperature`
  - `Maximum Temperature`
  - `Minimum Temperature`
  - `Relative Humidity`
  - `Wind Speed and Wind Direction`
  - `Mean Sea Level Pressure`

The request remarks should state:

> Meteorology adjustment and sensitivity analysis for a peer-reviewed study of
> daily and monthly PM2.5 at the U.S. Department of State/AirNow Dhaka monitor
> DK1010001. Please include station code, latitude, longitude, elevation,
> observation times and averaging definitions, units, missing-value codes,
> per-record QA flags, instrument or site changes, and the applicable citation.
> Please also confirm whether raw files, monthly aggregates, and derived
> regression results may be shared in a public reproducibility repository and
> published in the paper.

The requested end date is more than three months in the past, as required by
the portal. Although the PM2.5 inferential cutoff is February 2025, March is
included so the BMD delivery covers the terminal partial AirNow month and can
be audited on the same calendar.

## Indicative price checked 2026-07-18

The portal's public calculator charges, for daily data, Tk 800 per variable,
Tk 100 per station, and Tk 100 for each five-year block, then adds 15% VAT/TAX.
For seven listed variables, one station, and seven calendar years (two
five-year blocks), the displayed formula gives:

`7 * 800 + 1 * 100 + 2 * 100 = Tk 5,900` before VAT/TAX, or `Tk 6,785` after
15%. This is an estimate; BMD's submitted order and invoice govern. A monthly
request would be cheaper (estimated Tk 4,370 including 15%) but is not preferred
because it prevents daily completeness and aggregation checks.

## Handling a delivered file

1. Preserve the untouched delivery under `data/raw/bmd/`; this directory is
   ignored because BMD's terms may prohibit redistribution.
2. Record its SHA-256 digest, delivery date, invoice/order identifier, and
   permission terms locally.
3. Map—not impute—the provider columns into
   `data/staging/bmd_dhaka_daily.csv` using the contract in
   `src/data_sources/bmd.py`. Do not invent a station identifier or coordinates
   if BMD does not provide them; request clarification.
4. Run `python3 scripts/build_meteorology_dataset.py`. Rainfall is summed,
   ordinary scalar variables are averaged, wind direction uses a circular
   mean, and coverage is calculated independently for every variable-month.
5. Inspect units, QA flags, missingness, site/instrument changes, and distance
   from AirNow station DK1010001 before enabling any adjusted model.
6. Do not commit raw or processed BMD values unless BMD has explicitly permitted
   the intended redistribution.

The public BMD weekly agro-meteorological bulletins were also checked. They
contain preliminary seven-day station summaries, not a stable daily or monthly
machine-readable archive for 2019–2025, omit some requested variables, and
cannot replace the formal station delivery.

## Official metadata endpoints checked 2026-07-18

- Data types: `https://dataportal.bmd.gov.bd/back_api/all-data`
- Surface variables: `https://dataportal.bmd.gov.bd/back_api/variables/Surface%20Meteorological%20Data`
- Surface stations: `https://dataportal.bmd.gov.bd/back_api/stations/Surface%20Meteorological%20Data`
- Public climatology products: `https://bmd.gov.bd/web/en/climate`

The climatology products are useful context, but repeating a normal January,
February, and so on in every year would be collinear with calendar-month effects
and could not control the actual weather differences between 2019 and 2020.
