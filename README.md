# Dhaka PM2.5 monitor audit and analysis

This repository now supports a PM2.5-focused study of one identified ground
monitor in Dhaka. It does **not** represent citywide exposure and does not use
the legacy multi-pollutant CSVs as observations.

## Defensible study scope

- Primary provider: U.S. Department of State, distributed by US EPA AirNow
- Station: `DK1010001` (`050DK1010001`), Dhaka
- Coordinates: 23.796374 N, 90.424614 E
- Pollutant: 24-hour PM2.5, µg/m3
- Raw observed coverage: 2019-01-01 to 2025-03-24
- Complete analysis period: 2019-01 through 2025-02
- Completeness: at least 75% valid daily summaries; a terminal month must reach
  calendar month-end
- AQI: recalculated from physical PM2.5 with U.S. EPA 2024 breakpoints
- Validation source: Bangladesh Department of Environment reports/network,
  retained separately

March 2025 is partial and excluded from the primary monthly analysis. No later
date is reconstructed. The public Department of State feed ended in March 2025.

## Main findings

Across 71 complete months, the mean of monthly PM2.5 means is 98.65 µg/m3
(median 79.66). Annual monitor means for every complete year from 2019 through
2024 exceeded the WHO 2021 annual guideline, Bangladesh 2022 annual standard,
and current U.S. EPA annual standard (the latter comparison is descriptive, not
a formal NAAQS compliance determination).

Trend evidence is method-sensitive. Seasonal Mann-Kendall is positive
(`p=0.0046`) and month-adjusted HAC regression estimates +3.14 µg/m3/year
(95% CI 1.42 to 4.86), while trend-free prewhitening is not significant
(`p=0.18`). The season-matched March-August 2020 contrast is -7.32 µg/m3
(95% bootstrap CI -42.90 to 28.65); causal lockdown language is not supported.

Rolling-origin 12-month backtesting selects SARIMA (mean MASE 0.85) over the
seasonal-naive baseline (1.03). The empirical forecast horizon is 24 months,
March 2025 through February 2027. The 2030 outputs are deterministic benchmarks,
not forecasts.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/reproduce_all.py
```

The command uses the committed, checksummed AirNow extraction and does not need
network access. To refresh the official archive first:

```bash
.venv/bin/python scripts/download_data.py --start 2019-01-01 --end 2025-04-30
.venv/bin/python scripts/reproduce_all.py
```

Pandoc is required to regenerate the DOCX. LibreOffice and `pdftoppm` are used
for the optional PDF/page-render inspection.

Run quality checks:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src scripts tests
```

## Repository map

- `data/raw/airnow/`: exact extracted source lines and response hashes
- `data/provenance/`: request log and row-level legacy ledger
- `data/processed/`: standardized observed and explicitly separate empty/model schemas
- `src/`: acquisition, AQI, QA, statistics, and forecasting functions
- `scripts/`: one-purpose pipeline commands and end-to-end runner
- `tables/`, `figures/`: generated, manuscript-traceable results
- `docs/`: source audit, formal source decision, data documentation
- `reports/`: baseline snapshot, QA, forecasting, blockers, result comparison
- `paper/revised/`: provisional revised manuscript and claim traceability

The root `main.ipynb` and `analysis.ipynb` are legacy artifacts. Their stored
outputs are preserved under `reports/original_results_snapshot/`; do not use
them as the revised analysis.

See [DATA_SOURCES.md](DATA_SOURCES.md), [DATA_DICTIONARY.md](DATA_DICTIONARY.md),
[METHODOLOGY.md](METHODOLOGY.md), and [LIMITATIONS.md](LIMITATIONS.md).

