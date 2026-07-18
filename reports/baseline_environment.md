# Baseline environment

- Audit date: 2026-07-17 (America/New_York)
- Starting commit: `deb9b0dc064e9a6603f76415b50aa8f69fb394cf`
- Starting branch: `main`
- Revision branch: `codex/data-provenance-major-revision`
- Initial worktree: clean (`git status --short` returned no entries)
- Python executable: `/usr/bin/python3`
- Python version: 3.14.4
- Repository remote: `https://github.com/awnonbhowmik/dhaka-aqi.git`

## Repository instructions

No `AGENTS.md`, `CONTRIBUTING.md`, `.github/copilot-instructions.md`,
`pyproject.toml`, lock file, requirements file, or notebook-specific execution
documentation existed at the starting commit. The only execution instructions
were in `README.md`.

## Manuscript baseline

Neither requested manuscript path existed at audit time:

- `/mnt/data/paper_final.docx`: missing
- `paper/paper_final.docx`: missing

Consequently, the original DOCX cannot yet be archived, edited, rendered, or
compared. This is recorded in `reports/blockers.md`. No file has been invented
as a substitute for the missing manuscript.

## Original execution status

Both committed notebooks contain executed outputs and no stored exception
objects:

- `main.ipynb`: 47/47 code cells have execution counts; no stored errors.
- `analysis.ipynb`: 26/26 code cells have execution counts; no stored errors.

They cannot be re-executed in the baseline environment: required packages such
as pandas, scipy, statsmodels, scikit-learn, pmdarima, Prophet, geopandas,
openpyxl, nbformat, and pytest were absent. The repository also had no pinned
environment. The original stored outputs are extracted by
`scripts/snapshot_original.py` without re-running analytical code.

## Baseline data checksums

| File | Rows | Coverage | SHA-256 |
|---|---:|---|---|
| `data/final_dhaka_aqi_dataset_clean.csv` | 108 months | 2017-01 to 2025-12 | `eef69d8707cd02425fc8c2c5eb9c6bf36c7a5baec7817962d5b0f01b4ade2144` |
| `data/daily_dhaka_aqi_dataset.csv` | 3,407 days | 2017-01-01 to 2026-05-02 | `5ca0466893632979b9157e517e82dc7d4a3fb6713252c4f944a628bf648fdca8` |
| `data/cams_dhaka_pollutants.csv` | 2,191 days | 2017-01-01 to 2022-12-31 | `79f1c2b63a7f0c403402ae2baec249098f9d9d3e3bec7713e834bb69308440b4` |

The consolidated workbook has four sheets derived from those CSVs and does not
restore missing provenance.

