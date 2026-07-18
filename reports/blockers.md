# Blockers and external dependencies

## Original manuscript — resolved

The user supplied `paper/original/paper_final.docx` on 2026-07-18. It is
preserved unchanged with its checksum and has now been audited line by line.
The source DOCX contains no Word comments or tracked-change elements, so the
revised DOCX remains a clean revision accompanied by explicit change,
claim-disposition, and reference-audit files rather than being mislabeled as
tracked changes.

## Bangladesh DoE machine-readable observations

The Department of Environment publishes official monthly reports and a 2018–
2023 national monitoring report, but the web search did not identify an open,
documented machine-readable station-level API or raw download for all Dhaka
pollutants. The published report is suitable as a separate validation source;
obtaining raw DoE monitor records may require a formal request.

## AirNow end of public Department of State feed

The accessible archive contains Dhaka PM2.5 daily summaries through early March
2025. The U.S. Department of State stopped public transmission in March 2025.
The processing code therefore derives its cutoff from actual coverage and does
not fill later dates.

## Credentials

OpenAQ API v3 requires an API key. An adapter is documented, but no credential
is stored or requested in the repository. OpenAQ is not needed for the selected
primary series because the original AirNow archive is directly accessible.

## BMD Dhaka meteorology

BMD's public metadata confirms that the Dhaka surface station and the required
rainfall, temperature, humidity, wind, and pressure variables are available.
The historical observations must be ordered through BMD's paid portal and are
subject to purpose and redistribution restrictions. Submitting an order needs
the researcher's identity, organization, contact details, declared purpose,
and payment; none were inferred or submitted. The repository now contains an
exact request specification and a validated daily-to-monthly ingestion path,
but `data/processed/meteorology_monthly.csv` intentionally remains empty until
an authorized delivery is supplied. See `docs/bmd_data_request.md`.
