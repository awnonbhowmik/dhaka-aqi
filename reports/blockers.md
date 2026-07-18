# Blockers and external dependencies

## Missing original manuscript

The original manuscript is absent from both specified paths. Required DOCX
preservation, text revision, reference-by-reference verification, PDF rendering,
page inspection, and tracked claim comparison cannot be completed until the
user supplies `paper_final.docx` at either `/mnt/data/paper_final.docx` or
`paper/paper_final.docx`.

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

