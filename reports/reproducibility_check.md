# Reproducibility check

The documented offline command `.venv/bin/python scripts/reproduce_all.py`
completed successfully from the committed AirNow extraction. It rebuilt the
standardized products, legacy ledger, analysis, forecasts, figures, tables, and
revised Markdown/DOCX manuscript.

After fixing the Matplotlib SVG hash salt and suppressing only generated SVG
date metadata, the analysis and forecasting steps were run twice. SHA-256
digests were identical for all primary SVG figures, `trend_summary.json`, the
model ranking, forecast table, and scenario table. The test suite includes a
separate deterministic rolling-origin evaluation check.
