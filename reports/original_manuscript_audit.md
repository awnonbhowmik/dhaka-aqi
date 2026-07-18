# Original manuscript audit

## Preservation and structure

The user supplied `paper/original/paper_final.docx` on 2026-07-18. The unchanged
file is 2,334,730 bytes with SHA-256
`0d76706cb690403d25d4348fdb8ab986f8ca4b002b79d74f39718b358da4efa3`.
Pandoc extraction produced 11,957 words, 41 references, eight embedded images,
and 1,598 Markdown lines. No Word comments or tracked-change elements were
present.

The title and abstract are duplicated. Figure numbering is nonsequential
(Figure 8 appears before Figures 7, 5, and 6). The paper contains no machine-
readable links from reported table values to generated results.

## Central scientific issue

PM10, NO2, and SO2 are scientifically important criteria pollutants and can
determine a multi-pollutant AQI when their simultaneous concentration-derived
subindices exceed the PM2.5 subindex. Their removal from the revised empirical
analysis is a data-validity decision, not a claim that they are environmentally
unimportant.

| Pollutant | Original treatment | Provenance finding | Revised disposition |
|---|---|---|---|
| PM2.5 | Observed monthly target, 2017-2025 | 2017-2022 values were overwritten by CAMS; later lineage changed | Replaced by identified AirNow/DoS monitor observations |
| PM10 | Observed/supporting indicator and projected value | Source-mixed; modeled/repeated values; fixed-ratio forecast logic in legacy workflow | Excluded pending a station-identified physical series |
| NO2 | Observed concentration and forecast target | Source-mixed CAMS/repeated monthly values; unit and station lineage unresolved | Excluded from empirical trends and forecasts |
| SO2 | Observed concentration and forecast target | Source-mixed CAMS/repeated monthly values; unit and station lineage unresolved | Excluded from empirical trends and forecasts |
| AQI | Independent composite outcome | Scraped/source-mixed historical index; standard changes unresolved | Recalculated PM2.5 subindex under one declared EPA version |

Because only PM2.5 is verified, `dominant_pollutant=pm25` in the revised product
means “the only available calculated subindex,” not proof that PM2.5 dominated
Dhaka's complete multi-pollutant AQI on every day. PM10, NO2, and SO2 can be
reintroduced when simultaneous station IDs, physical units, averaging periods,
QA flags, methods, and completeness are available.

## Claims contradicted by repository evidence

- The claimed homogeneous 108-month observational panel is source-mixed.
- The methods say conflicting observations were not silently overwritten, but
  Git history shows CAMS overwriting pollutant columns.
- December 2025 values rely on web snippets and are not observations.
- The original trend direction is driven by a different/source-mixed record;
  the identified monitor produces method-sensitive increasing evidence.
- Monthly WHO/EPA exceedance calculations mix annual, daily, and hourly forms.
- Annual population/HDI/poverty values repeated over 12 months do not create 12
  independent observations for correlation or EKC inference.
- The normalized rainfall index is not a documented physical rainfall series.
- AQI-PM2.5 correlation is partly definitional and cannot establish an
  independent “driver.”
- COVID percentage changes and rebound claims lack a season-matched,
  meteorology-adjusted causal design.
- Source apportionment cannot be inferred by combining bulk pollutant masses.
- The negative 2030 PM2.5 forecast is physically impossible.

## Internal result contradictions

Table 7 reports a 2030 PM2.5 business-as-usual value of -11.6 ug/m3, whereas
Discussion section 4.5 says approximately 27 ug/m3. Table 7 gives AQI 59.3,
NO2 68.0, and SO2 74.7, while the discussion gives approximately 93, 66.6, and
73.0. These values cannot all describe the same scenario/model output.

## Reference audit

`reports/original_reference_audit.csv` contains all 41 citations. Crossref
resolved 21 supplied DOIs. Five of those resolved to unrelated article titles.
Seven further DOI strings returned no Crossref record; this category includes
some records registered outside Crossref as well as incorrect citations, so each
was assessed individually. Thirteen references have no DOI.

Specific corrections include:

- Pavel et al. belongs in *Frontiers in Sustainable Cities* with DOI
  `10.3389/frsc.2021.681759`, not the journal/DOI printed in the original.
- Qin et al. was published in 2014 with DOI
  `10.1016/j.atmosenv.2014.09.046`, not the supplied 2019 citation/DOI.
- The supplied Sarwar et al. Atmospheric Environment DOI does not resolve and
  the title could not be verified.
- Several supplied DOIs for the Bento, Hossain, Khatun, Samal, and Zhao
  citations resolve to different article titles.

News reports were not retained as evidence of monthly concentrations. The
revised manuscript uses a smaller set of official or verified peer-reviewed
sources and makes no numerical claim depend on the literature alone.
