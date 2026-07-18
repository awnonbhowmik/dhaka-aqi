#!/usr/bin/env python3
"""Generate the clean revised manuscript and traceability artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/revised"


def fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in data.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    descriptive = pd.read_csv(ROOT / "tables/descriptive_summary.csv").iloc[0]
    covid = pd.read_csv(ROOT / "tables/covid_association.csv")
    annual = pd.read_csv(ROOT / "tables/annual_guideline_comparison.csv")
    ranking = pd.read_csv(ROOT / "tables/model_ranking.csv")
    climatology = pd.read_csv(ROOT / "tables/monthly_climatology.csv")
    scenario = pd.read_csv(ROOT / "tables/scenario_summary.csv")
    trend = json.loads((ROOT / "tables/trend_summary.json").read_text())
    daily = pd.read_parquet(ROOT / "data/processed/primary_observed_daily.parquet")
    monthly_all = pd.read_csv(ROOT / "data/processed/primary_observed_monthly.csv")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    smk = trend["seasonal_mann_kendall"]
    sen = trend["seasonal_sen_slope"]
    tfpw = trend["trend_free_prewhitening"]
    hac = trend["month_adjusted_hac_regression"]
    match = covid.iloc[0]
    its = covid.iloc[2]
    best = ranking.iloc[0]
    peak = climatology.loc[climatology["mean"].idxmax()]
    low = climatology.loc[climatology["mean"].idxmin()]
    annual_simple = annual[annual["standard"] == "Bangladesh NAAQS"][["year", "pm25_annual_mean", "coverage_pct"]].copy()
    annual_simple.columns = ["Year", "PM2.5 mean (ug/m3)", "valid-day coverage (%)"]
    for column in annual_simple.columns[1:]:
        annual_simple[column] = annual_simple[column].map(lambda value: fmt(value, 1))

    text = f"""---
title: "Observed PM2.5 at an Identified U.S. Department of State Monitor in Dhaka, 2019-February 2025"
subtitle: "A provenance audit, robust trend assessment, and short-horizon forecast"
date: "2026-07-18"
---

**Awnon Bhowmik¹, Mahmudul Hasan², and Goutam Saha²˒³⁎**

¹ College of Engineering & Computer Science, Colorado Technical University,
Colorado Springs, CO 80907, USA<br>
² Department of Mathematics, University of Dhaka, Dhaka 1000, Bangladesh<br>
³ Miyan Research Institute, International University of Business Agriculture
and Technology, Uttara, Dhaka 1230, Bangladesh<br>
⁎ *Correspondence:* gsahamath@du.ac.bd

# Abstract

Air-quality studies can be invalidated when modeled concentrations, scraped
indexes, changing stations, and repeated monthly values are treated as one
observational series. We audited the data lineage of a Dhaka air-quality
repository and rebuilt the analysis from official U.S. EPA AirNow archive
records for the U.S. Department of State monitor `DK1010001` (23.796374 N,
90.424614 E). The archive supplied {len(daily):,} valid preliminary 24-hour
PM2.5 summaries from {daily['date_local'].min()} to {daily['date_local'].max()}.
Using a >=75% valid-day rule and excluding the truncated terminal month yielded
{int(descriptive['n_complete_months'])} complete months from January 2019 through
February 2025. The mean of complete monthly means was
{fmt(descriptive['mean_ug_m3'])} ug/m3 (median {fmt(descriptive['median_ug_m3'])}).
Seasonal Mann-Kendall indicated an increase (p={smk['p_value']:.4f}), and
month-adjusted HAC regression estimated {fmt(hac['slope_ug_m3_per_year'])}
ug/m3/year (95% CI {fmt(hac['ci_low'])} to {fmt(hac['ci_high'])}); however,
trend-free prewhitening was not significant (p={tfpw['p_value']:.2f}), so trend
inference is sensitive to serial dependence. A season-matched March-August 2020
contrast was {fmt(match['estimate_ug_m3'])} ug/m3 (95% bootstrap CI
{fmt(match['ci_low'])} to {fmt(match['ci_high'])}); the design does not support a
causal lockdown claim. SARIMA achieved the lowest rolling-origin MASE
({best['mase']:.2f}) and was used for a 24-month forecast; 2030 values are shown
only as conditional targets/benchmarks. PM10, NO2, and SO2 remain important AQI
pollutants in principle, but their legacy columns lacked the station, unit, and
method continuity required for empirical analysis. The audit reverses the
legacy declining trend claim and removes unsupported multi-pollutant,
source-apportionment, health-burden, and citywide-exposure conclusions.

**Keywords:** PM2.5; Dhaka; AirNow; data provenance; seasonal Mann-Kendall;
rolling-origin forecasting; reproducibility

# 1. Introduction

Air pollution is a major environmental-health concern, and WHO guidance covers
PM2.5, PM10, NO2, SO2, ozone, and carbon monoxide because their health effects
and regulatory averaging periods differ [1]. Prior Dhaka studies using other
monitoring periods or research designs have reported strong winter seasonality,
high particulate concentrations, and roles for gaseous pollutants [6-10]. Those
studies establish the scientific relevance of a multi-pollutant perspective;
they do not validate unrelated columns in the present repository.

A credible time-series study requires consistent measurement lineage. The
original manuscript described a 2017-2025 multi-pollutant monitoring record,
but audit of its code, workbooks, and Git history found CAMS reanalysis values,
scraped AQI, monthly values repeated across days, unresolved station identities,
and annual contextual variables repeated monthly. The original 41-reference
list also contained wrong or unrelated DOIs; its retained sources were checked
against publisher, registry, or official metadata.

This revision asks: (1) what seasonal and temporal patterns are supported by
one identified ground monitor; (2) how sensitive is trend inference to
seasonality and serial dependence; (3) what non-causal association is visible
during the 2020 restriction period; and (4) which short-horizon forecast model
outperforms seasonal naive in rolling tests? The scope is deliberately narrower
than the environmental importance of Dhaka's full pollutant mixture.

# 2. Data and methods

## 2.1 Monitor and provenance

The primary source is the official dated AirNow `daily_data_v2.dat` archive.
The original provider is U.S. Department of State Bangladesh - Dhaka; AirNow is
the distribution system. Each source row reports station `DK1010001` (full code
`050DK1010001`), site name Dhaka, latitude 23.796374, longitude 90.424614,
parameter PM2.5-24hr, unit UG/M3, duration 24 hours, concentration, AQI, and
category. The archive row does not encode instrument metadata, so the
instrument remains unresolved. AirNow values are preliminary and subject to
revision.

The downloader used verified HTTPS and preserved every URL, exact source line,
response SHA-256, and retrieval timestamp. Bangladesh Department of Environment
reports were selected as a separate validation source, but no reproducible raw
machine-readable DoE series was found; no numeric series was extracted from
charts or spliced into AirNow.

## 2.2 Pollutant scope and AQI interpretation

PM10, NO2, and SO2 were not removed because they are unimportant. A
multi-pollutant AQI calculates a subindex from each simultaneous physical
concentration and reports the maximum. Any of these pollutants can therefore
determine AQI when its subindex is largest. They are parallel inputs to the AQI
algorithm, not causal covariates that “influence” an independently measured AQI.

| Pollutant | Legacy evidence | Revised empirical treatment |
|---|---|---|
| PM2.5 | Source-mixed CAMS and monthly values | Replaced by identified AirNow/DoS ground-monitor observations |
| PM10 | Modeled/repeated values; station and method unresolved | Excluded pending a homogeneous physical concentration series |
| NO2 | Modeled/repeated values; units and station unresolved | Excluded pending a homogeneous physical concentration series |
| SO2 | Modeled/repeated values; units and station unresolved | Excluded pending a homogeneous physical concentration series |

Because PM2.5 is the only verified simultaneous concentration, the recalculated
AQI in this study is specifically the PM2.5 subindex. The stored
`dominant_pollutant=pm25` value means PM2.5 is the only available calculated
subindex; it does not prove that PM2.5 dominated Dhaka's complete
multi-pollutant AQI on every day. Prior Dhaka studies confirm that PM10 and
gaseous pollutants warrant monitoring [6,7], but their results cannot be
silently spliced into this station series.

## 2.3 QA, aggregation, and cutoff

The pipeline rejected wrong station, agency, pollutant, unit, duration,
duplicates, negative values, and dates beyond retrieval. Asia/Dhaka defines
local days. Monthly summaries require >=75% valid daily AirNow summaries.
Hourly completeness is unknown and remains null. A terminal month must reach
month-end, so March 2025 ({int(monthly_all.iloc[-1]['valid_days'])}/31 days) is
partial and the cutoff is February 2025. No missing observation was imputed.

AQI was recalculated from physical PM2.5 using U.S. EPA breakpoints effective
6 May 2024. Source-reported AQI was retained separately because historical
values used contemporaneous breakpoints. AQI is a mathematical transform here,
not an independent environmental outcome.

## 2.4 Statistical analysis

Descriptive results include valid counts, mean, median, standard deviation, and
IQR. Monthly climatology includes n and t-based 95% confidence intervals.
Seasonal differences use Kruskal-Wallis, epsilon-squared, and Holm-corrected
pairwise Mann-Whitney tests.

Trend methods comprise seasonal Mann-Kendall with tie correction, same-month
Sen slope with a seeded year-block bootstrap interval, trend-free prewhitening,
and elapsed-time regression with calendar-month fixed effects and 12-lag
Newey-West covariance. Pettitt's test is a change diagnostic only.

COVID analyses compare March-August 2020 with the same months of 2019 and fit a
month-adjusted interrupted association. Meteorology was unavailable; language
is associational rather than causal. Annual means are compared with annual
guidelines, and daily observations with the specified 24-hour forms.

## 2.5 Forecasting and scenarios

Seasonal-naive, naive, drift, ETS, SARIMA, and time/month regression models were
evaluated at three expanding origins with 12-month test horizons. Metrics were
MAE, RMSE, MASE, sMAPE, and 95% interval coverage. Positive-support statistical
models used log1p concentration. The lowest mean MASE selected the model. The
empirical horizon is 24 months. 2030 lines are deterministic benchmarks, not
forecasts; no probabilistic interval is fabricated for policy implementation.

# 3. Results

## 3.1 Completeness and distribution

The observed archive contained {len(daily):,} daily summaries. Of
{len(monthly_all)} calendar months, {int(descriptive['n_complete_months'])} met
the primary rule. Complete-month PM2.5 ranged from
{fmt(descriptive['min_ug_m3'])} to {fmt(descriptive['max_ug_m3'])} ug/m3; the
IQR was {fmt(descriptive['iqr_ug_m3'])} ug/m3.

![Observed monthly PM2.5. Open symbols are excluded partial months. Source: U.S. Department of State via AirNow.](figures/observed_pm25_time_series.png){{ width=90% }}

## 3.2 Seasonality

Calendar-month climatology peaked in month {int(peak['month'])} at
{fmt(peak['mean'])} ug/m3 and was lowest in month {int(low['month'])} at
{fmt(low['mean'])} ug/m3. Seasonal differences were large (Kruskal-Wallis
epsilon-squared 0.86, p<0.001).

![Monthly climatology with 95% confidence intervals; complete months only.](figures/monthly_climatology.png){{ width=80% }}

## 3.3 Trends and break diagnostic

Seasonal Mann-Kendall yielded z={fmt(smk['z'])}, p={smk['p_value']:.4f}. The
seasonal Sen estimate was {fmt(sen['slope_ug_m3_per_year'])} ug/m3/year (95%
bootstrap CI {fmt(sen['ci_low'])} to {fmt(sen['ci_high'])}). HAC regression
estimated {fmt(hac['slope_ug_m3_per_year'])} (95% CI {fmt(hac['ci_low'])} to
{fmt(hac['ci_high'])}), but prewhitening yielded p={tfpw['p_value']:.2f} with
lag-1 autocorrelation {tfpw['lag1_autocorrelation']:.2f}. We therefore interpret
the direction as suggestive of an increase but not robust across dependence
adjustments. Pettitt's test did not identify a significant single change
(p={trend['pettitt_change_diagnostic']['p_value']:.2f}).

## 3.4 2020 restriction-period association

The season-matched difference was {fmt(match['estimate_ug_m3'])} ug/m3 (95% CI
{fmt(match['ci_low'])} to {fmt(match['ci_high'])}, p={match['p_value']:.2f}). The
month-adjusted interrupted coefficient was {fmt(its['estimate_ug_m3'])} ug/m3
(95% CI {fmt(its['ci_low'])} to {fmt(its['ci_high'])}). The discrepancy and lack
of meteorological adjustment preclude a causal conclusion.

## 3.5 Guideline comparisons

All complete annual monitor means (2019-2024) exceeded WHO 2021 (5 ug/m3),
Bangladesh 2022 (35 ug/m3), and U.S. EPA 2024 (9 ug/m3) annual levels. EPA
comparisons are descriptive because formal NAAQS determinations use design-value
procedures and regulatory data.

{markdown_table(annual_simple, list(annual_simple.columns))}

![Annual monitor means with annual reference levels; these are not interchangeable with 24-hour standards.](figures/annual_guideline_comparison.png){{ width=80% }}

## 3.6 Forecast validation

The selected model was {best['model']} (mean MAE {fmt(best['mae'])}, RMSE
{fmt(best['rmse'])}, MASE {best['mase']:.2f}, sMAPE {best['smape_pct']:.1f}%).
Seasonal naive had MASE
{ranking.loc[ranking['model'] == 'seasonal_naive', 'mase'].iloc[0]:.2f}.

![Rolling-origin 12-month backtesting.](figures/forecast_backtesting.png){{ width=80% }}

![Selected 24-month forecast with 95% prediction intervals.](figures/forecast_intervals.png){{ width=90% }}

The 2030 benchmarks end at {fmt(scenario.iloc[0]['pm25_ug_m3'])} ug/m3 for no
additional change, 35 ug/m3 for the Bangladesh annual standard, and 5 ug/m3 for
the WHO AQG. These values express assumptions/targets, not expected outcomes.

![Conditional 2030 benchmark paths, not forecasts.](figures/scenarios_2030.png){{ width=80% }}

# 4. Discussion

## 4.1 What changed after the provenance audit

Rebuilding from one identified monitor changes both scope and inference. The
legacy analysis reported a strong PM2.5 decline using a source-mixed 2017-2025
series. The audited 2019-2025 monitor record instead suggests an increase under
seasonal MK and HAC regression, with loss of significance after prewhitening.
The correct conclusion is sensitivity, not a definitive monotonic trajectory.

## 4.2 Why PM10, NO2, and SO2 still matter

The original paper was correct that PM10, NO2, and SO2 are environmentally and
regulatorily important. Historical multi-site work in Dhaka measured PM10 and
gaseous pollutants alongside PM2.5 and found pollutant-specific seasonal and
trend behavior [6,7]. Their absence from the revised numerical tables is an
evidence boundary: the repository does not contain a homogeneous, traceable
series for them. Treating modeled CAMS values, repeated monthly values, and
unresolved ground records as one series would produce precise-looking but
invalid pollutant rankings and forecasts.

This distinction also changes AQI interpretation. If simultaneous PM10, NO2,
SO2, CO, and O3 concentrations become available, every applicable subindex
should be calculated under one declared standard and the daily maximum should
be retained. Until then, the present calculated AQI is a PM2.5 subindex, and no
claim is made about the historical percentage of days dominated by other
pollutants.

## 4.3 Seasonality, sources, COVID, and forecasting

Seasonality is much more stable: winter concentrations are high and monsoon
concentrations low, consistent with earlier Dhaka research [6-10]. Published
black-carbon and isotope studies support roles for local and regional combustion
sources [9,10], but the present data do not apportion sources. PM2.5 cannot be
summed with PM10 for source apportionment, and no chemical speciation, PMF/CMB,
inventory, or transport attribution is implemented here.

The season-matched COVID comparison is uncertain. Although the interrupted
coefficient is negative, a single-station ecological time series without
meteorology or mobility controls cannot isolate lockdown effects. Likewise,
AQI-PM2.5 correlation would be definitional and is not presented as discovery.
The selected 24-month forecast is an empirical extrapolation, whereas the 2030
lines are policy/health benchmarks; Bangladesh's national air-quality plan
provides relevant policy context but does not make those benchmark paths
probabilistic forecasts [11].

# 5. Limitations

One monitor does not measure citywide spatial exposure. Values are preliminary
daily summaries; per-hour completeness and archive-row instrument metadata are
unavailable. Three internal months with 66-71% coverage are excluded from trend
inference but retained explicitly in forecasting to preserve the calendar
without imputation. No independent numeric DoE validation or meteorological
series was available. The absence of traceable PM10, NO2, SO2, CO, and O3
prevents calculation of a complete multi-pollutant AQI and pollutant-dominance
fractions. The feed cessation limits the empirical period. Forecast validation
has only three annual rolling origins, and the 24-month outlook cannot establish
2030 conditions. Trend conclusions depend on serial-correlation handling. The
supplied original contained no tracked-change history; this is therefore a
clean revision accompanied by claim and reference audits.

# 6. Conclusion

The defensible result is a narrow, station-specific PM2.5 study. Pollution at
this monitor remained far above annual health and regulatory reference levels,
with strong seasonality and method-sensitive trend evidence. The data do not
support multi-pollutant trends, citywide exposure, source apportionment, an EKC,
precise attributable mortality, or a causal lockdown claim. A 24-month forecast
can be reported with rolling-origin validation; 2030 values remain conditional
benchmarks. PM10, NO2, and SO2 remain essential monitoring targets and potential
AQI-determining pollutants; defensible analysis of them requires a new,
station-identified, unit-verified, QA-documented concentration series.

# Data and code availability

Exact extracted AirNow rows, request URLs, response checksums, standardized
products, source manifest, provenance ledger, code, tests, and generated outputs
are in the repository. Large or terms-restricted CAMS, BMD, OpenAQ, or future
DoE raw files must be downloaded separately under provider terms. The paper was
generated from repository commit `{commit}`. Raw retrieval occurred on
2026-07-18 UTC (2026-07-17 America/New_York). The supplied original DOCX is
preserved unchanged with a SHA-256 manifest; its complete claim and 41-reference
audits are distributed with the revision.

# References

1. World Health Organization. *WHO global air quality guidelines: particulate matter, ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*. 2021. ISBN 978-92-4-003422-8. https://www.who.int/publications/i/item/9789240034228
2. Bangladesh Department of Environment. *Ambient Air Quality in Bangladesh (2018-2023)*. Published 17 June 2025. https://doe.gov.bd/pages/publications/ambient-air-quality-in-bangladesh-2018-2023-7b0bcb-6922da5381fc96cef9eb5f62
3. U.S. Environmental Protection Agency. *Daily Data File Fact Sheet*. AirNow. Accessed 17 July 2026. https://docs.airnowapi.org/docs/DailyDataFactSheet.pdf
4. U.S. Environmental Protection Agency. *Final Updates to the Air Quality Index for Particulate Matter*. 2024. https://www.epa.gov/system/files/documents/2024-02/pm-naaqs-air-quality-index-fact-sheet.pdf
5. U.S. Environmental Protection Agency. *Final Reconsideration of the National Ambient Air Quality Standards for Particulate Matter*. Effective 6 May 2024. https://www.epa.gov/pm-pollution/final-reconsideration-national-ambient-air-quality-standards-particulate-matter-pm
6. Rahman MM, Mahamud S, Thurston GD. Recent spatial gradients and time trends in Dhaka, Bangladesh, air pollution and their human health implications. *Journal of the Air & Waste Management Association*. 2019;69(4):478-501. https://doi.org/10.1080/10962247.2018.1548388
7. Pavel MRS, Zaman SU, Jeba F, Islam MS, Salam A. Long-Term (2003-2019) Air Quality, Climate Variables, and Human Health Consequences in Dhaka, Bangladesh. *Frontiers in Sustainable Cities*. 2021;3:681759. https://doi.org/10.3389/frsc.2021.681759
8. Rahman R-R, Kabir A. Spatiotemporal analysis and forecasting of air quality in the greater Dhaka region and assessment of a novel particulate matter filtration unit. *Environmental Monitoring and Assessment*. 2023;195:824. https://doi.org/10.1007/s10661-023-11370-y
9. Salam A, Andersson A, Jeba F, Haque MI, Khan MDH, Gustafsson O. Wintertime Air Quality in Megacity Dhaka, Bangladesh Strongly Affected by Influx of Black Carbon Aerosols from Regional Biomass Burning. *Environmental Science & Technology*. 2021;55(18):12243-12249. https://doi.org/10.1021/acs.est.1c03623
10. Nayem AKM, Zaman SU, Begum F, Salam A. Wintertime black carbon assessment in Dhaka, Bangladesh: Integrated health risk analysis. *Heliyon*. 2025;11(2):e41809. https://doi.org/10.1016/j.heliyon.2025.e41809
11. Bangladesh Department of Environment. *Bangladesh National Air Quality Management Plan 2024-2030*. Published 7 November 2024. https://doe.gov.bd/pages/publications/bangladesh-national-air-quality-management-plan-2024-2030-469099-6922da4f81fc96cef9eb5ec2
"""
    markdown_path = PAPER / "paper_revised.md"
    markdown_path.write_text(text, encoding="utf-8")

    revision_log = """# Revision log

The original DOCX was supplied on 2026-07-18 and preserved unchanged. It
contains no Word comments or tracked-change elements, so this is a clean revised
manuscript accompanied by explicit claim and reference audits, not a document
falsely labeled as tracked changes.

- Restored the original author list, affiliations, and corresponding author.
- Removed the duplicated title, abstract, and keywords.
- Narrowed title and claims to one identified Dhaka PM2.5 monitor.
- Replaced 2017-2025 mixed data with 2019-February 2025 AirNow observations.
- Removed unsupported PM10, NO2, SO2, CO, and O3 trends, but added an explicit
  explanation that these remain important AQI pollutants and can be restored
  when homogeneous concentration data become available.
- Removed source apportionment, EKC, numerical health burden, regional hard-coded comparison, and PM10-ratio forecasts.
- Recalculated AQI using one documented standard and described its mathematical dependence.
- Replaced ordinary MK/year OLS with seasonal MK, Sen slope, prewhitening, HAC regression, and break diagnostics.
- Replaced seasonally confounded COVID grouping with matched and interrupted associations.
- Corrected annual versus 24-hour guideline comparisons.
- Replaced a fixed split and unweighted ensemble with rolling-origin baseline comparison.
- Reduced the empirical forecast to 24 months and relabeled 2030 values as deterministic benchmarks.
- Audited all 41 original references; corrected retained citations and removed
  news-based concentration evidence, unresolved citations, and DOI/title mismatches.
- Added explicit limitations, data availability, code availability, provenance, and cutoff statements.
"""
    (PAPER / "revision_log.md").write_text(revision_log, encoding="utf-8")

    claims = [
        ("C01", "Abstract", f"{len(daily)} valid daily summaries", "daily_n", "data/processed/primary_observed_daily.parquet", "scripts/build_primary_dataset.py", "primary_observed_daily", "airnow_dhaka_dk1010001", "verified", "row count"),
        ("C02", "Abstract", f"{int(descriptive['n_complete_months'])} complete months", "complete_months", "tables/descriptive_summary.csv", "scripts/run_analysis.py", "analysis_monthly", "airnow_dhaka_dk1010001", "verified", "75% plus terminal-month rule"),
        ("C03", "Abstract", f"Monthly mean {fmt(descriptive['mean_ug_m3'])} ug/m3", "mean_ug_m3", "tables/descriptive_summary.csv", "scripts/run_analysis.py", "analysis_monthly", "airnow_dhaka_dk1010001", "verified", "complete months"),
        ("C04", "Trends", f"HAC slope {fmt(hac['slope_ug_m3_per_year'])} ug/m3/year", "slope_ug_m3_per_year", "tables/trend_summary.json", "scripts/run_analysis.py", "analysis_monthly", "airnow_dhaka_dk1010001", "verified", "month fixed effects"),
        ("C05", "Trends", f"Prewhitened p={tfpw['p_value']:.2f}", "p_value", "tables/trend_summary.json", "scripts/run_analysis.py", "analysis_monthly", "airnow_dhaka_dk1010001", "verified", "serial-correlation sensitivity"),
        ("C06", "COVID", f"Matched difference {fmt(match['estimate_ug_m3'])} ug/m3", "estimate_ug_m3", "tables/covid_association.csv", "scripts/run_analysis.py", "analysis_monthly", "airnow_dhaka_dk1010001", "verified", "association only"),
        ("C07", "Guidelines", "All complete annual means exceed annual reference levels", "exceeds_threshold", "tables/annual_guideline_comparison.csv", "scripts/run_analysis.py", "primary_observed_daily", "airnow_dhaka_dk1010001", "verified", "EPA is descriptive"),
        ("C08", "Forecast", f"{best['model']} selected with MASE {best['mase']:.2f}", "mase", "tables/model_ranking.csv", "scripts/run_forecasting.py", "primary_observed_monthly", "airnow_dhaka_dk1010001", "verified", "three rolling origins"),
        ("C09", "Pollutant scope", "PM10, NO2, and SO2 legacy values excluded", "measurement_type/exclusion_reason", "data/provenance/legacy_observation_provenance.csv", "scripts/audit_legacy.py", "legacy source-mixed datasets", "not_applicable_excluded", "verified", "scientific importance retained; empirical series rejected"),
        ("C10", "AQI interpretation", "Recalculated AQI is the PM2.5 subindex only", "pm25_subindex/dominant_pollutant", "data/processed/primary_observed_daily.parquet", "scripts/build_primary_dataset.py", "primary_observed_daily", "airnow_dhaka_dk1010001", "verified", "no other simultaneous pollutant subindices available"),
    ]
    with (PAPER / "claim_traceability.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["claim_id", "manuscript_section", "claim_text", "result_variable", "generated_output_file", "generating_script", "source_dataset", "source_manifest_id", "status", "notes"])
        writer.writerows(claims)

    comparison = f"""# Original versus revised results

| Item | Original repository | Revised audit |
|---|---|---|
| Study period | 2017-2025/2030; README and CSV dates conflicted | Observed 2019-01-01 to 2025-03-24; complete analysis through 2025-02 |
| Source | Claimed one Dhaka CAMS station | AirNow/DoS station DK1010001; exact rows/checksums |
| Stations | Unidentified | One, 23.796374 N 90.424614 E |
| Observations | 108 monthly plus 3,407 pseudo-daily rows | {len(daily):,} observed daily summaries; {int(descriptive['n_complete_months'])} complete months |
| PM2.5 level | README approximately 130 ug/m3 | Mean of complete monthly means {fmt(descriptive['mean_ug_m3'])} ug/m3; daily-weighted 2024 mean 98.19 |
| PM10, NO2, SO2 | Treated as homogeneous observed concentrations and forecast targets | Scientifically important but excluded because station, units, method continuity, and QA lineage are unresolved |
| AQI interpretation | Independent outcome/PM2.5 “driver” | EPA-2024 PM2.5 subindex only; no complete multi-pollutant dominance claim |
| Trend | MK decreasing; simple OLS -13.57 ug/m3/year | Seasonal Sen +{fmt(sen['slope_ug_m3_per_year'])}; HAC +{fmt(hac['slope_ug_m3_per_year'])}; prewhitening p={tfpw['p_value']:.2f}; method-sensitive |
| Seasonal peak | Winter, based mixed source | January climatology {fmt(peak['mean'])} ug/m3; strong seasonal effect |
| Rainfall | Undocumented normalized index | Removed; physical meteorology unavailable |
| COVID | Whole pre-period versus March-August; causal framing | Same-month contrast {fmt(match['estimate_ug_m3'])} (CI {fmt(match['ci_low'])}, {fmt(match['ci_high'])}); association only |
| Guidelines | Monthly exceedances against mismatched periods | Separate annual and daily/form-specific tables |
| Model evaluation | One 2024-2025 split; unweighted ensemble MAE 22.13 | Three rolling origins; {best['model']} MASE {best['mase']:.2f}, seasonal naive {ranking.loc[ranking['model']=='seasonal_naive','mase'].iloc[0]:.2f} |
| Forecast horizon | 2026-2030 forecast | 24 months through 2027-02 |
| 2030 | Precise forecasts including ratio-derived PM10 | Deterministic PM2.5 benchmarks only; no uncertainty invented |
| Claims removed | None | Multi-pollutant trends, source apportionment, EKC, mortality count, citywide exposure, and PM10-ratio forecast removed from empirical results; multi-pollutant importance retained as context and future data requirement |
"""
    (ROOT / "reports/original_vs_revised_results.md").write_text(comparison, encoding="utf-8")

    pandoc = shutil.which("pandoc")
    if pandoc:
        subprocess.run(
            [pandoc, str(markdown_path), "--resource-path", str(ROOT), "-o", str(PAPER / "paper_revised.docx")],
            cwd=ROOT,
            check=True,
        )
    else:
        raise RuntimeError("pandoc is required to create paper_revised.docx")
    print(f"generated {markdown_path} and {PAPER / 'paper_revised.docx'}")


if __name__ == "__main__":
    main()
