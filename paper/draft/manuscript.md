# Severe and Persistent Winter Air Pollution in Dhaka: An Official-Source Analysis of Burden, Episodes, and Trend Robustness, 2022–2025

## Abstract

**Background.** Dhaka experiences recurrent poor air quality, but public analyses are often limited by unclear provenance, inconsistent monitoring support, or long-range extrapolation from short records. We constructed an auditable dataset directly from Bangladesh Department of Environment (DoE) daily air-quality index (AQI) reports and monthly monitoring summaries to characterize recent burden, seasonality, episode persistence, and trend robustness.

**Methods.** The primary monthly analysis covered January 2022–December 2025 and included AQI, PM~2.5~, and PM~10~. The daily analysis covered January 2024–December 2025. Seasonal distributions were summarized with nonparametric tests and bootstrap confidence intervals. High-pollution episodes were defined as consecutive reported calendar days with AQI >150. Trends were estimated using month-adjusted ordinary least squares with heteroskedasticity- and autocorrelation-consistent standard errors, a 2023–2025 sensitivity window, Theil–Sen slopes on monthly anomalies, capture-weighted station averages, and a near-complete fixed-station series. National population and tree-cover-loss data were retained only as descriptive context because their geography and temporal resolution do not match Dhaka monitoring observations.

**Results.** Mean monthly AQI was 167.0 during 2022–2025. Winter mean AQI was 245.3 (bootstrap 95% CI: 219.8–274.6), compared with 109.8 (102.5–117.0) during monsoon. Winter PM~2.5~ averaged 167.7 µg/m³ (148.6–188.9), compared with 43.5 µg/m³ (39.0–47.8) during monsoon. AQI exceeded 150 on 48.1% of reported days in 2024 and 57.4% in 2025. A 141-day episode extended from 12 November 2024 through 1 April 2025 and peaked at AQI 367. Month-adjusted 2022–2025 models indicated annual changes of −12.4 AQI units (95% CI: −21.8 to −3.1), −8.7 µg/m³ PM~2.5~ (−12.8 to −4.5), and −7.0 µg/m³ PM~10~ (−15.3 to 1.3). However, the AQI slope became −0.6 (−5.3 to 4.1) when 2022 was excluded, demonstrating substantial baseline sensitivity.

**Conclusions.** Dhaka’s recent official record shows a severe, particulate-dominated winter burden characterized by long high-AQI episodes. Some trend specifications suggest improvement, especially for PM~2.5~, but the short comparable record and changing monitoring support preclude a simple claim of sustained citywide decline. Policy evaluation should prioritize stable station definitions, explicit units, continuous daily concentration reporting, and episode-oriented indicators in addition to annual averages.

**Keywords:** Dhaka; air quality index; PM~2.5~; PM~10~; seasonality; pollution episodes; official monitoring; trend sensitivity

## 1. Introduction

Air pollution is a major environmental health risk, and particulate matter is especially important because it penetrates deep into the respiratory tract and is associated with cardiovascular and respiratory disease. The World Health Organization’s 2021 air-quality guidelines emphasize that health effects occur across a wide concentration range and recommend long- and short-term targets for PM~2.5~ and PM~10~ (World Health Organization, 2021). Dhaka’s rapid urban growth, traffic, construction activity, industrial sources, brick production, regional transport, and dry-season meteorology create a complex exposure environment. The Bangladesh National Air Quality Management Plan 2024–2030 accordingly treats air pollution as a multisectoral policy problem requiring better monitoring and source-specific intervention (Department of Environment, 2024).

Previous research has documented strong dry-season pollution in Dhaka and other Bangladeshi cities. Long-term and multisite studies have linked particulate variability to meteorology, urban activity, and regional patterns, while forecasting studies have shown that strong seasonality can make short-horizon prediction feasible. Yet the evidentiary quality of any new analysis depends on the public record actually available. A smooth model cannot repair missing reporting years, changing station composition, unclear units, or the absence of comparable daily concentrations.

This study therefore begins with provenance. It uses official DoE archive pages and linked reports, preserves the source URL and document hash for each extracted record, distinguishes AQI from pollutant concentration, and reports incomplete or partial extraction explicitly. It then asks three focused questions:

1. How large and seasonally concentrated were Dhaka’s official AQI, PM~2.5~, and PM~10~ burdens during the recent comparable monitoring period?
2. How frequently did high-AQI conditions occur, and how persistent were consecutive high-pollution episodes?
3. Are estimated temporal trends robust to the choice of baseline period, estimator, station weighting, and fixed-station restriction?

The aim is not source apportionment or causal inference. It is a reproducible assessment of the burden and persistence visible in the official monitoring record, with uncertainty and sensitivity analyses proportionate to the data.

## 2. Methods

### 2.1 Study design and data sources

We conducted an observational time-series analysis of public DoE reports for Dhaka. The extraction pipeline discovered attachments from the DoE daily AQI archive and monthly air-quality report pages, downloaded each report temporarily, extracted structured observations, wrote processed tables, and removed the source files after successful validation. Source URLs, SHA-256 hashes, retrieval metadata, extraction status, and QA flags remain in the processed dataset, allowing every observation to be traced without retaining more than 600 MB of duplicate report files.

Daily DoE reports provide a published city AQI, a controlling or responsible pollutant when reported, an AQI category, and comments. AQI is a composite index and is not interpreted as a pollutant concentration. Monthly reports contain station-level summary statistics for PM~2.5~, PM~10~, NO~2~, SO~2~, CO, and O~3~, including station averages, extrema, data capture, and exceedance fields when available. The city monthly PM summaries used here are unweighted means of reporting-station monthly averages unless a sensitivity specification states otherwise.

The complete public monthly archive was discontinuous: linked reports were available for 2013–2019 and from 2022 onward, with no linked 2020–2021 monthly series. Numeric monthly AQI was available from 2022. The primary monthly analysis was therefore prespecified as January 2022–December 2025, providing 48 complete AQI months and 47 PM months. Partial 2026 observations remain in the dataset but were excluded from all estimates. The primary daily analysis used selected unambiguous records from 1 January 2024 through 31 December 2025: 366 reported days in 2024 and 357 of 365 days in 2025.

### 2.2 Outcomes and seasons

The core outcomes were monthly mean AQI, monthly mean PM~2.5~, monthly mean PM~10~, and daily AQI. Gaseous pollutants were retained in the source workbook but excluded from the main analysis because recent source tables did not always state units clearly and because a seven-pollutant narrative obscured the central particulate burden.

Seasons were defined as winter (December–February), pre-monsoon (March–May), monsoon (June–September), and post-monsoon (October–November). This four-season classification reflects Bangladesh’s climatic cycle and was applied consistently to monthly and daily observations.

### 2.3 Burden and episode metrics

For each outcome, we calculated the mean, standard deviation, median, interquartile range, minimum, and maximum. Seasonal means were accompanied by percentile bootstrap 95% confidence intervals based on 5,000 resamples with a fixed random seed. Seasonal distributions were compared with Kruskal–Wallis tests because the sample was small and strongly seasonal. Benjamini–Hochberg q-values controlled the false-discovery rate across the three core outcomes.

Daily burden was summarized as the number and proportion of reported days above AQI 100, 150, and 200. Monthly proportions above AQI 150 were accompanied by Wilson 95% confidence intervals. These proportions use reported days as the denominator; they do not impute missing daily values.

A high-pollution episode was defined as consecutive reported calendar days with AQI >150. A missing calendar date ended an episode rather than being assumed polluted or unpolluted. Episode summaries included start and end dates, duration, mean and peak AQI, and cumulative AQI above 150. The episode beginning on 1 January 2024 is left-censored by the daily analysis window and is not interpreted as having truly started on that date.

### 2.4 Trend and monitoring-sensitivity analysis

The principal trend model regressed each monthly outcome on continuous time and calendar-month indicators. Heteroskedasticity- and autocorrelation-consistent covariance estimates with three lags accounted for short-range residual dependence. Slopes are reported as change per year with 95% confidence intervals.

Four sensitivity checks addressed the fragility of inference from a short, changing network:

1. The month-adjusted model was repeated for 2023–2025 to determine whether the 2022 baseline drove the result.
2. Theil–Sen slopes were fitted to calendar-month anomalies, providing a robust estimator less influenced by individual extreme months.
3. PM station averages were weighted by their reported monthly data-capture percentage.
4. PM models were restricted to BARC, the station with 47 PM~2.5~ months and 45–47 PM~10~ months during the analysis window, to reduce bias from changing station composition.

Trend estimates were interpreted as descriptive temporal change, not effects of a policy or intervention. No COVID-19 effect was estimated because the public monthly archive does not contain 2020–2021 observations.

### 2.5 Population and tree-cover-loss context

The source workbook includes an annual Bangladesh population series from United Nations World Urbanization Prospects 2025. A sparse Worldometer table for 2010, 2015, 2020, and 2022–2026 was added as a transparent cross-check; Worldometer identifies the underlying source as the United Nations World Population Prospects 2024 Revision. It is not an independent demographic series.

Annual Bangladesh tree-cover loss for 2001–2024 was obtained from the Global Forest Watch series distributed by Our World in Data. The metric represents stand-replacement disturbance detectable in 30 m pixels from all causes. It must not be relabeled automatically as permanent deforestation.

Population and tree-cover loss are national, while the air-quality outcomes describe Dhaka and change monthly or daily. Both context series also trend over time, creating a high risk of ecological and temporal confounding. We therefore did not correlate them with air pollution or include them as regression predictors. Their role is limited to documenting the broader national context and identifying future hypotheses that require spatially matched data.

### 2.6 Reproducibility

The analysis was implemented in Python using pandas, SciPy, statsmodels, Matplotlib, and openpyxl. The repository contains the extraction code, automated daily updater, analysis code, tests, processed CSV files, workbooks, and figure-generation workflow. The updater compares the current DoE archive inventory with the retained manifest and reruns the dataset build, tests, analysis tables, and figures only when source reports change.

## 3. Results

### 3.1 Coverage and monitoring support

The processed archive contained 1,183 daily attachments and 129 monthly attachments. All daily reports and 121 monthly reports extracted with an “ok” status; eight monthly reports were partial because at least one pollutant block was not parsed. No source report had a failed extraction status. The monthly AQI series was complete for all 48 primary months. PM~2.5~ and PM~10~ each had 47 observations, with one unavailable month.

Figure 1 shows the sharp distinction between historical pollutant availability and the shorter numeric AQI series, as well as variation in reporting-station support and mean data capture. This variation motivated the capture-weighted and fixed-station trend checks.

![Figure 1. Coverage and monitoring support.](../../analysis/figures/figure_1_coverage_and_monitoring.png)

**Figure 1.** Official monthly data availability and monitoring support. Blue cells in panel A indicate available observations. Panel B shows the number of reporting PM stations and mean PM~2.5~ data capture during the comparable 2022–2025 window. Partial 2026 data are displayed for provenance but excluded from estimates.

### 3.2 Overall and seasonal burden

Across 2022–2025, mean monthly AQI was 167.0 (SD 62.2; median 149.0), mean PM~2.5~ was 92.4 µg/m³ (SD 54.3; median 72.7), and mean PM~10~ was 146.0 µg/m³ (SD 74.3; median 124.6).

Seasonal differences were large for every core outcome. Winter mean AQI was 245.3 (bootstrap 95% CI: 219.8–274.6), more than twice the monsoon mean of 109.8 (102.5–117.0). Winter PM~2.5~ averaged 167.7 µg/m³ (148.6–188.9), compared with 43.5 µg/m³ (39.0–47.8) during monsoon. Winter PM~10~ averaged 246.5 µg/m³ (223.5–268.1), compared with 74.7 µg/m³ (65.5–84.4) during monsoon. Kruskal–Wallis evidence remained strong after false-discovery-rate adjustment for AQI (*H*=38.71), PM~2.5~ (*H*=38.92), and PM~10~ (*H*=38.52); all three q-values were <0.001.

![Figure 2. Daily AQI burden.](../../analysis/figures/figure_2_daily_aqi_burden.png)

**Figure 2.** Daily AQI burden in 2024–2025. Panel A displays daily AQI, a 30-day rolling mean, and descriptive AQI category bands. Panel B shows the percentage of reported days in each month with AQI >150 and Wilson 95% confidence intervals.

![Figure 3. Seasonal particulate burden.](../../analysis/figures/figure_3_seasonal_particulate_burden.png)

**Figure 3.** Seasonal distributions of monthly AQI, PM~2.5~, and PM~10~ during 2022–2025. Points are individual months; boxes show the median and interquartile range.

### 3.3 Daily threshold burden and episodes

Daily coverage was 100% in 2024 and 97.8% in 2025. AQI exceeded 100 on 280 days in 2024 and 310 reported days in 2025. It exceeded 150 on 176 days (48.1%) in 2024 and 205 days (57.4%) in 2025, and exceeded 200 on 72 days in 2024 and 61 days in 2025.

Winter conditions were particularly persistent: 98.9% of reported winter days exceeded AQI 150 and 64.6% exceeded 200. In contrast, 7.5% of monsoon days exceeded 150 and none exceeded 200. Pre-monsoon and post-monsoon remained consequential transition periods, with 55.5% and 68.6% of reported days above 150, respectively.

We identified 42 AQI >150 episodes. The longest lasted 141 days, from 12 November 2024 through 1 April 2025, with a mean AQI of 210.9, a peak of 367, and cumulative excess of 8,584 AQI units above 150. The next longest fully observed episode lasted 37 days from 25 November through 31 December 2025. A 55-day episode beginning on 1 January 2024 was left-censored by the analysis window and may have begun earlier.

![Figure 4. High-pollution episodes.](../../analysis/figures/figure_4_pollution_episodes.png)

**Figure 4.** Persistence and seasonal concentration of high AQI. Panel A ranks the twelve longest episodes; color indicates peak AQI. Panel B shows the proportion of reported days above AQI 100, 150, and 200 by season.

### 3.4 Trend robustness

The principal 2022–2025 model estimated a decline of 12.4 AQI units per year (95% CI: −21.8 to −3.1; q=0.018), 8.7 µg/m³ PM~2.5~ per year (−12.8 to −4.5; q<0.001), and 7.0 µg/m³ PM~10~ per year (−15.3 to 1.3; q=0.111). The robust Theil–Sen estimates were directionally similar: −7.3 AQI units, −6.7 µg/m³ PM~2.5~, and −7.8 µg/m³ PM~10~ per year.

The baseline-period sensitivity materially changed interpretation. After removing 2022, the AQI estimate was −0.6 units per year (95% CI: −5.3 to 4.1), whereas PM~2.5~ remained negative at −5.4 µg/m³ per year (−10.8 to −0.1) and PM~10~ was −9.4 µg/m³ per year (−13.5 to −5.4). Capture-weighted PM estimates were similar to the city monthly means. The BARC fixed-station PM~2.5~ estimate was more negative at −15.0 µg/m³ per year (−21.9 to −8.1), while the fixed-station PM~10~ interval crossed zero.

These results support evidence of recent PM~2.5~ improvement across several specifications, but not a blanket claim that Dhaka AQI has followed a stable downward trajectory. The apparent four-year AQI decline depends strongly on the elevated 2022 baseline.

![Figure 5. Trend sensitivity.](../../analysis/figures/figure_5_trend_sensitivity.png)

**Figure 5.** Estimated annual changes and 95% confidence intervals across temporal, robust-estimator, capture-weighted, and fixed-station specifications. Intervals crossing zero indicate that the corresponding specification does not distinguish a trend from no trend at the 5% level.

### 3.5 National demographic and forest context

Bangladesh’s population and urban share increased over the study period in both the primary UN series and the Worldometer presentation of UN estimates. The Global Forest Watch-derived series recorded 47,470 hectares of tree-cover loss during 2022–2024. These observations are environmentally relevant but do not show that population growth or forest loss caused Dhaka’s month-to-month pollution. National aggregation, temporal trend, land-cover definitions, regional transport, and multiple unmeasured emission sources prevent such an inference.

## 4. Discussion

### 4.1 Principal findings

The official record reveals a coherent public-health pattern: high pollution is not limited to isolated winter peaks. It forms a sustained seasonal regime. Almost every reported winter day in 2024–2025 exceeded AQI 150, nearly two-thirds exceeded 200, and one episode persisted for 141 consecutive reported days. Monthly PM~2.5~ and PM~10~ distributions corroborate the AQI pattern, with winter means approximately four and three times their respective monsoon means.

This episode-oriented view adds information that annual means conceal. A community experiencing four or five months of near-continuous high AQI faces a different monitoring and policy problem from one experiencing the same annual mean through scattered peaks. Episode start, duration, peak, and cumulative excess can therefore complement annual concentrations as operational indicators for warnings, source-control timing, school and workplace guidance, and evaluation of dry-season interventions.

### 4.2 Interpreting apparent improvement

The analysis provides qualified evidence of improvement, most consistently for PM~2.5~. The full-period, Theil–Sen, capture-weighted, and fixed-station PM~2.5~ estimates were all negative. However, the AQI result was highly sensitive to 2022: a statistically distinguishable decline over 2022–2025 became nearly flat over 2023–2025. This distinction matters because a short series can convert an unusually high first year into a persuasive-looking trend.

AQI and concentrations also answer different questions. AQI depends on pollutant-specific breakpoints and the controlling pollutant, while monthly PM values summarize station concentrations. Their slopes need not match. The proper conclusion is therefore not “air quality is solved” or “nothing improved,” but that recent PM~2.5~ measurements show evidence of decline while high winter AQI remains common and the persistence of citywide improvement is not yet established.

### 4.3 Implications for monitoring and policy

The strongest immediate recommendation is measurement stability. Public monthly reports should use consistent station names, state units in every table, document station openings and instrument changes, and publish machine-readable daily pollutant concentrations alongside AQI. Capture-adjusted city summaries and fixed-site indicators should accompany changing-network averages. These improvements would make future intervention and source-control evaluations more credible.

The pronounced seasonal pattern also supports intervention timing. Measures targeting construction dust, high-emitting vehicles, brick production, open burning, and industrial emissions are likely to have greatest short-term public-health relevance before and during the dry season. Formal attribution still requires emissions, meteorology, back trajectories, and source-apportionment data; the present analysis identifies when the burden is greatest, not which source contributes how much.

Population growth and forest dynamics merit future investigation only with appropriately matched data. Useful extensions would include gridded population exposure, land-cover change within a defined airshed, fire detections, boundary-layer height, rainfall, wind, and regional transport. A national annual forest-loss total is too coarse to test a Dhaka mechanism.

### 4.4 Strengths and limitations

The study’s strengths are its official-source lineage, automatic discovery of new year pages, observation-level source URLs and hashes, explicit duplicate policy, separation of AQI from concentration, archive-aware missingness, reproducible figures and tables, and multiple trend sensitivity analyses. Temporary source-report deletion improves operational efficiency without sacrificing traceability.

Several limitations remain. First, the primary comparable monthly period contains only four years, limiting power and making slopes baseline-sensitive. Second, station composition and data capture changed over time. Capture weighting and a BARC restriction probe but do not eliminate this problem. Third, city monthly values are averages of station monthly summaries rather than population-weighted exposures. Fourth, daily AQI does not provide daily concentrations of every pollutant, and the controlling pollutant is not a complete source decomposition. Fifth, recent gas units were not always explicit, so gases were excluded from the main results. Sixth, missing 2020–2021 public monthly reports preclude a direct COVID-19 intervention analysis. Finally, population and tree-cover-loss series are national; causal or Dhaka-specific interpretations would be ecological overreach.

## 5. Conclusion

Dhaka’s official monitoring record shows severe, particulate-dominated, and unusually persistent winter pollution. The most consequential result is not a single annual mean but the continuity of exposure: a 141-day AQI >150 episode crossed the 2024–2025 winter, and 98.9% of reported winter days exceeded that threshold. PM~2.5~ trends suggest possible recent improvement, but AQI trend inference is sensitive to the 2022 baseline and should not be presented as an established long-term decline.

A stronger evidence system would combine continuous public daily concentrations, stable site definitions, capture-adjusted and fixed-site summaries, meteorology, emissions, and spatially matched population and land-cover information. Until then, the most defensible use of the official archive is transparent burden surveillance, episode tracking, and carefully qualified trend assessment.

## Data and code availability

All processed observations, provenance fields, analysis tables, figures, tests, and source code are contained in the project repository. The primary data workbook is `data/processed/dhaka_doe_air_quality.xlsx`; paper-ready result tables are in `analysis/dhaka_doe_analysis.xlsx`. Raw DoE reports are not retained after successful processing, but their public URLs, SHA-256 hashes, report types, extraction status, and retrieval metadata remain in the source manifest.

## References

1. Bangladesh Department of Environment. *Daily Air Quality Index archive*. https://doe.gov.bd/pages/static-pages/6922dfba933eb65569e23b0a
2. Bangladesh Department of Environment. *Monthly Air Quality Report archive*. https://doe.gov.bd/pages/static-pages/6922de32933eb65569e18f46
3. Bangladesh Department of Environment. (2024). *Bangladesh National Air Quality Management Plan 2024–2030*. https://doe.gov.bd/pages/publications/bangladesh-national-air-quality-management-plan-2024-2030-469099-6922da4f81fc96cef9eb5ec2
4. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
5. Global Forest Watch. (2025). *Tree cover loss by dominant driver*, distributed by Our World in Data. https://ourworldindata.org/grapher/tree-cover-loss
6. Hansen, M. C., et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342(6160), 850–853. https://doi.org/10.1126/science.1244693
7. Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583–621.
8. Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
9. Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall’s tau. *Journal of the American Statistical Association*, 63(324), 1379–1389.
10. United Nations Department of Economic and Social Affairs, Population Division. (2025). *World Urbanization Prospects 2025*. https://population.un.org/wup/
11. World Health Organization. (2021). *WHO global air quality guidelines*. https://www.who.int/publications/i/item/9789240034228
12. Worldometer. (2026). *Bangladesh population*. Retrieved 6 August 2026. https://www.worldometers.info/world-population/bangladesh-population/
