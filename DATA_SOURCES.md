# Data sources

## Bangladesh Department of Environment

- Daily AQI reports: <https://doe.gov.bd/pages/static-pages/6922dfba933eb65569e23b0a>
- Monthly air-quality reports: <https://doe.gov.bd/pages/static-pages/6922de32933eb65569e18f46>
- Air-quality monitoring landing page: <https://doe.gov.bd/pages/static-pages/6922e141933eb65569e2b272>

The extractor reads the rendered archive tables, follows only HTTPS links on the DoE object-storage host, validates PDF/DOCX signatures, and records SHA-256 hashes. Older monthly pages supply the year in the page title while rows contain only month names; `source_label` states when the year was inherited from the official year page.

The monthly master page links year pages for 2013–2019 and 2022 onward. It does not expose 2020 or 2021 year pages, so the pollutant record preserves that gap. Numeric daily AQI is extracted from Table 6 of reports from 2022 onward; older AQI category-percentage graphics are not converted into invented numeric values.

## Population and HDI

`data/context/bangladesh_population.csv` and `data/context/bangladesh_hdi.csv` both describe Bangladesh nationally, not Dhaka city.

- Population series used in the workbook: UN DESA Population Division, *World Urbanization Prospects 2025*, file `WUP2025-F14-National-Definitions_Pop_by_category.xlsx`. <https://population.un.org/wup/downloads>
- Worldometer comparison: its Bangladesh table displays selected years and identifies United Nations population publications as its underlying sources. It is useful as a presentation/check, but it does not provide the complete annual 2013–2025 national total/urban/rural set requested here. <https://www.worldometers.info/world-population/bangladesh-population/>
- Retained 2013–2024 HDI source: `AIDS_BD_2000_2024.xlsx`. <https://github.com/awnonbhowmik/AIDS_BD-Data-Analysis/blob/main/data/AIDS_BD_2000_2024.xlsx>
- HDI reference named inside that workbook: CountryEconomy, whose Bangladesh page labels the source as “UN.” <https://countryeconomy.com/hdi/bangladesh>
- Official comparison series: UNDP Human Development Report 2025 complete time series. <https://hdr.undp.org/data-center/documentation-and-downloads>

The UN population workbook was verified on 18 July 2026 with SHA-256 `f359eb5677a9a92f6ef8b098e50320064876c131fe7585c09b956c7cf6a7011f`. It reports annual rural, urban, and total values in thousands. The CSV converts them to persons and calculates rural as total minus urban; every 2013–2025 result exactly matches the independently reported rural series. Dhaka's Worldometer urban-area estimate is not used because it has no compatible rural-Dhaka complement.

The AIDS workbook was verified on 18 July 2026 with SHA-256 `7154d167ba5e78304f381a4eae325ab6fb637639efec255bb29901d44701fda2`. Its 2013–2024 HDI values exactly match the retained context column. Its cited CountryEconomy page ends at 2022, while the workbook repeats `0.670` for 2023 and 2024; those two values are therefore treated as apparent forward-fills, not observed annual HDI.

Exact UNDP file verified on 18 July 2026: `HDR25_Composite_indices_complete_time_series.csv`, downloaded from <https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv>, SHA-256 `61ed82e5b66c88dfca8ff9fac775c63981ecab6a254862af97acacc41c143117`.

Source revisions and definition choices matter. The workbook therefore labels the UN WUP 2025 national-definition series explicitly and does not mix it with a different Worldometer/UN revision.

The retained `hdi` series differs from UNDP's current observation-year series for 2017–2023 because UNDP revises historical estimates across releases. The paper explains that its 2025 context value comes from the 2025 Human Development Report; that report's value of 0.685 is an observation for 2023, not 2025. The separate `hdi_undp_same_year` column supplies the verified UNDP 2017–2023 comparison series, while the source and verification fields distinguish workbook values, apparent forward-fills, and official observations. No same-year UNDP observations are asserted for 2024 or 2025.

## Bangladesh Meteorological Department

BMD data are not included in this fresh build. No public, source-verifiable BMD series was needed to reproduce the requested DoE workbook. Meteorology can be added later as a separately sourced table without modifying the official DoE observations.
