# Stored outputs: main.ipynb

## Cell 3

```text
Saved → figures/fig01_study_area.png

```
```text
<Figure size 800x900 with 1 Axes>
```
Stored image: `cell_003_001.png`

## Cell 5

```text
Loaded 108 monthly observations
Date range: 2017-01-01 → 2025-12-01
Columns: ['month_start', 'year', 'month', 'pm25_mean', 'pm25_median', 'pm25_min', 'pm25_max', 'pm10_mean', 'pm10_median', 'pm10_min', 'pm10_max', 'no2_mean', 'no2_median', 'no2_min', 'no2_max', 'so2_mean', 'so2_median', 'so2_min', 'so2_max', 'aqi_mean', 'aqi_median', 'aqi_min', 'aqi_max', 'population_total', 'urban_population', 'urban_share_pct', 'hdi', 'poverty_rate_pct', 'norm_rain', 'season', 'pm_ratio']

```

## Cell 7

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Descriptive statistics and Shapiro–Wilk normality test (p<sub>SW</sub>) for all variables. H<sub>0</sub>: normality; p < 0.05 ⇒ non-normal.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Variable</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">n</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Mean</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Std</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Min</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">Q<sub>1</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Median</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">Q<sub>3</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Max</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Skew</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Kurt</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">p<sub>SW</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Dist.</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>2.5</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">108</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">110.62</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">55.46</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">11.80</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">57.51</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">122.97</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">142.39</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">213.41</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-0.124</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-0.874</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0002</td><td style="padding:3px 12px;vertical-align:middle;text-align:left">Non-normal</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>10</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">108</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">312.14</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">149.72</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">102.50</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">147.75</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">320.60</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">417.72</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">617.40</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.144</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-1.142</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:left">Non-normal</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">NO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">108</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">47.30</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">20.24</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">16.80</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">26.65</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">49.00</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">62.42</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">86.60</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.179</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-1.078</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0002</td><td style="padding:3px 12px;vertical-align:middle;text-align:left">Non-normal</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">108</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">43.28</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">20.08</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">14.00</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">26.98</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.90</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">56.38</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">88.00</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.490</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-0.658</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0004</td><td style="padding:3px 12px;vertical-align:middle;text-align:left">Non-normal</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">108</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">172.69</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">53.27</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">44.48</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">140.70</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">186.84</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">201.03</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">263.21</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-0.615</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-0.205</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:left">Non-normal</td></tr></tbody></table></div>

## Cell 9

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Columns with missing values.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Column</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Missing</b> n</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>%</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">norm_rain</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">48</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">44.4</td></tr></tbody></table><div style="font-family:Georgia,serif;font-size:11.5px;color:#555;margin:3px 0"><i>Note:</i> <code>norm_rain</code> is available only for 2017–2021.</div></div>

## Cell 11

```text
Saved → figures/fig02_distributions.png

```
```text
<Figure size 1920x1200 with 6 Axes>
```
Stored image: `cell_011_002.png`

## Cell 13

```text
Saved → figures/fig03_cv_analysis.png

```
```text
<Figure size 1680x540 with 2 Axes>
```
Stored image: `cell_013_003.png`
```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Coefficient of variation (%) by year.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:center"><b>Year</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">CV<sub>PM<sub>2.5</sub></sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">CV<sub>NO<sub>2</sub></sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">CV<sub>SO<sub>2</sub></sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">CV<sub>AQI</sub></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2017</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">20.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">10.5</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2018</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">20.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">11.2</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2019</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">19.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">11.7</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2020</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">19.2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">11.9</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2021</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">18.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">11.9</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2022</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">55.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">35.6</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2023</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">51.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">32.4</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2024</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">53.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">36.1</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2025</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">61.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">36.0</td></tr></tbody></table></div>

## Cell 15

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Monthly AQI category distribution, Dhaka City (2017–2025).</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Category</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Months</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>% of Months</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Good</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.9</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Moderate</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">14</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">13.0</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Unhealthy for Sensitive</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">14</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">13.0</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Unhealthy</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">48</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">44.4</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Very Unhealthy</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">28</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">25.9</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Hazardous</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.9</td></tr></tbody></table></div>
```text
Saved → figures/fig04_aqi_category_pie.png

```
```text
<Figure size 780x624 with 1 Axes>
```
Stored image: `cell_015_004.png`

## Cell 17

```text
Saved → figures/fig05_monthly_time_series.png

```
```text
<Figure size 1680x1920 with 5 Axes>
```
Stored image: `cell_017_005.png`

## Cell 19

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Mann–Kendall monotonic trend test (n = 108 months).</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Variable</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">S</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">z</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">p</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Trend</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>2.5</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-1673</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-4.439</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">↓ Decreasing</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>10</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-824</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-2.185</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0289</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">↓ Decreasing</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">NO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">+1152</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">3.056</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0022</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">↑ Increasing</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">+1540</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">4.086</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">↑ Increasing</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-1661</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-4.407</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">↓ Decreasing</td></tr></tbody></table><div style="font-family:Georgia,serif;font-size:11.5px;color:#555;margin:3px 0"><i>Significance:</i> p < 0.05 ⇒ statistically significant trend.</div></div>

## Cell 21

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> OLS regression: pollutant ~ year (2017–2025).</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Variable</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Slope</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>SE</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">t</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">p</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">R<sup>2</sup></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Sig.</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>2.5</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-13.574</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.6044</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-8.460</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.403</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">***</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>10</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-8.603</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">5.5434</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-1.550</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.1237</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.022</td><td style="padding:3px 12px;vertical-align:middle;text-align:right"></td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">NO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">2.295</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.7245</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">3.170</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0020</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.086</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">**</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">3.489</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.6710</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">5.200</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.203</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">***</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-12.598</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.5749</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-8.000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.376</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">***</td></tr></tbody></table><div style="font-family:Georgia,serif;font-size:11.5px;color:#555;margin:3px 0"><i>Significance codes:</i> <sup>***</sup>p<0.001, <sup>**</sup>p<0.01, <sup>*</sup>p<0.05.</div></div>

## Cell 23

```text
Saved → figures/fig06_annual_trends.png

```
```text
<Figure size 1680x540 with 2 Axes>
```
Stored image: `cell_023_006.png`
```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table 1.</b> Annual mean pollutant concentrations and AQI, Dhaka City (2017–2025).</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:center"><b>Year</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>PM<sub>2.5</sub></b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>PM<sub>10</sub></b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>NO<sub>2</sub></b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>SO<sub>2</sub></b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>AQI</b></th></tr><tr><th style="padding:3px 12px;vertical-align:middle;text-align:center"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2017</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">131.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">306.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">39.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">32.9</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">194.4</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2018</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">138.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">321.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">34.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">199.2</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2019</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">145.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">337.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">43.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">36.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">204.5</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2020</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">152.2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">354.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">35.9</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">29.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">209.8</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2021</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">158.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">369.2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">48.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">41.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">215.1</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2022</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">119.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">326.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">52.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">50.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">177.9</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2023</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">50.3</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">270.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">54.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">54.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">118.9</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2024</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">47.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">257.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">54.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">55.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">115.0</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2025</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">52.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">266.2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">55.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">55.9</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">119.4</td></tr></tbody></table></div>

## Cell 25

```text
Saved → figures/fig07_interannual_boxplots.png

```
```text
<Figure size 1680x1080 with 2 Axes>
```
Stored image: `cell_025_007.png`

## Cell 27

```text
Saved → figures/fig08_monthly_climatology.png

```
```text
<Figure size 1680x540 with 2 Axes>
```
Stored image: `cell_027_008.png`

## Cell 29

```text
Saved → figures/fig09_aqi_heatmap.png

```
```text
<Figure size 2160x840 with 2 Axes>
```
Stored image: `cell_029_009.png`

## Cell 31

```text
Saved → figures/fig10_seasonal_distributions.png

```
```text
<Figure size 2400x2160 with 10 Axes>
```
Stored image: `cell_031_010.png`

## Cell 33

```text
Saved → figures/fig11_violin_by_season.png

```
```text
<Figure size 2400x720 with 5 Axes>
```
Stored image: `cell_033_011.png`

## Cell 35

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Augmented Dickey–Fuller stationarity tests.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Variable</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>ADF stat</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">p</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Lags</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Result</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>2.5</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-1.1686</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.6870</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">13</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Non-stationary ✗</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>10</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-1.0496</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.7347</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">12</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Non-stationary ✗</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">NO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-0.8248</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.8117</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">12</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Non-stationary ✗</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-0.5488</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.8821</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">12</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Non-stationary ✗</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-0.9229</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.7804</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">13</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Non-stationary ✗</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>2.5</sub> (Δ<sup>1</sup>)</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-2.2337</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.1942</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">12</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Non-stationary ✗</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>10</sub> (Δ<sup>1</sup>)</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-2.3709</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.1501</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">11</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Non-stationary ✗</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">NO<sub>2</sub> (Δ<sup>1</sup>)</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-3.4974</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0081</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">11</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Stationary ✓</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SO<sub>2</sub> (Δ<sup>1</sup>)</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-3.1250</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0248</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">11</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Stationary ✓</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI (Δ1)</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-2.5063</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.1139</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">12</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Non-stationary ✗</td></tr></tbody></table><div style="font-family:Georgia,serif;font-size:11.5px;color:#555;margin:3px 0"><i>Interpretation:</i> If original series is non-stationary but Δ<sup>1</sup> is stationary ⇒ d=1 for SARIMA.</div></div>

## Cell 37

```text
Saved → figures/fig12_acf_pacf.png

```
```text
<Figure size 1680x1440 with 8 Axes>
```
Stored image: `cell_037_012.png`

## Cell 39

```text
Saved → figures/fig13_stl_decomposition.png

```
```text
<Figure size 1680x1440 with 4 Axes>
```
Stored image: `cell_039_013.png`

## Cell 41

```text
Saved → figures/fig14_anomaly_detection.png

```
```text
<Figure size 1680x480 with 1 Axes>
```
Stored image: `cell_041_014.png`
```text
<IPython.core.display.HTML object>
```
<div style="font-family:Georgia,serif;font-size:13px;margin:6px 0;color:#222"><i>No anomalous months detected</i> (|Z| > 2) for PM<sub>2.5</sub> in the 2017–2025 period.</div>
```text
Anomalous months (|Z| > 2): 0

```

## Cell 43

```text
Saved → figures/fig15_pm_ratio.png

```
```text
<Figure size 1680x540 with 2 Axes>
```
Stored image: `cell_043_015.png`
```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Annual mean PM<sub>2.5</sub>/PM<sub>10</sub> ratio (source apportionment proxy).</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:center"><b>Year</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">PM<sub>2.5</sub>/PM<sub>10</sub></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2017</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.534</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2018</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.535</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2019</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.535</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2020</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.535</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2021</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.536</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2022</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.417</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2023</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.190</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2024</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.183</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2025</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.191</td></tr></tbody></table><div style="font-family:Georgia,serif;font-size:11.5px;color:#555;margin:3px 0"><i>Note:</i> ratio > 0.6 = combustion-dominated; < 0.4 = coarse-particle sources.</div></div>

## Cell 45

```text
Saved → figures/fig16_correlation_matrix.png

```
```text
<Figure size 1560x1320 with 2 Axes>
```
Stored image: `cell_045_016.png`

## Cell 47

```text
Saved → figures/fig17_pairwise_scatter.png

```
```text
<Figure size 1920x1080 with 6 Axes>
```
Stored image: `cell_047_017.png`

## Cell 49

```text
Saved → figures/fig18_source_apportionment.png

```
```text
<Figure size 1680x480 with 1 Axes>
```
Stored image: `cell_049_018.png`

## Cell 51

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Kruskal–Wallis H-test for seasonal differences (H<sub>0</sub>: equal medians across seasons).</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Variable</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">H</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">p</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Sig.</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>2.5</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">18.47</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.000353</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">***</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>10</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">92.87</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.000000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">***</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">NO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">87.36</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.000000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">***</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">78.77</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.000000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">***</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">18.51</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.000345</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">***</td></tr></tbody></table><div style="font-family:Georgia,serif;font-size:11.5px;color:#555;margin:3px 0"><sup>***</sup>p<0.001, <sup>**</sup>p<0.01, <sup>*</sup>p<0.05, ns: not significant.</div></div>

## Cell 53

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Dunn post-hoc test (Bonferroni-corrected p-values): PM<sub>2.5</sub>.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Winter</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Pre-monsoon</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Monsoon</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Post-monsoon</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Winter</b></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0300</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0003</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0116</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Pre-monsoon</b></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0300</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Monsoon</b></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0003</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Post-monsoon</b></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0116</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td></tr></tbody></table></div>
```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Dunn post-hoc test (Bonferroni-corrected p-values): AQI.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Winter</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Pre-monsoon</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Monsoon</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Post-monsoon</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Winter</b></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0397</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0003</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0121</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Pre-monsoon</b></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0397</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Monsoon</b></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0003</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Post-monsoon</b></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0121</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.0000</td></tr></tbody></table></div>

## Cell 55

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> COVID-19 lockdown impact: pre-lockdown vs. lockdown period (March–August 2020).</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Variable</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><span style="text-decoration:overline">x</span><sub>pre</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><span style="text-decoration:overline">x</span><sub>lock</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">Δ%</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">d</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Effect</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">p<sub>MWU</sub></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>2.5</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">141.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">136.3</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-3.9</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.258</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Small</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.1055</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>10</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">335.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">280.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-16.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.367</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Small</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.8026</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">NO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">42.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">29.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-31.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.913</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Large</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0200</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">35.2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">23.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-33.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.983</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Large</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0183</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">202.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">196.2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">-3.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.345</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Small</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.1043</td></tr></tbody></table><div style="font-family:Georgia,serif;font-size:11.5px;color:#555;margin:3px 0"><i>Note:</i> d = Cohen's d; |d|<0.5 = Small, 0.5–0.8 = Medium, >0.8 = Large.</div></div>
```text
Pre-COVID: 38 months | Lockdown: 6 months | Post-COVID: 64 months

```

## Cell 57

```text
Saved → figures/fig19_covid_impact.png

```
```text
<Figure size 2040x540 with 4 Axes>
```
Stored image: `cell_057_019.png`

## Cell 59

```text
Saved → figures/fig20_rainfall_scatter.png

```
```text
<Figure size 1560x1080 with 4 Axes>
```
Stored image: `cell_059_020.png`

## Cell 61

```text
Saved → figures/fig21_rainfall_dual_axis.png

```
```text
<Figure size 1680x480 with 2 Axes>
```
Stored image: `cell_061_021.png`

## Cell 63

```text
Saved → figures/fig22_rolling_correlation.png

```
```text
<Figure size 1560x840 with 3 Axes>
```
Stored image: `cell_063_022.png`

## Cell 65

```text
Saved → figures/fig23_socioeconomic.png

```
```text
<Figure size 1800x540 with 6 Axes>
```
Stored image: `cell_065_023.png`

## Cell 67

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Environmental Kuznets Curve (EKC) regression results.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Model</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Equation &amp; fit</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Linear</td><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI = 1107.2 - 1422.7·HDI, R<sup>2</sup> = 0.447, p(HDI) = 0.0491</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Quadratic</td><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI = -18919.9 + 59988.2·HDI - 47036.3·HDI<sup>2</sup>, R<sup>2</sup> = 0.616, p(HDI) = 0.1625, p(HDI<sup>2</sup>) = 0.1543</td></tr></tbody></table><div style="font-family:Georgia,serif;font-size:11.5px;color:#555;margin:3px 0"><i>Verdict:</i> Inconclusive: β<sub>2</sub> < 0 but not significant (p ≥ 0.05).</div></div>

## Cell 69

```text
Saved → figures/fig24_per_capita.png

```
```text
<Figure size 1560x540 with 2 Axes>
```
Stored image: `cell_069_024.png`

## Cell 71

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> WHO 2021 and US EPA guideline exceedance summary.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Pollutant</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>WHO</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Mean</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">×<b>WHO</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">\textbf{% > WHO}</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">\textbf{% > EPA}</th></tr><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>2.5</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">110.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">22.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">100.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">85.2</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>10</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">15</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">312.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">20.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">100.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">74.1</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">NO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">10</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">47.3</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">4.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">100.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.0</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">43.3</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">50.9</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">10.2</td></tr></tbody></table></div>

## Cell 72

```text
Saved → figures/fig25_exceedance.png

```
```text
<Figure size 2400x840 with 2 Axes>
```
Stored image: `cell_072_025.png`

## Cell 74

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Estimated PM<sub>2.5</sub>-attributable mortality, Dhaka City (2017–2025). IHME GBD 2019 South Asia CRF; β = 0.00575.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:center"><b>Year</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><span style="text-decoration:overline">PM<sub>2.5</sub></span></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Excess</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>AF</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Total deaths</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Attrib. deaths</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Per 100k</b></th></tr><tr><th style="padding:3px 12px;vertical-align:middle;text-align:center"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2017</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">131.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">126.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.5174</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">891,820</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">461,442</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">284.6</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2018</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">138.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">133.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.5356</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">899,426</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">481,771</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">294.6</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2019</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">145.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">140.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.5541</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">907,032</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">502,572</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">304.7</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2020</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">152.2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">147.2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.5711</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">914,639</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">522,375</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">314.1</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2021</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">158.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">153.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.5868</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">923,128</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">541,691</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">322.7</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2022</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">119.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">114.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.4809</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">931,616</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">447,974</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">264.5</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2023</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">50.3</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">45.3</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.2294</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">943,068</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">216,308</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">126.2</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2024</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">47.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">42.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.2179</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">960,856</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">209,407</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">119.9</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2025</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">52.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">47.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">0.2367</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">966,277</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">228,687</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">130.2</td></tr></tbody></table></div>

## Cell 75

```text
Saved → figures/fig26_health_burden.png

```
```text
<Figure size 1560x540 with 2 Axes>
```
Stored image: `cell_075_026.png`

## Cell 77

```text
Saved → figures/fig27_regional_comparison.png

```
```text
<Figure size 1920x600 with 2 Axes>
```
Stored image: `cell_077_027.png`

## Cell 79

```text
Training set: 84 months (2017-01-01 – 2023-12-01)
Test set:     24 months  (2024-01-01 – 2025-12-01)

```

## Cell 80

```text
Fitting models on training data and evaluating on test set...

```
```text
22:24:01 - cmdstanpy - INFO - Chain [1] start processing

```
```text
22:24:01 - cmdstanpy - INFO - Chain [1] done processing

```
```text
22:24:10 - cmdstanpy - INFO - Chain [1] start processing

```
```text
22:24:11 - cmdstanpy - INFO - Chain [1] done processing

```
```text
22:24:14 - cmdstanpy - INFO - Chain [1] start processing

```
```text
22:24:14 - cmdstanpy - INFO - Chain [1] done processing

```
```text
22:24:17 - cmdstanpy - INFO - Chain [1] start processing

```
```text
22:24:18 - cmdstanpy - INFO - Chain [1] done processing

```
```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Model evaluation — MAE (µg m<sup>-3</sup>) (held-out test period 2024–2025). Lower = better.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Model</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">PM<sub>2.5</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>AQI</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">NO<sub>2</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">SO<sub>2</sub></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">OLS</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">34.95</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">32.01</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">3.50</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">5.33</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">ETS</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">29.51</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">23.61</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">8.96</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">12.77</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SARIMA</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">39.85</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">43.97</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">3.84</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">12.55</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Prophet</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">48.52</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">82.22</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">1.72</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">2.75</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Ensemble</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">22.13</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">33.16</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">3.86</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">6.95</td></tr></tbody></table></div>
```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Model evaluation — RMSE (µg m<sup>-3</sup>) (held-out test period 2024–2025). Lower = better.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Model</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">PM<sub>2.5</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>AQI</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">NO<sub>2</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">SO<sub>2</sub></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">OLS</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">39.49</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">39.64</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">4.24</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">6.26</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">ETS</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">33.97</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">28.65</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">9.90</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">14.32</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SARIMA</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">49.73</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">51.13</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">4.68</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">13.27</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Prophet</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">55.57</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">93.78</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">2.64</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">3.97</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Ensemble</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">29.19</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.61</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">4.60</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">8.01</td></tr></tbody></table></div>
```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Model evaluation — MAPE% (%) (held-out test period 2024–2025). Lower = better.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Model</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">PM<sub>2.5</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>AQI</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">NO<sub>2</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">SO<sub>2</sub></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">OLS</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">116.76</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">40.11</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">9.53</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">12.73</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">ETS</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">67.08</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">22.55</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">23.69</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">33.34</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SARIMA</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">78.35</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">38.54</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">11.12</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">29.60</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Prophet</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">98.15</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">67.83</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">2.85</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">5.03</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">Ensemble</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">42.22</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">28.46</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">10.93</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">18.34</td></tr></tbody></table></div>
```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table.</b> Best model per variable (lowest MAPE%).</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:left"><b>Variable</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>Best model</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>MAPE (%)</b></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">PM<sub>2.5</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Ensemble</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">42.2</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">AQI</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">ETS</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">22.6</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">NO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Prophet</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">2.9</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:left">SO<sub>2</sub></td><td style="padding:3px 12px;vertical-align:middle;text-align:right">Prophet</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">5.0</td></tr></tbody></table></div>

## Cell 82

```text
Saved → figures/fig28_model_performance_heatmap.png

```
```text
<Figure size 2160x540 with 6 Axes>
```
Stored image: `cell_082_028.png`

## Cell 83

```text
Saved → figures/fig29_model_evaluation.png

```
```text
<Figure size 1920x1200 with 4 Axes>
```
Stored image: `cell_083_029.png`

## Cell 85

```text
Forecasting 60 months: 2026-01-01 → 2030-12-01

```

## Cell 86

```text
Fitting full-sample forecasts to 2030...

```
```text
22:24:44 - cmdstanpy - INFO - Chain [1] start processing

```
```text
22:24:44 - cmdstanpy - INFO - Chain [1] done processing

```
```text
PM₂.₅  | OLS R²=0.636 | Ensemble Dec 2030 = 44.3

```
```text
22:25:15 - cmdstanpy - INFO - Chain [1] start processing

```
```text
22:25:15 - cmdstanpy - INFO - Chain [1] done processing

```
```text
AQI    | OLS R²=0.642 | Ensemble Dec 2030 = 117.5

```
```text
22:25:28 - cmdstanpy - INFO - Chain [1] start processing

```
```text
22:25:28 - cmdstanpy - INFO - Chain [1] done processing

```
```text
NO₂    | OLS R²=0.953 | Ensemble Dec 2030 = 89.1

```
```text
22:25:35 - cmdstanpy - INFO - Chain [1] start processing

```
```text
22:25:35 - cmdstanpy - INFO - Chain [1] done processing

```
```text
SO₂    | OLS R²=0.916 | Ensemble Dec 2030 = 95.2

```

## Cell 87

```text
Saved → figures/fig30_forecasts_2030.png

```
```text
<Figure size 1920x1200 with 4 Axes>
```
Stored image: `cell_087_030.png`

## Cell 89

```text
<IPython.core.display.HTML object>
```
<style>.ptbl{border-collapse:collapse;font-family:Georgia,"Times New Roman",serif;font-size:12.5px}.ptbl thead tr{border-top:2.5px solid #222;border-bottom:1.5px solid #222}.ptbl tbody tr:last-child td{border-bottom:2.5px solid #222}</style><div style="margin:8px 0"><div style="font-family:Georgia,serif;font-size:12.5px;margin:0 0 4px"><b>Table 2.</b> Ensemble forecast — projected annual mean concentrations and AQI, Dhaka City (2026–2030). Business-as-usual scenario.</div><table class="ptbl"><thead><tr><th style="padding:3px 12px;vertical-align:middle;text-align:center"><b>Year</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">PM<sub>2.5</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">PM<sub>10</sub><sup>†</sup></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">NO<sub>2</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">SO<sub>2</sub></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"><b>AQI</b></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">×<b>WHO</b></th></tr><tr><th style="padding:3px 12px;vertical-align:middle;text-align:center"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right">(µg m<sup>-3</sup>)</th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th><th style="padding:3px 12px;vertical-align:middle;text-align:right"></th></tr></thead><tbody><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2026</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">49.9</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">92.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">58.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">61.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">120.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">10.0</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2027</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">42.5</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">78.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">60.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">64.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">113.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">8.5</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2028</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">35.8</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">66.3</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">62.4</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">66.7</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">106.1</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">7.2</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2029</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">30.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">56.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">64.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">70.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">99.3</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">6.1</td></tr><tr><td style="padding:3px 12px;vertical-align:middle;text-align:center">2030</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">26.2</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">48.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">66.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">73.0</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">92.6</td><td style="padding:3px 12px;vertical-align:middle;text-align:right">5.2</td></tr></tbody></table><div style="font-family:Georgia,serif;font-size:11.5px;color:#555;margin:3px 0"><sup>†</sup>PM<sub>10</sub> estimated from PM<sub>2.5</sub> using mean 2017–2025 ratio. WHO PM<sub>2.5</sub> annual guideline = 5 µg m<sup>-3</sup>.</div></div>
