# Stored outputs: analysis.ipynb

## Cell 2

```text
Setup complete.

```

## Cell 4

```text
Daily rows : 3,407  (2017-01-01 → 2026-05-02)
Source split: {'daily_aqi': 2141, 'monthly_avg': 1266}

Null counts:
pm25     120
pm10     120
no2      120
so2      120
co      1216
o3      1216
dtype: int64

```

## Cell 6

```text
Daily Descriptive Statistics (2017–2026)

```
```text
                  n   Mean    Std    Min    Q25 Median     Q75     Max  \
Variable                                                                 
AQI            3407  164.4   56.0   19.0  122.0  182.5   191.4   376.0   
PM₂.₅ (µg/m³)  3287   90.0   65.8   11.8   38.8   66.0   128.2   340.7   
PM₁₀ (µg/m³)   3287  191.2  118.8   21.4   91.9  159.7   287.0   479.9   
NO₂ (µg/m³)    3287   33.5   21.3    4.8   17.7   27.3    42.6    98.4   
SO₂ (µg/m³)    3287   27.8   23.9    2.5    8.0   21.1    31.8    88.0   
CO (µg/m³)     2191  984.5  759.1  133.4  305.9  737.4  1570.2  3435.7   
O₃ (µg/m³)     2191   58.7   14.9   20.3   48.4   56.8    67.4   118.3   

              Skewness Kurtosis    SW_p Normal  
Variable                                        
AQI               -0.2     -0.4  0.0000     No  
PM₂.₅ (µg/m³)      1.0      0.2  0.0000     No  
PM₁₀ (µg/m³)       0.5     -0.9  0.0000     No  
NO₂ (µg/m³)        1.1      0.1  0.0000     No  
SO₂ (µg/m³)        1.1      0.2  0.0000     No  
CO (µg/m³)         0.7     -0.6  0.0000     No  
O₃ (µg/m³)         0.5      0.3  0.0000     No  
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Min</th>
      <th>Q25</th>
      <th>Median</th>
      <th>Q75</th>
      <th>Max</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
      <th>SW_p</th>
      <th>Normal</th>
    </tr>
    <tr>
      <th>Variable</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AQI</th>
      <td>3407</td>
      <td>164.4</td>
      <td>56.0</td>
      <td>19.0</td>
      <td>122.0</td>
      <td>182.5</td>
      <td>191.4</td>
      <td>376.0</td>
      <td>-0.2</td>
      <td>-0.4</td>
      <td>0.0000</td>
      <td>No</td>
    </tr>
    <tr>
      <th>PM₂.₅ (µg/m³)</th>
      <td>3287</td>
      <td>90.0</td>
      <td>65.8</td>
      <td>11.8</td>
      <td>38.8</td>
      <td>66.0</td>
      <td>128.2</td>
      <td>340.7</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>0.0000</td>
      <td>No</td>
    </tr>
    <tr>
      <th>PM₁₀ (µg/m³)</th>
      <td>3287</td>
      <td>191.2</td>
      <td>118.8</td>
      <td>21.4</td>
      <td>91.9</td>
      <td>159.7</td>
      <td>287.0</td>
      <td>479.9</td>
      <td>0.5</td>
      <td>-0.9</td>
      <td>0.0000</td>
      <td>No</td>
    </tr>
    <tr>
      <th>NO₂ (µg/m³)</th>
      <td>3287</td>
      <td>33.5</td>
      <td>21.3</td>
      <td>4.8</td>
      <td>17.7</td>
      <td>27.3</td>
      <td>42.6</td>
      <td>98.4</td>
      <td>1.1</td>
      <td>0.1</td>
      <td>0.0000</td>
      <td>No</td>
    </tr>
    <tr>
      <th>SO₂ (µg/m³)</th>
      <td>3287</td>
      <td>27.8</td>
      <td>23.9</td>
      <td>2.5</td>
      <td>8.0</td>
      <td>21.1</td>
      <td>31.8</td>
      <td>88.0</td>
      <td>1.1</td>
      <td>0.2</td>
      <td>0.0000</td>
      <td>No</td>
    </tr>
    <tr>
      <th>CO (µg/m³)</th>
      <td>2191</td>
      <td>984.5</td>
      <td>759.1</td>
      <td>133.4</td>
      <td>305.9</td>
      <td>737.4</td>
      <td>1570.2</td>
      <td>3435.7</td>
      <td>0.7</td>
      <td>-0.6</td>
      <td>0.0000</td>
      <td>No</td>
    </tr>
    <tr>
      <th>O₃ (µg/m³)</th>
      <td>2191</td>
      <td>58.7</td>
      <td>14.9</td>
      <td>20.3</td>
      <td>48.4</td>
      <td>56.8</td>
      <td>67.4</td>
      <td>118.3</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.0000</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>

## Cell 8

```text
<Figure size 1680x960 with 2 Axes>
```
Stored image: `cell_008_001.png`
```text
Saved → figures/analysis_fig01_daily_time_series.png

```

## Cell 10

```text
<Figure size 1560x600 with 1 Axes>
```
Stored image: `cell_010_002.png`
```text
Saved → figures/analysis_fig02_monthly_pm25_boxplot.png

```

## Cell 11

```text
<Figure size 1560x600 with 2 Axes>
```
Stored image: `cell_011_003.png`
```text
Saved → figures/analysis_fig03_pm25_heatmap.png

```

## Cell 13

```text
Days per year exceeding PM₂.₅ thresholds:

```
```text
' year  > WHO 24-hr (15)  > Bangladesh 24-hr (65)  > Severe (150 µg/m³)  > Hazardous (250 µg/m³)\n 2017               365                      223                    96                        2\n 2018               365                      238                   122                        7\n 2019               365                      240                   102                        7\n 2020               366                      229                   116                       15\n 2021               365                      229                   126                       22\n 2022               365                      226                   112                       14\n 2023               334                       59                     0                        0\n 2024               304                      121                     0                        0\n 2025               365                       90                     0                        0'
```

## Cell 14

```text
<Figure size 1320x600 with 1 Axes>
```
Stored image: `cell_014_004.png`
```text
Saved → figures/analysis_fig04_pm25_exceedance.png

```

## Cell 15

```text
<Figure size 1200x600 with 1 Axes>
```
Stored image: `cell_015_005.png`
```text
Saved → figures/analysis_fig05_annual_pm25_vs_standards.png

```

## Cell 17

```text
<Figure size 1680x840 with 6 Axes>
```
Stored image: `cell_017_006.png`
```text
Saved → figures/analysis_fig06_cams_seasonal_cycle.png

```

## Cell 18

```text
<Figure size 840x720 with 2 Axes>
```
Stored image: `cell_018_007.png`
```text
Saved → figures/analysis_fig07_pollutant_correlation.png

```

## Cell 20

```text
COVID analysis — PM₂.₅ (µg/m³):
  Pre-lockdown  (2019–Mar2020): mean=122.3, median=112.4
  Lockdown      (Mar–Aug 2020): mean=53.6, median=43.4
  Post-lockdown (Sep2020–2022): mean=116.2, median=98.6
  Mann-Whitney U (pre > lock): p = 0.0000

```

## Cell 21

```text
<Figure size 1560x600 with 1 Axes>
```
Stored image: `cell_021_008.png`
```text
Saved → figures/analysis_fig08_covid_daily.png

```

## Cell 23

```text
<Figure size 1440x600 with 1 Axes>
```
Stored image: `cell_023_009.png`
```text
Saved → figures/analysis_fig09_aqi_category_days.png

AQI category % per year:

```
```text
aqi_cat  Good  Moderate  Unhealthy for SG  Unhealthy  Very Unhealthy  \
year                                                                   
2017      0.0       0.0               0.0       75.3            24.7   
2018      0.0       0.0               0.0       75.3            24.7   
2019      0.0       0.0               0.0       75.3            24.7   
2020      1.6      28.7              20.2       23.5            25.7   
2021      2.2      20.0              23.0       24.4            26.0   
2022      0.0      14.5              28.2       34.0            22.7   
2023      0.0      16.4              34.8       32.9            15.9   
2024      3.6      28.1              21.6       32.0            14.2   
2025      0.0      50.1              19.5       27.9             2.5   

aqi_cat  Hazardous  
year                
2017           0.0  
2018           0.0  
2019           0.0  
2020           0.3  
2021           4.4  
2022           0.5  
2023           0.0  
2024           0.5  
2025           0.0  
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>aqi_cat</th>
      <th>Good</th>
      <th>Moderate</th>
      <th>Unhealthy for SG</th>
      <th>Unhealthy</th>
      <th>Very Unhealthy</th>
      <th>Hazardous</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.3</td>
      <td>24.7</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.3</td>
      <td>24.7</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.3</td>
      <td>24.7</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>1.6</td>
      <td>28.7</td>
      <td>20.2</td>
      <td>23.5</td>
      <td>25.7</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>2.2</td>
      <td>20.0</td>
      <td>23.0</td>
      <td>24.4</td>
      <td>26.0</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>0.0</td>
      <td>14.5</td>
      <td>28.2</td>
      <td>34.0</td>
      <td>22.7</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>0.0</td>
      <td>16.4</td>
      <td>34.8</td>
      <td>32.9</td>
      <td>15.9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>3.6</td>
      <td>28.1</td>
      <td>21.6</td>
      <td>32.0</td>
      <td>14.2</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>0.0</td>
      <td>50.1</td>
      <td>19.5</td>
      <td>27.9</td>
      <td>2.5</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

## Cell 25

```text
<Figure size 1800x600 with 3 Axes>
```
Stored image: `cell_025_010.png`
```text
Saved → figures/analysis_fig10_socioeconomic.png

```

## Cell 27

```text
Table 1: Annual PM₂.₅ Summary

```
```text
' year  n_days       mean   median       p95  days_exc_who  days_exc_bgd  who_multiple  bgd_exc_pct  who_exc_pct\n 2017     365 101.493205  82.8400 206.31600           365           223          20.3         61.1        100.0\n 2018     365 114.332055  96.2980 227.80200           365           238          22.9         65.2        100.0\n 2019     365 107.884627  91.1150 226.63160           365           240          21.6         65.8        100.0\n 2020     366 111.521109  92.8245 243.27725           366           229          22.3         62.6        100.0\n 2021     365 118.126129 100.5420 256.24360           365           229          23.6         62.7        100.0\n 2022     365 107.460575  84.3320 240.79080           365           226          21.5         61.9        100.0\n 2023     365  50.231781  51.3600 109.95000           334            59          10.0         16.2         91.5\n 2024     366  47.677486  48.4500  86.71000           304           121           9.5         33.1         83.1\n 2025     365  51.735260  37.9900 108.39000           365            90          10.3         24.7        100.0'
```

## Cell 28

```text
Table 2: Seasonal PM₂.₅ Summary (all years)

```
```text
'      season  n_days  pm25_mean  pm25_median  pm25_p5  pm25_p95  aqi_mean\n      Winter     812      160.5        170.2     70.4     263.7     221.1\n Pre-monsoon     828       78.6         64.6     32.8     156.8     161.1\n     Monsoon    1098       36.9         31.8     12.5      76.7     126.0\nPost-monsoon     549      109.5         98.4     48.4     209.0     163.7'
```

## Cell 29

```text
Table 3: COVID Lockdown PM₂.₅ Impact

```
```text
'       Period  n_days  PM25_mean  PM25_median  AQI_mean  vs_pre_pct\n Pre-lockdown     815      118.7        106.3     205.2         0.0\n     Lockdown     159       53.6         43.4     115.2       -54.8\nPost-lockdown     852      116.2         98.6     160.5        -2.1'
```

## Cell 30

```text
Analysis figures saved (10 total):
  figures/analysis_fig01_daily_time_series.png
  figures/analysis_fig02_monthly_pm25_boxplot.png
  figures/analysis_fig03_pm25_heatmap.png
  figures/analysis_fig04_pm25_exceedance.png
  figures/analysis_fig05_annual_pm25_vs_standards.png
  figures/analysis_fig06_cams_seasonal_cycle.png
  figures/analysis_fig07_pollutant_correlation.png
  figures/analysis_fig08_covid_daily.png
  figures/analysis_fig09_aqi_category_days.png
  figures/analysis_fig10_socioeconomic.png

```

## Cell 32

```text
<Figure size 1680x1200 with 4 Axes>
```
Stored image: `cell_032_011.png`
```text
Saved → figures/analysis_fig11_stl_daily_pm25.png

```

## Cell 33

```text
<Figure size 1560x720 with 2 Axes>
```
Stored image: `cell_033_012.png`
```text
Saved → figures/analysis_fig12_stl_monthly_pm25.png
STL trend slope: -1.070 µg/m³/month  (-12.84 µg/m³/year)
Over full period: -115.6 µg/m³ total change

```

## Cell 35

```text
2024 baseline annual mean PM2.5 : 47.8 µg/m³
NAQMP Nationwide target (2030)  : 32.8 µg/m³
NAQMP Dhaka GDA target (2030)   : 17.8 µg/m³
WHO annual guideline            :  5.0 µg/m³

Forecast comparison at 2030-12:
  BAU ensemble : 53.4 µg/m³
  NAQMP nation : 42.1 µg/m³
  NAQMP GDA    : 22.8 µg/m³

```

## Cell 36

```text
<Figure size 1680x720 with 1 Axes>
```
Stored image: `cell_036_013.png`
```text
Saved → figures/analysis_fig13_forecast_2026_2030.png

```

## Cell 37

```text
<Figure size 1560x600 with 1 Axes>
```
Stored image: `cell_037_014.png`
```text
Saved → figures/analysis_fig14_annual_forecast_bar.png
2030 projected annual mean PM₂.₅:
  BAU ensemble     : 22.5 µg/m³  (4.5× WHO annual)
  NAQMP Nationwide : 34.1 µg/m³  (6.8× WHO annual)
  NAQMP Dhaka GDA  : 20.4 µg/m³  (4.1× WHO annual)
  WHO annual guide : 5 µg/m³

```

## Cell 39

```text
<Figure size 1560x600 with 1 Axes>
```
Stored image: `cell_039_015.png`
```text
Saved → figures/analysis_fig15_rolling_annual_mean.png

```

## Cell 40

```text
All analysis figures (15 total):
  figures/analysis_fig01_daily_time_series.png  (623 KB)
  figures/analysis_fig02_monthly_pm25_boxplot.png  (172 KB)
  figures/analysis_fig03_pm25_heatmap.png  (273 KB)
  figures/analysis_fig04_pm25_exceedance.png  (172 KB)
  figures/analysis_fig05_annual_pm25_vs_standards.png  (162 KB)
  figures/analysis_fig06_cams_seasonal_cycle.png  (706 KB)
  figures/analysis_fig07_pollutant_correlation.png  (151 KB)
  figures/analysis_fig08_covid_daily.png  (521 KB)
  figures/analysis_fig09_aqi_category_days.png  (148 KB)
  figures/analysis_fig10_socioeconomic.png  (333 KB)
  figures/analysis_fig11_stl_daily_pm25.png  (669 KB)
  figures/analysis_fig12_stl_monthly_pm25.png  (319 KB)
  figures/analysis_fig13_forecast_2026_2030.png  (468 KB)
  figures/analysis_fig14_annual_forecast_bar.png  (184 KB)
  figures/analysis_fig15_rolling_annual_mean.png  (269 KB)

```
