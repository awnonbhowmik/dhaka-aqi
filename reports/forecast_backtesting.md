# Forecast backtesting

Models were evaluated by expanding-window rolling origins with a 12-month test horizon.
All complex models were compared with seasonal-naive, naive, and drift baselines.
MASE uses the in-sample seasonal-naive scale. sMAPE is reported instead of MAPE.
All statistical models are fit to log1p concentration, so forecasts and intervals are non-negative without clipping a negative-scale model.

Selected model: `sarima` (lowest mean cross-validated MASE).
Primary horizon: 24 months (2025-03-01 to 2027-02-01).

Three internal months (2020-05, 2021-09, 2022-07) have 66-71% valid-day coverage. They remain explicit partial-month means in forecasting to preserve the regular calendar without interpolation. This is a material limitation; no value was filled.

The 2030 lines are deterministic benchmarks, not empirical forecasts. Uncertainty intervals are intentionally blank for policy targets because no defensible implementation-probability model exists; inventing one would create false precision.

```csv
model,mae,rmse,mase,smape_pct,interval_coverage_pct,folds
sarima,13.642581894895514,18.05067562280457,0.8450383610660497,14.830117217833248,100.0,3
ets,14.317403960354298,18.017521121600733,0.8872751686646084,15.330165283153358,100.0,3
regression,15.163911376591132,18.940913010659404,0.9377667652140441,15.701272748199642,94.44444444444444,3
seasonal_naive,16.554083333333335,21.23435140922049,1.034843846515366,18.100383202586382,97.22222222222221,3
drift,79.34053039914367,89.04206619509789,4.845335787222802,67.40519420477303,100.0,3
naive,80.74227777777777,90.8011525268265,4.940064215848419,68.2013313000798,100.0,3
```
