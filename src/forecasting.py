"""
forecasting.py
--------------
Predictive modelling and multi-model ensemble for Dhaka AQI 2026-2030.

Models
------
  OLS      – Year + month-dummy linear regression
  ETS      – Holt-Winters Exponential Smoothing (additive trend & season)
  SARIMA   – Auto-ARIMA with seasonal differencing (pmdarima)
  Prophet  – Facebook Prophet (multiplicative seasonality)
  Ensemble – Simple mean of all four model forecasts
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Compute MAE, RMSE and MAPE for a single model.

    Parameters
    ----------
    y_true : observed values
    y_pred : predicted values
    name   : model label

    Returns
    -------
    dict with keys: Model, MAE, RMSE, MAPE%
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {
        "Model": name,
        "MAE":   round(mae, 2),
        "RMSE":  round(rmse, 2),
        "MAPE%": round(mape, 2),
    }


# ── Individual model helpers ───────────────────────────────────────────────────

def _ols_month_dummies(train_df: pd.DataFrame, target: str,
                       pred_df: pd.DataFrame) -> np.ndarray:
    """OLS with year + month dummies. Used for both test-set eval and forecasting."""
    X_tr = pd.get_dummies(train_df["month"].astype(str), prefix="m", drop_first=True)
    X_tr["year"] = train_df["year"].values
    mdl = LinearRegression().fit(X_tr, train_df[target].values)
    X_pr = pd.get_dummies(pred_df["month"].astype(str), prefix="m", drop_first=True)
    X_pr = X_pr.reindex(columns=X_tr.columns, fill_value=0)
    X_pr["year"] = pred_df["year"].values
    return np.clip(mdl.predict(X_pr), 0, None)


def _ets_forecast(train_series: pd.Series, n: int) -> np.ndarray:
    """Holt-Winters additive trend + additive seasonal (period = 12)."""
    mdl = ExponentialSmoothing(
        train_series, trend="add", seasonal="add", seasonal_periods=12
    ).fit(optimized=True)
    return np.clip(mdl.forecast(n).values, 0, None)


def _sarima_forecast(train_series: pd.Series, n: int) -> np.ndarray:
    """Auto-ARIMA with seasonal order selection via AIC."""
    import pmdarima as pm
    auto = pm.auto_arima(
        train_series, seasonal=True, m=12,
        information_criterion="aic", stepwise=True,
        suppress_warnings=True, error_action="ignore",
        max_p=3, max_q=3, max_P=2, max_Q=2,
    )
    return np.clip(np.array(auto.predict(n_periods=n)), 0, None)


def _prophet_forecast(df: pd.DataFrame, target: str,
                      n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Facebook Prophet with multiplicative yearly seasonality.

    Returns
    -------
    yhat, yhat_lower, yhat_upper  (all clipped to ≥ 0)
    """
    from prophet import Prophet

    pt = (
        df[["month_start", target]]
        .rename(columns={"month_start": "ds", target: "y"})
    )
    m = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        interval_width=0.90,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(pt)
    fut = m.make_future_dataframe(periods=n, freq="MS", include_history=False)
    fc  = m.predict(fut)
    return (
        np.clip(fc["yhat"].values, 0, None),
        np.clip(fc["yhat_lower"].values, 0, None),
        np.clip(fc["yhat_upper"].values, 0, None),
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_models(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    targets:  list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit all four models on *df_train*, evaluate on *df_test*, and return
    a performance table and a dict of predictions.

    Parameters
    ----------
    df_train : training split
    df_test  : held-out test split
    targets  : list of (column_name, label) pairs; defaults to PM2.5, AQI, NO2, SO2

    Returns
    -------
    eval_df    : pd.DataFrame — model evaluation metrics (long format)
    model_preds: dict — {target_col: {"true": …, "preds": {model: array}, "label": …}}
    """
    if targets is None:
        targets = [
            ("pm25_mean", "PM₂.₅"),
            ("aqi_mean",  "AQI"),
            ("no2_mean",  "NO₂"),
            ("so2_mean",  "SO₂"),
        ]

    eval_table  = []
    model_preds = {}

    for col, label in targets:
        y_true       = df_test[col].values
        n_test       = len(df_test)
        train_series = df_train.set_index("month_start")[col]

        preds = {}

        yhat_ols = _ols_month_dummies(df_train, col, df_test)
        eval_table.append({**eval_metrics(y_true, yhat_ols, "OLS"),     "Variable": label})
        preds["OLS"] = yhat_ols

        yhat_ets = _ets_forecast(train_series, n_test)
        eval_table.append({**eval_metrics(y_true, yhat_ets, "ETS"),     "Variable": label})
        preds["ETS"] = yhat_ets

        yhat_sarima = _sarima_forecast(train_series, n_test)
        eval_table.append({**eval_metrics(y_true, yhat_sarima, "SARIMA"), "Variable": label})
        preds["SARIMA"] = yhat_sarima

        yhat_prophet, _, _ = _prophet_forecast(df_train, col, n_test)
        eval_table.append({**eval_metrics(y_true, yhat_prophet, "Prophet"), "Variable": label})
        preds["Prophet"] = yhat_prophet

        yhat_ens = np.mean([yhat_ols, yhat_ets, yhat_sarima, yhat_prophet], axis=0)
        eval_table.append({**eval_metrics(y_true, yhat_ens, "Ensemble"),  "Variable": label})
        preds["Ensemble"] = yhat_ens

        model_preds[col] = {"true": y_true, "preds": preds, "label": label}

    eval_df = pd.DataFrame(eval_table)
    return eval_df, model_preds


def forecast_to_2030(
    df:         pd.DataFrame,
    future_df:  pd.DataFrame,
    targets:    list[tuple[str, str]] | None = None,
) -> dict:
    """
    Fit all four models on the full observed dataset and project to 2030.

    Parameters
    ----------
    df         : complete observed monthly dataset
    future_df  : future dates dataframe (from data_loader.build_future_dates)
    targets    : list of (column_name, label) pairs

    Returns
    -------
    dict keyed by column name, each value containing model arrays,
    Prophet CI bounds, ensemble, and OLS R².
    """
    if targets is None:
        targets = [
            ("pm25_mean", "PM₂.₅"),
            ("aqi_mean",  "AQI"),
            ("no2_mean",  "NO₂"),
            ("so2_mean",  "SO₂"),
        ]

    n_fut      = len(future_df)
    forecasts  = {}

    for target, label in targets:
        series = df.set_index("month_start")[target]

        # OLS with R²
        X_hist = pd.get_dummies(df["month"].astype(str), prefix="m", drop_first=True)
        X_hist["year"] = df["year"].values
        ols_mdl = LinearRegression().fit(X_hist, df[target].values)
        r2_ols  = ols_mdl.score(X_hist, df[target].values)
        X_fut   = pd.get_dummies(future_df["month"].astype(str), prefix="m", drop_first=True)
        X_fut   = X_fut.reindex(columns=X_hist.columns, fill_value=0)
        X_fut["year"] = future_df["year"].values
        ols_yhat = np.clip(ols_mdl.predict(X_fut), 0, None)

        ets_yhat              = _ets_forecast(series, n_fut)
        sarima_yhat           = _sarima_forecast(series, n_fut)
        pr_yhat, pr_lo, pr_hi = _prophet_forecast(df, target, n_fut)
        ens_yhat              = np.mean([ols_yhat, ets_yhat, sarima_yhat, pr_yhat], axis=0)

        forecasts[target] = {
            "label":    label,
            "r2_ols":   r2_ols,
            "ols":      ols_yhat,
            "ets":      ets_yhat,
            "sarima":   sarima_yhat,
            "prophet":  pr_yhat,
            "ensemble": ens_yhat,
            "p_lo":     pr_lo,
            "p_hi":     pr_hi,
        }
        print(f"{label:6s} | OLS R²={r2_ols:.3f} | Ensemble Dec 2030 = {ens_yhat[-1]:.1f}")

    return forecasts


def scenario_projections(
    df:          pd.DataFrame,
    future_df:   pd.DataFrame,
    target:      str,
    monthly_clim: pd.DataFrame,
    clim_col:    str,
) -> dict:
    """
    Generate three policy scenario projections (monthly) to 2030.

    Scenarios
    ---------
    BAU        : continuation of OLS annual trend
    Moderate   : 30 % reduction in trend slope relative to last observed value
    Ambitious  : 50 % reduction in concentration by 2030 (linear decline)

    Returns
    -------
    dict with keys: yrs, bau_annual, mod_annual, amb_annual,
                    and monthly versions of each (Timestamps + arrays)
    """
    ann_vals = df.groupby("year")[target].mean().reset_index()
    slope, intercept, _, _, se_slope = pd.Series(
        np.polyfit(ann_vals["year"].values, ann_vals[target].values, 1, full=False)
        if False else (0, 0, 0, 0, 0)
    ), 0, 0, 0, 0

    # Recompute properly with scipy
    import scipy.stats as sc
    slope, intercept, _, _, se_slope = sc.linregress(ann_vals["year"], ann_vals[target])

    last_val = ann_vals[ann_vals["year"] == ann_vals["year"].max()][target].values[0]
    yrs      = np.arange(future_df["year"].min(), future_df["year"].max() + 1)

    bau_vals = intercept + slope * yrs
    mod_vals = last_val + (slope * 0.70) * (yrs - ann_vals["year"].max())
    amb_vals = np.linspace(last_val, last_val * 0.50, len(yrs))

    def _to_monthly(annual_proj, yrs_arr):
        sc_vals = monthly_clim[clim_col].values / monthly_clim[clim_col].mean()
        months, vals = [], []
        for yr, am in zip(yrs_arr, annual_proj):
            for mo_i, sc_i in enumerate(sc_vals):
                months.append(pd.Timestamp(f"{yr}-{mo_i + 1:02d}-01"))
                vals.append(max(am * sc_i, 0))
        return pd.DatetimeIndex(months), np.array(vals)

    mo_bau, bau_mo = _to_monthly(bau_vals, yrs)
    mo_mod, mod_mo = _to_monthly(mod_vals, yrs)
    mo_amb, amb_mo = _to_monthly(amb_vals, yrs)

    n_ann = len(ann_vals)
    bau_se = se_slope * np.sqrt(
        (yrs - ann_vals["year"].mean()) ** 2
        / np.sum((ann_vals["year"] - ann_vals["year"].mean()) ** 2) + 1 / n_ann
    ) * 12
    bau_se_mo = np.repeat(bau_se, 12)[: len(mo_bau)]

    return {
        "yrs":       yrs,
        "bau_vals":  bau_vals,
        "mod_vals":  mod_vals,
        "amb_vals":  amb_vals,
        "mo_bau":    mo_bau, "bau_mo": bau_mo, "bau_se_mo": bau_se_mo,
        "mo_mod":    mo_mod, "mod_mo": mod_mo,
        "mo_amb":    mo_amb, "amb_mo": amb_mo,
    }


def forecast_summary_table(future_df: pd.DataFrame, forecasts: dict) -> pd.DataFrame:
    """
    Annual mean ensemble forecast table for 2026-2030.

    Returns
    -------
    pd.DataFrame  index = Year
    """
    fdf = future_df.copy()
    for target, res in forecasts.items():
        fdf[f"ens_{target}"] = res["ensemble"]

    records = []
    for yr in range(fdf["year"].min(), fdf["year"].max() + 1):
        sub    = fdf[fdf["year"] == yr]
        pm25   = sub["ens_pm25_mean"].mean() if "ens_pm25_mean" in sub.columns else np.nan
        no2    = sub["ens_no2_mean"].mean()  if "ens_no2_mean"  in sub.columns else np.nan
        so2    = sub["ens_so2_mean"].mean()  if "ens_so2_mean"  in sub.columns else np.nan
        aqi    = sub["ens_aqi_mean"].mean()  if "ens_aqi_mean"  in sub.columns else np.nan
        pm10_e = pm25 / 0.54 if not np.isnan(pm25) else np.nan   # approximate ratio
        from .config import WHO as _WHO
        who_mult = pm25 / _WHO["pm25"] if not np.isnan(pm25) else np.nan
        records.append({
            "Year":           yr,
            "PM₂.₅ (µg/m³)": round(pm25, 1),
            "PM₁₀ est.":      round(pm10_e, 1),
            "NO₂ (µg/m³)":   round(no2, 1),
            "SO₂ (µg/m³)":   round(so2, 1),
            "AQI":            round(aqi, 1),
            "× WHO PM₂.₅":   round(who_mult, 1),
        })
    return pd.DataFrame(records).set_index("Year")
