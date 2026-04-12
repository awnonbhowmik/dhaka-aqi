"""
analysis.py
-----------
Statistical analysis functions used in the Dhaka AQI study.

Covers:
  - Mann-Kendall monotonic trend test
  - Augmented Dickey-Fuller stationarity test
  - Effect-size utilities (Cohen's d)
  - WHO / EPA guideline exceedance summary
  - Health-burden estimation (IHME GBD concentration-response)
  - Environmental Kuznets Curve (EKC) formal test
  - AQI category classification
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, shapiro
from statsmodels.tsa.stattools import adfuller

from .config import WHO, EPA, AQI_CATS, POL_COLS, POL_NAMES, POL_SHORT


# ── Trend Analysis ─────────────────────────────────────────────────────────────

def mann_kendall(x: np.ndarray) -> dict:
    """
    Two-sided Mann-Kendall monotonic trend test.

    Parameters
    ----------
    x : array-like
        Time-ordered scalar values.

    Returns
    -------
    dict with keys: S, z, p, Trend (text arrow label)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = sum(
        np.sign(x[j] - x[i])
        for i in range(n - 1)
        for j in range(i + 1, n)
    )
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    if z > 0 and p < 0.05:
        trend = "↑ Increasing"
    elif z < 0 and p < 0.05:
        trend = "↓ Decreasing"
    else:
        trend = "→ No significant trend"
    return {"S": int(s), "z": round(z, 3), "p": round(p, 4), "Trend": trend}


def run_mann_kendall_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Mann-Kendall for all five pollutant/AQI columns.

    Returns
    -------
    pd.DataFrame  (index = variable names)
    """
    rows = {}
    for col, label in zip(POL_COLS, POL_NAMES):
        rows[label] = mann_kendall(df[col].dropna().values)
    return pd.DataFrame(rows).T


def ols_trend_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    OLS regression of each pollutant against calendar year.

    Returns
    -------
    pd.DataFrame with columns: Slope, SE, t, p, R²
    """
    records = []
    for col, label in zip(POL_COLS, POL_NAMES):
        slope, intercept, r, p, se = stats.linregress(df["year"], df[col])
        records.append({
            "Variable": label,
            "Slope":    round(slope, 3),
            "SE":       round(se, 4),
            "t":        round(slope / se, 2),
            "p":        round(p, 4),
            "R²":       round(r**2, 3),
            "Sig":      "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "",
        })
    return pd.DataFrame(records).set_index("Variable")


# ── Stationarity ───────────────────────────────────────────────────────────────

def run_adf(series: pd.Series, label: str = "") -> dict:
    """
    Augmented Dickey-Fuller test on a time series.

    Returns
    -------
    dict with: label, adf_stat, p, lags, result ("Stationary" / "Non-stationary")
    """
    adf_stat, p_val, lags, *_ = adfuller(series.dropna(), autolag="AIC")
    return {
        "Variable":  label,
        "ADF stat":  round(adf_stat, 4),
        "p-value":   round(p_val, 4),
        "Lags":      lags,
        "Result":    "Stationary ✓" if p_val < 0.05 else "Non-stationary ✗",
    }


def run_adf_all(df: pd.DataFrame) -> pd.DataFrame:
    """ADF for all pollutant series (levels and first differences)."""
    level_rows = [run_adf(df[c], l) for c, l in zip(POL_COLS, POL_NAMES)]
    diff_rows  = [run_adf(df[c].diff().dropna(), l + " (Δ1)") for c, l in zip(POL_COLS, POL_NAMES)]
    return pd.DataFrame(level_rows + diff_rows)


# ── Effect sizes ───────────────────────────────────────────────────────────────

def cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    """Pooled Cohen's d effect size between two groups."""
    pooled_std = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
    return (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else np.nan


def covid_effect_table(pre: pd.DataFrame, lock: pd.DataFrame) -> pd.DataFrame:
    """
    Compute COVID lockdown effect sizes (Cohen's d, Mann-Whitney U, % change).

    Parameters
    ----------
    pre  : pre-COVID dataframe
    lock : lockdown-period dataframe

    Returns
    -------
    pd.DataFrame
    """
    records = []
    for col, label in zip(POL_COLS, POL_NAMES):
        pre_mu  = pre[col].mean()
        lock_mu = lock[col].mean()
        delta   = (lock_mu - pre_mu) / pre_mu * 100
        d       = cohens_d(pre[col], lock[col])
        _, p_mwu = mannwhitneyu(pre[col], lock[col], alternative="two-sided")
        mag = ("Small" if abs(d) < 0.5 else "Medium" if abs(d) < 0.8 else "Large")
        records.append({
            "Variable":     label,
            "Pre-COVID μ":  round(pre_mu, 1),
            "Lockdown μ":   round(lock_mu, 1),
            "Δ%":           round(delta, 1),
            "Cohen's d":    round(d, 3),
            "Magnitude":    mag,
            "MWU p":        round(p_mwu, 4),
        })
    return pd.DataFrame(records).set_index("Variable")


# ── Guideline Exceedance ───────────────────────────────────────────────────────

def exceedance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHO and US EPA guideline exceedance rates for all pollutants.

    Returns
    -------
    pd.DataFrame  (index = pollutant name)
    """
    records = []
    pairs = [
        ("PM₂.₅", "pm25_mean", WHO["pm25"], EPA["pm25"]),
        ("PM₁₀",  "pm10_mean", WHO["pm10"], EPA["pm10"]),
        ("NO₂",   "no2_mean",  WHO["no2"],  EPA["no2"]),
        ("SO₂",   "so2_mean",  WHO["so2"],  EPA["so2"]),
    ]
    for pol, col, who_val, epa_val in pairs:
        mean_val   = df[col].mean()
        above_who  = (df[col] > who_val).mean() * 100
        above_epa  = (df[col] > epa_val).mean() * 100
        records.append({
            "Pollutant":              pol,
            "WHO guideline (µg/m³)":  who_val,
            "Dataset mean (µg/m³)":   round(mean_val, 1),
            "× WHO guideline":        round(mean_val / who_val, 1),
            "% months > WHO":         round(above_who, 1),
            "% months > EPA":         round(above_epa, 1),
        })
    return pd.DataFrame(records).set_index("Pollutant")


# ── Health Burden ──────────────────────────────────────────────────────────────

def health_burden(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate annual PM₂.₅-attributable mortality using the IHME GBD 2019
    South Asia concentration-response function.

    Parameters
    ----------
    annual : annual-summary dataframe (from data_loader.compute_annual)

    Returns
    -------
    pd.DataFrame indexed by Year
    """
    CMR   = 5.5 / 1000   # crude mortality rate Bangladesh (World Bank 2022)
    BETA  = 0.00575       # IHME GBD 2019 South Asia, all-cause mortality
    C0    = WHO["pm25"]   # WHO annual guideline (5 µg/m³)

    records = []
    for _, row in annual.iterrows():
        C   = row["pm25_mean"]
        pop = row["population_total"]
        excess       = max(C - C0, 0)
        AF           = 1 - np.exp(-BETA * excess)
        deaths_total = pop * CMR
        deaths_attr  = deaths_total * AF
        records.append({
            "Year":                       int(row["year"]),
            "PM₂.₅ mean (µg/m³)":        round(C, 1),
            "Excess above WHO":           round(excess, 1),
            "Attributable Fraction":      round(AF, 4),
            "Est. total deaths":          int(deaths_total),
            "PM₂.₅-attributable deaths": int(deaths_attr),
            "Deaths per 100k pop":        round(deaths_attr / pop * 100000, 1),
        })
    return pd.DataFrame(records).set_index("Year")


# ── Environmental Kuznets Curve ────────────────────────────────────────────────

def ekc_analysis(annual: pd.DataFrame) -> dict:
    """
    Formal EKC test: AQI ~ β₀ + β₁·HDI + β₂·HDI²

    Returns
    -------
    dict with keys: linear_model, quad_model, verdict, turning_point (or None)
    """
    hdi_vals = annual["hdi"].values
    aqi_vals = annual["aqi_mean"].values

    X_lin  = sm.add_constant(hdi_vals)
    mdl_lin = sm.OLS(aqi_vals, X_lin).fit()

    hdi_sq  = hdi_vals ** 2
    X_quad  = sm.add_constant(np.column_stack([hdi_vals, hdi_sq]))
    mdl_quad = sm.OLS(aqi_vals, X_quad).fit()

    beta2   = mdl_quad.params[2]
    p_beta2 = mdl_quad.pvalues[2]

    if beta2 < 0 and p_beta2 < 0.05:
        verdict       = "EKC CONFIRMED: β₂ < 0 and significant → inverted-U relationship."
        turning_point = -mdl_quad.params[1] / (2 * mdl_quad.params[2])
    elif beta2 > 0:
        verdict       = "EKC NOT confirmed: Bangladesh is still on the upward slope."
        turning_point = None
    else:
        verdict       = "Inconclusive: β₂ < 0 but not significant (p ≥ 0.05)."
        turning_point = None

    return {
        "linear_model":  mdl_lin,
        "quad_model":    mdl_quad,
        "verdict":       verdict,
        "turning_point": turning_point,
    }


# ── AQI category utilities ─────────────────────────────────────────────────────

def classify_aqi(value: float) -> str:
    """Return the US EPA AQI category label for a given AQI value."""
    for lo, hi, label, _ in AQI_CATS:
        if lo <= value <= hi:
            return label
    return "Hazardous"


def aqi_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Frequency distribution of monthly AQI values across US EPA categories.

    Returns
    -------
    pd.DataFrame  columns: Category, Count, % of Months
    """
    df = df.copy()
    df["AQI_Cat"] = df["aqi_mean"].apply(classify_aqi)
    cat_order = [c for _, _, c, _ in AQI_CATS]
    counts = (
        df["AQI_Cat"]
        .value_counts()
        .reindex(cat_order, fill_value=0)
    )
    pct = (counts / len(df) * 100).round(1)
    return pd.DataFrame({"Category": counts.index, "Count": counts.values, "% of Months": pct.values})


# ── Normality / Descriptive ────────────────────────────────────────────────────

def descriptive_with_normality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended descriptive statistics table including skewness, kurtosis,
    and Shapiro-Wilk normality p-value.
    """
    from .config import POL_COLS, POL_NAMES
    summary = df[POL_COLS].describe().T.round(2)
    summary.columns = ["n", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
    summary.index = POL_NAMES
    summary["Skewness"] = df[POL_COLS].skew().values.round(3)
    summary["Kurtosis"] = df[POL_COLS].kurt().values.round(3)
    summary["Shapiro p"] = [
        round(shapiro(df[c].dropna())[1], 4) for c in POL_COLS
    ]
    summary["Distribution"] = [
        "Non-normal" if p < 0.05 else "Normal" for p in summary["Shapiro p"]
    ]
    return summary
