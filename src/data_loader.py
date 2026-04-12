"""
data_loader.py
--------------
Functions for loading, validating, and preparing the Dhaka AQI dataset.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd

from .config import POL_COLS, SEASON_ORDER

# ── Default path (relative to project root) ────────────────────────────────────
DEFAULT_CSV = pathlib.Path("data/final_dhaka_aqi_dataset_clean.csv")


def load_data(path: str | pathlib.Path = DEFAULT_CSV) -> pd.DataFrame:
    """
    Load and lightly validate the monthly Dhaka AQI dataset.

    Parameters
    ----------
    path : path-like
        Path to ``final_dhaka_aqi_dataset_clean.csv``.

    Returns
    -------
    pd.DataFrame
        Sorted, typed monthly dataframe with season as an ordered Categorical.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Place final_dhaka_aqi_dataset_clean.csv in data/ relative to "
            "the project root."
        )

    df = pd.read_csv(path, parse_dates=["month_start"])
    df = df.sort_values("month_start").reset_index(drop=True)

    # Ordered season categorical for correct sort/groupby behaviour
    df["season"] = pd.Categorical(
        df["season"], categories=SEASON_ORDER, ordered=True
    )

    # Derived column: PM2.5/PM10 ratio (source apportionment indicator)
    if "pm25_mean" in df.columns and "pm10_mean" in df.columns:
        df["pm_ratio"] = df["pm25_mean"] / df["pm10_mean"]

    missing_pct = df.isnull().mean() * 100
    cols_with_missing = missing_pct[missing_pct > 0]
    if not cols_with_missing.empty:
        warnings.warn(
            "Columns with missing values:\n"
            + cols_with_missing.round(1).to_string(),
            stacklevel=2,
        )

    return df


def compute_annual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly observations to annual summaries.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly dataframe returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        One row per calendar year with mean pollutant concentrations,
        AQI range, and first-value socioeconomic variables.
    """
    annual = df.groupby("year").agg(
        pm25_mean        = ("pm25_mean",        "mean"),
        pm10_mean        = ("pm10_mean",        "mean"),
        no2_mean         = ("no2_mean",         "mean"),
        so2_mean         = ("so2_mean",         "mean"),
        aqi_mean         = ("aqi_mean",         "mean"),
        aqi_min          = ("aqi_min",          "min"),
        aqi_max          = ("aqi_max",          "max"),
        population_total = ("population_total", "first"),
        urban_share_pct  = ("urban_share_pct",  "first"),
        hdi              = ("hdi",              "first"),
        poverty_rate_pct = ("poverty_rate_pct", "first"),
    ).reset_index()

    # Population-normalised metrics
    pop_m = annual["population_total"] / 1e6
    annual["aqi_per_1m"]  = annual["aqi_mean"]  / pop_m
    annual["pm25_per_1m"] = annual["pm25_mean"] / pop_m

    return annual


def compute_monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-year monthly climatology (mean ± SD for each calendar month).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        12-row dataframe indexed by calendar month (1–12).
    """
    clim = df.groupby("month").agg(
        pm25    = ("pm25_mean", "mean"),
        pm10    = ("pm10_mean", "mean"),
        no2     = ("no2_mean",  "mean"),
        so2     = ("so2_mean",  "mean"),
        aqi     = ("aqi_mean",  "mean"),
        pm25_sd = ("pm25_mean", "std"),
        pm10_sd = ("pm10_mean", "std"),
        no2_sd  = ("no2_mean",  "std"),
        so2_sd  = ("so2_mean",  "std"),
        aqi_sd  = ("aqi_mean",  "std"),
    ).reset_index()
    return clim


def covid_periods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into pre-COVID, lockdown, and post-COVID sub-sets.

    Lockdown period: March – August 2020 (Government of Bangladesh).

    Returns
    -------
    tuple
        (pre_df, lockdown_df, post_df)
    """
    pre  = df[df["month_start"] < "2020-03-01"].copy()
    lock = df[
        (df["month_start"] >= "2020-03-01") &
        (df["month_start"] <= "2020-08-31")
    ].copy()
    post = df[df["month_start"] > "2020-08-31"].copy()
    return pre, lock, post


def build_future_dates(df: pd.DataFrame, end: str = "2030-12-01") -> pd.DataFrame:
    """
    Create a dataframe of future monthly dates from one month after the
    last observation through *end*.

    Parameters
    ----------
    df  : observed dataframe
    end : ISO date string for final forecast month

    Returns
    -------
    pd.DataFrame
        Columns: ``month_start``, ``year``, ``month``
    """
    last_date    = df["month_start"].max()
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        end=end,
        freq="MS",
    )
    return pd.DataFrame({
        "month_start": future_dates,
        "year":        future_dates.year,
        "month":       future_dates.month,
    })
