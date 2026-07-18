"""Statistical methods for the audited monthly PM2.5 series."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def seasonal_mann_kendall(values: pd.Series, months: pd.Series) -> dict[str, float]:
    """Seasonal Mann-Kendall test with within-season tie correction."""
    frame = pd.DataFrame({"value": values, "month": months}).dropna()
    total_s = 0.0
    total_var = 0.0
    for _, group in frame.groupby("month"):
        x = group["value"].to_numpy(dtype=float)
        n = len(x)
        if n < 2:
            continue
        total_s += sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
        _, tie_counts = np.unique(x, return_counts=True)
        tie_term = sum(count * (count - 1) * (2 * count + 5) for count in tie_counts if count > 1)
        total_var += (n * (n - 1) * (2 * n + 5) - tie_term) / 18
    if total_var <= 0:
        return {"s": total_s, "variance": total_var, "z": 0.0, "p_value": 1.0}
    z = (total_s - 1) / math.sqrt(total_var) if total_s > 0 else (total_s + 1) / math.sqrt(total_var)
    return {
        "s": float(total_s),
        "variance": float(total_var),
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def seasonal_sen_slope(
    values: pd.Series,
    dates: pd.Series,
    seed: int = 20260717,
    bootstrap_samples: int = 2000,
) -> dict[str, float]:
    """Median same-calendar-month slope and year-block bootstrap interval."""
    frame = pd.DataFrame({"value": values, "date": pd.to_datetime(dates)}).dropna()
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month

    def slope(sample: pd.DataFrame) -> float:
        slopes: list[float] = []
        for _, group in sample.sort_values("date").groupby("month"):
            rows = list(group.itertuples(index=False))
            for first, second in itertools.combinations(rows, 2):
                elapsed_years = second.year - first.year
                if elapsed_years > 0:
                    slopes.append((second.value - first.value) / elapsed_years)
        return float(np.median(slopes)) if slopes else float("nan")

    estimate = slope(frame)
    years = np.array(sorted(frame["year"].unique()))
    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        pieces = []
        for synthetic_year, source_year in enumerate(sampled_years):
            piece = frame[frame["year"] == source_year].copy()
            piece["year"] = synthetic_year
            pieces.append(piece)
        current = slope(pd.concat(pieces, ignore_index=True))
        if np.isfinite(current):
            bootstrap.append(current)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {"slope_ug_m3_per_year": estimate, "ci_low": float(low), "ci_high": float(high)}


def trend_free_prewhitened_smk(values: pd.Series, dates: pd.Series) -> dict[str, float]:
    """Trend-free prewhitening sensitivity followed by seasonal MK."""
    frame = pd.DataFrame({"value": values, "date": pd.to_datetime(dates)}).dropna().sort_values("date")
    elapsed = (frame["date"] - frame["date"].min()).dt.days.to_numpy() / 365.2425
    slope = stats.theilslopes(frame["value"].to_numpy(), elapsed).slope
    detrended = frame["value"].to_numpy() - slope * elapsed
    rho = float(pd.Series(detrended).autocorr(lag=1))
    if not np.isfinite(rho):
        rho = 0.0
    whitened = detrended[1:] - rho * detrended[:-1] + slope * elapsed[1:]
    result = seasonal_mann_kendall(pd.Series(whitened), frame["date"].dt.month.iloc[1:].reset_index(drop=True))
    result["lag1_autocorrelation"] = rho
    result["theil_slope_for_detrending"] = float(slope)
    return result


def month_adjusted_hac_regression(values: pd.Series, dates: pd.Series) -> dict[str, float]:
    """Elapsed-time OLS with calendar-month effects and Newey-West covariance."""
    frame = pd.DataFrame({"value": values, "date": pd.to_datetime(dates)}).dropna().sort_values("date")
    frame["elapsed_years"] = (frame["date"] - frame["date"].min()).dt.days / 365.2425
    month_dummies = pd.get_dummies(frame["date"].dt.month, prefix="month", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([frame[["elapsed_years"]].reset_index(drop=True), month_dummies.reset_index(drop=True)], axis=1))
    fit = sm.OLS(frame["value"].to_numpy(dtype=float), design).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    ci = fit.conf_int().loc["elapsed_years"]
    return {
        "slope_ug_m3_per_year": float(fit.params["elapsed_years"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(fit.pvalues["elapsed_years"]),
        "r_squared": float(fit.rsquared),
        "n": int(fit.nobs),
    }


def pettitt_test(values: pd.Series, dates: pd.Series) -> dict[str, object]:
    """Pettitt single-change test as a source-break diagnostic."""
    frame = pd.DataFrame({"value": values, "date": pd.to_datetime(dates)}).dropna().sort_values("date")
    ranks = stats.rankdata(frame["value"])
    n = len(ranks)
    u = 2 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    index = int(np.argmax(np.abs(u)))
    statistic = float(abs(u[index]))
    p_value = min(1.0, 2 * math.exp((-6 * statistic**2) / (n**3 + n**2)))
    return {"change_month": frame.iloc[index]["date"].date().isoformat(), "statistic": statistic, "p_value": p_value}


def seasonal_tests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis plus Holm-corrected pairwise Mann-Whitney comparisons."""
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Pre-monsoon",
        4: "Pre-monsoon",
        5: "Pre-monsoon",
        6: "Monsoon",
        7: "Monsoon",
        8: "Monsoon",
        9: "Monsoon",
        10: "Post-monsoon",
        11: "Post-monsoon",
    }
    work = frame.copy()
    work["season"] = pd.to_datetime(work["month_start"]).dt.month.map(season_map)
    groups = [group["pm25_mean"].dropna().to_numpy() for _, group in work.groupby("season")]
    h_stat, p_value = stats.kruskal(*groups)
    n = sum(len(group) for group in groups)
    k = len(groups)
    epsilon_sq = max(0.0, (h_stat - k + 1) / (n - k))
    overall = pd.DataFrame([{"test": "Kruskal-Wallis", "h_statistic": h_stat, "p_value": p_value, "epsilon_squared": epsilon_sq, "n": n}])

    pairs = []
    seasons = sorted(work["season"].dropna().unique())
    for first, second in itertools.combinations(seasons, 2):
        x = work.loc[work["season"] == first, "pm25_mean"].dropna()
        y = work.loc[work["season"] == second, "pm25_mean"].dropna()
        statistic, raw_p = stats.mannwhitneyu(x, y, alternative="two-sided")
        effect = 2 * statistic / (len(x) * len(y)) - 1
        pairs.append({"season_1": first, "season_2": second, "u_statistic": statistic, "p_raw": raw_p, "rank_biserial": effect})
    order = np.argsort([row["p_raw"] for row in pairs])
    adjusted = np.empty(len(pairs))
    running = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, pairs[index]["p_raw"] * (len(pairs) - rank))
        running = max(running, current)
        adjusted[index] = running
    for row, value in zip(pairs, adjusted, strict=True):
        row["p_holm"] = value
    return overall, pd.DataFrame(pairs)


def covid_comparison(monthly: pd.DataFrame) -> pd.DataFrame:
    """Season-matched contrasts and month-adjusted interrupted regression."""
    frame = monthly.copy()
    frame["date"] = pd.to_datetime(frame["month_start"])
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    rows: list[dict[str, object]] = []
    for label, months in {"March-August": range(3, 9), "April-June sensitivity": range(4, 7)}.items():
        baseline = frame[(frame["year"] == 2019) & frame["month"].isin(months)]["pm25_mean"]
        covid = frame[(frame["year"] == 2020) & frame["month"].isin(months)]["pm25_mean"]
        difference = covid.mean() - baseline.mean()
        low, high = stats.bootstrap(
            (covid.to_numpy(), baseline.to_numpy()),
            lambda x, y: np.mean(x) - np.mean(y),
            paired=False,
            method="percentile",
            n_resamples=5000,
            rng=np.random.default_rng(20260717),
        ).confidence_interval
        rows.append({"analysis": f"season-matched {label}", "estimate_ug_m3": difference, "ci_low": low, "ci_high": high, "p_value": stats.mannwhitneyu(covid, baseline).pvalue, "interpretation": "unadjusted association"})

    frame["elapsed"] = np.arange(len(frame), dtype=float)
    frame["lockdown"] = ((frame["year"] == 2020) & frame["month"].between(3, 8)).astype(float)
    design = pd.concat(
        [frame[["elapsed", "lockdown"]], pd.get_dummies(frame["month"], prefix="month", drop_first=True, dtype=float)],
        axis=1,
    )
    fit = sm.OLS(frame["pm25_mean"], sm.add_constant(design)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    ci = fit.conf_int().loc["lockdown"]
    rows.append({"analysis": "interrupted time series March-August", "estimate_ug_m3": fit.params["lockdown"], "ci_low": ci.iloc[0], "ci_high": ci.iloc[1], "p_value": fit.pvalues["lockdown"], "interpretation": "month-adjusted association; meteorology unavailable"})
    return pd.DataFrame(rows)
