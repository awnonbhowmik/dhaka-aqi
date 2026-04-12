"""
viz.py
------
Seaborn-first visualisation functions for the Dhaka AQI study.

All public functions
  1. create and return a (fig, axes) pair
  2. optionally save to figures/<name>.png at 300 DPI via save_fig()

Matplotlib is used for features seaborn does not support natively
(time-series with fill_between, twin-axis, polar/radar, stackplot).
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.stats import spearmanr

from .config import (
    AQI_CATS, EPA, MONTH_LABELS, PAL, POL_COLS, POL_LATEX, POL_NAMES,
    POL_SHORT, POL_SHORT_LATEX, SEASON_ORDER, SEASON_PAL, WHO,
)

FIGURES_DIR = pathlib.Path("figures")


# ── Utility ────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str, dpi: int = 300) -> pathlib.Path:
    """
    Save *fig* to ``figures/<name>.png`` at *dpi* DPI.
    Creates the figures/ directory if it does not exist.
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {out}")
    return out


# ── §2 — Distributions & Data Quality ─────────────────────────────────────────

def plot_distributions(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 1. Histograms (seaborn) + KDE for each pollutant/AQI, plus
    a normalised Z-score box plot for visual comparison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, col, label, color in zip(axes[:5], POL_COLS, POL_LATEX,
                                     [PAL["pm25"], PAL["pm10"], PAL["no2"],
                                      PAL["so2"], PAL["aqi"]]):
        data = df[col].dropna()
        sns.histplot(data, bins=18, color=color, alpha=0.75, edgecolor="white",
                     linewidth=0.6, kde=True, ax=ax, line_kws={"lw": 1.8})
        ax.axvline(data.mean(),   color="#333", lw=2.0, ls="--",
                   label=f"Mean {data.mean():.0f}")
        ax.axvline(data.median(), color="#999", lw=1.8, ls=":",
                   label=f"Median {data.median():.0f}")
        who_key = col.replace("_mean", "")
        if who_key in WHO:
            ax.axvline(WHO[who_key], color="red", lw=1.5, ls="-.", alpha=0.85,
                       label=f"WHO {WHO[who_key]}")
        ax.set_xlabel(label); ax.set_ylabel("Frequency")
        ax.legend(fontsize=9, handlelength=1.5)

    # Normalised Z-score boxplot
    ax6 = axes[5]
    norm_data = [
        (df[c].dropna() - df[c].mean()) / df[c].std() for c in POL_COLS
    ]
    norm_df = pd.DataFrame(
        {s: d.values[:min(len(d) for d in norm_data)] for s, d in zip(POL_SHORT_LATEX, norm_data)}
    ).melt(var_name="Pollutant", value_name="Z-score")
    palette  = dict(zip(POL_SHORT_LATEX, [PAL["pm25"], PAL["pm10"], PAL["no2"], PAL["so2"], PAL["aqi"]]))
    sns.boxplot(data=norm_df, x="Pollutant", y="Z-score",
                palette=palette, width=0.55, ax=ax6, flierprops={"marker": "o", "markersize": 4})
    ax6.axhline(0, color="#333", lw=1.2, ls="--", alpha=0.6)
    ax6.set_title("Normalised Overview (Z-scores)")

    plt.tight_layout()
    return fig


def plot_cv_analysis(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 2. Coefficient of Variation (CV = σ/μ × 100 %) by year and season.
    """
    cv_cols = ["pm25_mean", "no2_mean", "so2_mean", "aqi_mean"]
    cv_year   = df.groupby("year")[cv_cols].apply(lambda g: (g.std() / g.mean() * 100).round(1))
    cv_season = df.groupby("season", observed=True)[cv_cols].apply(
        lambda g: (g.std() / g.mean() * 100).round(1)
    ).reindex(SEASON_ORDER)

    colors4 = [PAL["pm25"], PAL["no2"], PAL["so2"], PAL["aqi"]]
    labels4 = ["PM₂.₅", "NO₂", "SO₂", "AQI"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Year plot
    for col, label, color in zip(cv_cols, labels4, colors4):
        axes[0].plot(cv_year.index, cv_year[col], "o-", color=color, lw=1.8, ms=5, label=label)
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("CV (%)"); axes[0].set_title("Inter-annual Variability (CV)")
    axes[0].set_xticks(cv_year.index); axes[0].legend(fontsize=8)

    # Season bar
    cv_season_melted = cv_season.reset_index().melt(id_vars="season", var_name="Pollutant", value_name="CV")
    cv_season_melted["Pollutant"] = cv_season_melted["Pollutant"].map(
        dict(zip(cv_cols, labels4))
    )
    sns.barplot(data=cv_season_melted, x="season", y="CV", hue="Pollutant",
                palette=dict(zip(labels4, colors4)), alpha=0.8, ax=axes[1])
    axes[1].set_xlabel("Season"); axes[1].set_ylabel("CV (%)")
    axes[1].set_title("Intra-seasonal Variability (CV)")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    return fig


# ── §3 — Temporal Trends ───────────────────────────────────────────────────────

def plot_monthly_time_series(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 3. Monthly mean pollutant and AQI time series with rolling mean
    and COVID lockdown shading.
    """
    series_cfg = [
        ("aqi_mean",  "aqi_min",  "aqi_max",  "AQI",                                    PAL["aqi"],  None),
        ("pm25_mean", "pm25_min", "pm25_max", r"$\mathrm{PM}_{2.5}$ ($\mu$g m$^{-3}$)", PAL["pm25"], WHO["pm25"]),
        ("pm10_mean", "pm10_min", "pm10_max", r"$\mathrm{PM}_{10}$ ($\mu$g m$^{-3}$)",  PAL["pm10"], WHO["pm10"]),
        ("no2_mean",  "no2_min",  "no2_max",  r"$\mathrm{NO}_2$ ($\mu$g m$^{-3}$)",     PAL["no2"],  WHO["no2"]),
        ("so2_mean",  "so2_min",  "so2_max",  r"$\mathrm{SO}_2$ ($\mu$g m$^{-3}$)",     PAL["so2"],  WHO["so2"]),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    for ax, (mc, mnc, mxc, ylabel, color, who) in zip(axes, series_cfg):
        x = df["month_start"]
        ax.fill_between(x, df[mnc], df[mxc], alpha=0.18, color=color)
        ax.plot(x, df[mc], color=color, lw=1.5, label="Monthly mean")
        roll = df[mc].rolling(12, center=True).mean()
        ax.plot(x, roll, color="#333", lw=1.8, ls="--", alpha=0.8, label="12-mo rolling mean")
        if who:
            ax.axhline(who, color="red", lw=1, ls="-.", alpha=0.7,
                       label=f"WHO guideline ({who} " + r"$\mu$g m$^{-3}$)")
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-31"),
                   alpha=0.12, color="#DAA520", label="COVID-19 lockdown")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=4, framealpha=0.85)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    return fig


def plot_annual_trends(df: pd.DataFrame, annual: pd.DataFrame) -> plt.Figure:
    """
    Figure 4. Annual mean pollutant concentrations and AQI with OLS trend lines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    ax = axes[0]
    for col, label, color in [
        ("pm25_mean", r"$\mathrm{PM}_{2.5}$", PAL["pm25"]),
        ("pm10_mean", r"$\mathrm{PM}_{10}$",  PAL["pm10"]),
        ("no2_mean",  r"$\mathrm{NO}_2$",     PAL["no2"]),
        ("so2_mean",  r"$\mathrm{SO}_2$",     PAL["so2"]),
    ]:
        ax.plot(annual["year"], annual[col], "o-", color=color, lw=1.8, ms=5, label=label)
        slope, intercept, *_ = stats.linregress(annual["year"], annual[col])
        x_line = np.array([annual["year"].min(), annual["year"].max()])
        ax.plot(x_line, slope * x_line + intercept, "--", color=color, lw=1, alpha=0.55)
    ax.set_xlabel("Year"); ax.set_ylabel(r"Concentration ($\mu$g m$^{-3}$)")
    ax.set_title("Annual Mean Pollutant Concentrations")
    ax.legend(fontsize=9); ax.set_xticks(annual["year"])

    ax2 = axes[1]
    bar_colors = [PAL["aqi"] if yr != 2020 else "#DAA520" for yr in annual["year"]]
    ax2.bar(annual["year"], annual["aqi_mean"], color=bar_colors, alpha=0.75, width=0.65, zorder=3)
    yoy = annual["aqi_mean"].pct_change() * 100
    for yr, val, pct in zip(annual["year"], annual["aqi_mean"], yoy):
        if pd.notna(pct):
            c = "#C0392B" if pct > 0 else "#27AE60"
            ax2.text(yr, val + 1.5, f"{pct:+.1f}%", ha="center", va="bottom",
                     fontsize=7.5, color=c, fontweight="bold")
    slope, intercept, *_ = stats.linregress(annual["year"], annual["aqi_mean"])
    x_line = np.linspace(annual["year"].min() - 0.5, annual["year"].max() + 0.5, 100)
    ax2.plot(x_line, slope * x_line + intercept, "k--", lw=1.5, alpha=0.7,
             label=f"Trend: {slope:+.1f} AQI/yr")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Mean AQI")
    ax2.set_title("Annual Mean AQI (% YoY change; gold = COVID year)")
    ax2.legend(fontsize=9); ax2.set_xticks(annual["year"])
    ax2.set_xlim(annual["year"].min() - 0.6, annual["year"].max() + 0.6)
    plt.tight_layout()
    return fig


# ── §4 — Seasonality ──────────────────────────────────────────────────────────

def plot_monthly_climatology(df: pd.DataFrame, clim: pd.DataFrame) -> plt.Figure:
    """
    Figure 5. Multi-year monthly climatology of pollutant concentrations and AQI.
    """
    season_spans = [
        (0.5,  2.5,  "Winter",       SEASON_PAL["Winter"]),
        (2.5,  5.5,  "Pre-monsoon",  SEASON_PAL["Pre-monsoon"]),
        (5.5,  9.5,  "Monsoon",      SEASON_PAL["Monsoon"]),
        (9.5,  11.5, "Post-monsoon", SEASON_PAL["Post-monsoon"]),
        (11.5, 12.5, "",             SEASON_PAL["Winter"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    ax = axes[0]
    for col, label, color in [
        ("pm25", "PM₂.₅", PAL["pm25"]), ("pm10", "PM₁₀", PAL["pm10"]),
        ("no2",  "NO₂",   PAL["no2"]),  ("so2",  "SO₂",  PAL["so2"]),
    ]:
        ax.plot(clim["month"], clim[col], "o-", color=color, lw=2, ms=5, label=label)
    for s, e, _, c in season_spans:
        ax.axvspan(s, e, alpha=0.07, color=c)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Mean Concentration (µg/m³)"); ax.set_title("All Pollutants")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    season_mo = ["Winter","Winter","Pre-monsoon","Pre-monsoon","Pre-monsoon",
                 "Monsoon","Monsoon","Monsoon","Monsoon","Post-monsoon","Post-monsoon","Winter"]
    bcolors = [SEASON_PAL[s] for s in season_mo]
    ax2.bar(clim["month"], clim["aqi"], color=bcolors, alpha=0.75, width=0.7, zorder=3)
    ax2.errorbar(clim["month"], clim["aqi"], yerr=clim["aqi_sd"],
                 fmt="none", color="#333", capsize=3, lw=1, zorder=4)
    for s, e, _, c in season_spans:
        ax2.axvspan(s, e, alpha=0.07, color=c)
    ax2.set_xticks(range(1, 13)); ax2.set_xticklabels(MONTH_LABELS)
    ax2.set_ylabel("Mean AQI"); ax2.set_title("Monthly Mean AQI (±1 SD)")
    legend_els = [mpatches.Patch(facecolor=c, label=s, alpha=0.75)
                  for s, c in SEASON_PAL.items()]
    ax2.legend(handles=legend_els, fontsize=8)
    plt.tight_layout()
    return fig


def plot_aqi_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 6. Year × month heat map of mean AQI (seaborn heatmap).
    """
    pivot = df.pivot_table(index="year", columns="month", values="aqi_mean")
    pivot.columns = MONTH_LABELS

    fig, ax = plt.subplots(figsize=(18, 7))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0, linecolor="none", ax=ax, annot_kws={"size": 12},
                cbar_kws={"label": "Mean AQI", "shrink": 0.85})
    ax.set_xlabel("Month", fontsize=13, labelpad=8)
    ax.set_ylabel("Year", fontsize=13, labelpad=8)
    ax.tick_params(axis="both", labelsize=12)
    ax.collections[0].colorbar.ax.tick_params(labelsize=11)
    ax.collections[0].colorbar.set_label("Mean AQI", fontsize=12)
    for text in ax.texts:
        text.set_fontsize(12)
    # Remove all grid lines from the underlying axes
    ax.grid(False)
    plt.tight_layout()
    return fig


def plot_seasonal_distributions(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 7. Seasonal distribution of all pollutants + AQI (seaborn boxplots).
    Includes a radar (spider) chart and grouped-bar comparison.
    """
    pol_season = list(zip(POL_COLS, POL_LATEX))
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    # Panels 1–5: seaborn boxplots
    for ax, (col, label) in zip(axes[:5], pol_season):
        sns.boxplot(
            data=df, x="season", y=col, order=SEASON_ORDER,
            palette=SEASON_PAL, width=0.55, ax=ax,
            medianprops={"color": "#333", "lw": 2},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
        )
        ax.set_xticklabels(["Winter", "Pre-Mon.", "Monsoon", "Post-Mon."],
                           fontsize=11, rotation=30, ha="right")
        ax.set_title(label, fontsize=13, pad=8)
        ax.set_xlabel(""); ax.set_ylabel(label, fontsize=12)

    # Panel 6: radar chart
    pol_cols_radar   = ["pm25_mean", "pm10_mean", "no2_mean", "so2_mean", "aqi_mean"]
    pol_labels_radar = POL_SHORT_LATEX
    N      = len(pol_cols_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    ax6 = fig.add_subplot(3, 3, 6, polar=True)
    axes[5].set_visible(False)
    season_means = {s: df[df["season"] == s][pol_cols_radar].mean().values for s in SEASON_ORDER}
    all_vals  = np.array(list(season_means.values()))
    col_min   = all_vals.min(axis=0); col_max = all_vals.max(axis=0)
    for s in SEASON_ORDER:
        raw    = season_means[s]
        norm   = ((raw - col_min) / (col_max - col_min + 1e-9)).tolist()
        values = norm + norm[:1]
        ax6.plot(angles, values, lw=2, color=SEASON_PAL[s], label=s)
        ax6.fill(angles, values, alpha=0.15, color=SEASON_PAL[s])
    ax6.set_xticks(angles[:-1]); ax6.set_xticklabels(pol_labels_radar, fontsize=11)
    ax6.set_yticklabels([]); ax6.set_title("Seasonal Profiles\n(normalized)", fontsize=12, pad=18)
    ax6.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10, framealpha=0.85)

    # Panel 7: grouped bar
    ax7 = axes[6]
    pol_labels_short = POL_SHORT_LATEX
    x = np.arange(len(pol_labels_short)); w = 0.2
    for k, s in enumerate(SEASON_ORDER):
        vals = df[df["season"] == s][pol_cols_radar].mean().values
        ax7.bar(x + k * w, vals, w, label=s, color=SEASON_PAL[s], alpha=0.8)
    ax7.set_xticks(x + w * 1.5); ax7.set_xticklabels(pol_labels_short, fontsize=11)
    ax7.set_ylabel("Mean Concentration / AQI"); ax7.set_title("Seasonal Mean Comparison")
    ax7.legend(fontsize=9)

    # Panel 8: year-over-year overlay
    ax8 = axes[7]
    cmap = plt.cm.get_cmap("coolwarm_r", df["year"].nunique())
    for i, (yr, grp) in enumerate(df.groupby("year")):
        ax8.plot(grp["month"], grp["aqi_mean"], "o-", color=cmap(i),
                 lw=2.2 if yr in (df["year"].min(), df["year"].max()) else 1.2,
                 ls="-" if yr != 2020 else "--", ms=4, label=str(yr), alpha=0.85)
    ax8.set_xticks(range(1, 13)); ax8.set_xticklabels(MONTH_LABELS)
    ax8.set_ylabel("Monthly Mean AQI"); ax8.set_title("Year-over-Year AQI Overlay")
    ax8.legend(title="Year", fontsize=7, ncol=3)

    # Panel 9: AQI category distribution
    ax9 = axes[8]
    cat_colors = {c: col for _, _, c, col in AQI_CATS}
    cat_order  = [c for _, _, c, _ in AQI_CATS]
    df_cat = df.copy()
    from .analysis import classify_aqi
    df_cat["AQI_Cat"] = df_cat["aqi_mean"].apply(classify_aqi)
    counts = df_cat["AQI_Cat"].value_counts().reindex(cat_order, fill_value=0)
    ax9.bar(counts.index, counts.values,
            color=[cat_colors[c] for c in counts.index], alpha=0.85, edgecolor="white")
    ax9.set_xlabel("AQI Category"); ax9.set_ylabel("Number of Months")
    ax9.set_title("AQI Category Distribution (US EPA)")
    ax9.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    return fig


# ── §5 — Time-Series Diagnostics ──────────────────────────────────────────────

def plot_stl_decomposition(df: pd.DataFrame, target: str = "pm25_mean") -> plt.Figure:
    """
    Figure 8. Additive STL decomposition of monthly PM₂.₅.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    label = dict(zip(POL_COLS, POL_NAMES)).get(target, target)
    series = df.set_index("month_start")[target]
    decomp = seasonal_decompose(series, model="additive", period=12,
                                extrapolate_trend="freq")
    components = [
        (series,          "Observed",  PAL.get(target.replace("_mean", ""), "#333")),
        (decomp.trend,    "Trend",     "#333"),
        (decomp.seasonal, "Seasonal",  "#E67E22"),
        (decomp.resid,    "Residual",  "#888"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax, (data, comp_label, color) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, lw=1.8)
        if comp_label == "Residual":
            ax.axhline(0, color="#333", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel(comp_label)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    return fig


def plot_anomaly_detection(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 9. Z-score anomaly detection for PM₂.₅ (|Z| > 2 flagged).
    """
    anom_df = df[["month_start", "pm25_mean"]].copy()
    mu, sigma = anom_df["pm25_mean"].mean(), anom_df["pm25_mean"].std()
    anom_df["z"] = (anom_df["pm25_mean"] - mu) / sigma
    normal = anom_df[anom_df["z"].abs() <= 2]
    anom   = anom_df[anom_df["z"].abs() > 2]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(anom_df["month_start"], anom_df["pm25_mean"],
            color=PAL["pm25"], lw=1.5, alpha=0.8)
    sns.scatterplot(data=normal, x="month_start", y="pm25_mean",
                    color=PAL["pm25"], s=25, alpha=0.5, ax=ax, zorder=4)
    sns.scatterplot(data=anom, x="month_start", y="pm25_mean",
                    color="red", s=80, marker="D", ax=ax, zorder=5, label="Anomaly (|Z|>2)")
    for _, row in anom.iterrows():
        ax.annotate(row["month_start"].strftime("%b %Y"),
                    (row["month_start"], row["pm25_mean"]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=7, color="red", ha="center")
    ax.set_ylabel("PM₂.₅ (µg/m³)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_pm_ratio(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 10. PM₂.₅/PM₁₀ ratio time series and seasonal boxplot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    ax = axes[0]
    ax.plot(df["month_start"], df["pm_ratio"], color="#7D3C98", lw=1.8)
    roll = df["pm_ratio"].rolling(12, center=True).mean()
    ax.plot(df["month_start"], roll, color="#333", lw=2, ls="--", label="12-mo rolling mean")
    ax.axhline(0.6, color="#C0392B", lw=1, ls="-.", label="Combustion threshold (0.6)")
    ax.axhline(0.4, color="#E67E22", lw=1, ls=":",  label="Dust threshold (0.4)")
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-31"),
               alpha=0.12, color="#DAA520", label="COVID-19 lockdown")
    ax.set_ylabel("PM₂.₅ / PM₁₀"); ax.set_title("Monthly PM₂.₅/PM₁₀ Ratio")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8)

    sns.boxplot(data=df, x="season", y="pm_ratio", order=SEASON_ORDER,
                palette=SEASON_PAL, width=0.55, ax=axes[1])
    axes[1].set_xticklabels(["Winter", "Pre-Mon.", "Monsoon", "Post-Mon."],
                             rotation=30, ha="right")
    axes[1].axhline(0.6, color="#C0392B", lw=1, ls="-.", alpha=0.7)
    axes[1].axhline(0.4, color="#E67E22", lw=1, ls=":",  alpha=0.7)
    axes[1].set_ylabel("PM₂.₅ / PM₁₀")
    axes[1].set_title("PM₂.₅/PM₁₀ Ratio by Season")
    axes[1].set_xlabel("")
    plt.tight_layout()
    return fig


# ── §6 — Correlations & Inter-relationships ───────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 11. Full-square Spearman correlation matrix (seaborn heatmap).
    Diagonal cells are blanked; significance stars are annotated on each cell.
    """
    corr_cols  = ["pm25_mean","pm10_mean","no2_mean","so2_mean","aqi_mean",
                   "norm_rain","hdi","urban_share_pct","poverty_rate_pct","month","year"]
    corr_names = [r"$\mathrm{PM}_{2.5}$", r"$\mathrm{PM}_{10}$",
                   r"$\mathrm{NO}_2$", r"$\mathrm{SO}_2$", "AQI",
                   "Rainfall", "HDI", "Urban %", "Poverty %", "Month", "Year"]

    spearman_mat = df[corr_cols].corr(method="spearman")
    spearman_mat.columns = corr_names
    spearman_mat.index   = corr_names

    # Build annotation matrix (numeric ρ only)
    n = len(corr_cols)
    annot = np.empty((n, n), dtype=object)
    for i, ci in enumerate(corr_cols):
        for j, cj in enumerate(corr_cols):
            if i == j:
                annot[i, j] = "—"
            else:
                r = spearman_mat.iloc[i, j]
                annot[i, j] = f"{r:.2f}"

    # Mask only the diagonal (leave full matrix visible)
    diag_mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        spearman_mat, mask=diag_mask, annot=annot, fmt="",
        annot_kws={"size": 9}, cmap="RdYlBu_r",
        vmin=-1, vmax=1, center=0,
        linewidths=0.4, linecolor="#dddddd",
        ax=ax, cbar_kws={"shrink": 0.75, "label": "Spearman ρ"},
    )
    # Paint diagonal cells grey
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                   color="#eeeeee", zorder=3))
        ax.text(i + 0.5, i + 0.5, "—", ha="center", va="center",
                fontsize=9, color="#aaaaaa", zorder=4)

    ax.tick_params(axis="x", labelsize=11, rotation=45)
    ax.tick_params(axis="y", labelsize=11, rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    ax.collections[0].colorbar.set_label("Spearman ρ", fontsize=11)
    ax.grid(False)
    plt.tight_layout()
    return fig


def plot_pairwise_scatter(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 12. Pairwise scatter plots coloured by season with OLS fit.
    """
    _pm25l = r"$\mathrm{PM}_{2.5}$ ($\mu$g m$^{-3}$)"
    _pm10l = r"$\mathrm{PM}_{10}$ ($\mu$g m$^{-3}$)"
    _no2l  = r"$\mathrm{NO}_2$ ($\mu$g m$^{-3}$)"
    _so2l  = r"$\mathrm{SO}_2$ ($\mu$g m$^{-3}$)"
    pairs = [
        ("pm25_mean", "no2_mean",  _pm25l, _no2l),
        ("pm25_mean", "so2_mean",  _pm25l, _so2l),
        ("pm10_mean", "pm25_mean", _pm10l, _pm25l),
        ("no2_mean",  "so2_mean",  _no2l,  _so2l),
        ("pm25_mean", "aqi_mean",  _pm25l, "AQI"),
        ("pm10_mean", "aqi_mean",  _pm10l, "AQI"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (xc, yc, xl, yl) in zip(axes.flat, pairs):
        sns.scatterplot(data=df, x=xc, y=yc, hue="season", hue_order=SEASON_ORDER,
                        palette=SEASON_PAL, s=40, alpha=0.8,
                        edgecolors="none", ax=ax, legend=(ax is axes[0, 0]))
        slope, intercept, r, p, _ = stats.linregress(df[xc], df[yc])
        x_fit = np.linspace(df[xc].min(), df[xc].max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "k--", lw=1.5, alpha=0.6)
        r_sp, _ = spearmanr(df[xc], df[yc])
        p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
        ax.set_title(f"Pearson r={r:.2f} ({p_str})\nSpearman ρ={r_sp:.2f}", fontsize=8.5)
    if axes[0, 0].get_legend():
        axes[0, 0].legend(fontsize=7, ncol=2, title="Season")
    plt.tight_layout()
    return fig


def plot_source_apportionment(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 13. Stacked-area relative contribution of each pollutant.
    """
    pol4  = ["pm25_mean","pm10_mean","no2_mean","so2_mean"]
    total = df[pol4].sum(axis=1)
    pct   = df[pol4].div(total, axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.stackplot(df["month_start"], [pct[c] for c in pol4],
                 labels=["PM₂.₅","PM₁₀","NO₂","SO₂"],
                 colors=[PAL["pm25"],PAL["pm10"],PAL["no2"],PAL["so2"]], alpha=0.82)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-31"),
               alpha=0.15, color="#DAA520", label="COVID-19 lockdown")
    ax.set_ylabel("Share of Total Pollutant Load (%)")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    plt.tight_layout()
    return fig


# ── §8 — COVID Impact ─────────────────────────────────────────────────────────

def plot_covid_impact(df: pd.DataFrame,
                      pre:  pd.DataFrame,
                      lock: pd.DataFrame,
                      post: pd.DataFrame) -> plt.Figure:
    """
    Figure 14. COVID-19 impact: period comparison, lockdown % change, 2019 vs 2020.
    """
    period_means = pd.DataFrame({
        "Pre-COVID\n(Jan 2017–Feb 2020)":     pre[POL_COLS].mean(),
        "Lockdown\n(Mar–Aug 2020)":           lock[POL_COLS].mean(),
        "Post-COVID\n(Sep 2020–Sep 2025)":    post[POL_COLS].mean(),
    }).T
    period_means.columns = POL_SHORT

    pct_lock = (
        (period_means.iloc[1] - period_means.iloc[0]) / period_means.iloc[0] * 100
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # (a) grouped bar
    ax = axes[0]
    period_colors = ["#2980B9","#C0392B","#27AE60"]
    x = np.arange(len(POL_SHORT)); w = 0.25
    for i, (period, row) in enumerate(period_means.iterrows()):
        ax.bar(x + i * w, row.values, w, label=period.replace("\n", " "),
               color=period_colors[i], alpha=0.78)
    ax.set_xticks(x + w); ax.set_xticklabels(POL_SHORT, fontsize=9)
    ax.set_ylabel("Mean Concentration / AQI")
    ax.set_title("(a) Period Comparison")
    ax.legend(fontsize=7)

    # (b) % change seaborn barplot
    ax2 = axes[1]
    pal_map = dict(zip(POL_SHORT, [PAL["pm25"],PAL["pm10"],PAL["no2"],PAL["so2"],PAL["aqi"]]))
    pct_df  = pd.DataFrame({"Pollutant": pct_lock.index, "% Change": pct_lock.values})
    sns.barplot(data=pct_df, x="Pollutant", y="% Change",
                palette=pal_map, alpha=0.85, ax=ax2)
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_ylabel("% Change vs Pre-COVID")
    ax2.set_title("(b) Lockdown Impact (%)")
    for bar, val in zip(ax2.patches, pct_lock.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + (1 if val >= 0 else -3.5),
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                 color="#C0392B" if val > 0 else "#27AE60")

    # (c) 2019 vs 2020
    ax3 = axes[2]
    y19 = df[df["year"] == 2019].set_index("month")
    y20 = df[df["year"] == 2020].set_index("month")
    ax3.plot(y19.index, y19["pm25_mean"], "s--", color="#555", lw=1.5, ms=4, label="PM₂.₅ 2019")
    ax3.plot(y20.index, y20["pm25_mean"], "s-",  color=PAL["pm25"], lw=2, ms=5, label="PM₂.₅ 2020")
    ax3b = ax3.twinx()
    ax3b.plot(y19.index, y19["aqi_mean"], "o--", color="#aaa", lw=1.5, ms=4, label="AQI 2019")
    ax3b.plot(y20.index, y20["aqi_mean"], "o-",  color=PAL["aqi"], lw=2, ms=5, label="AQI 2020")
    ax3.axvspan(3, 8, alpha=0.1, color="#DAA520", label="Lockdown")
    ax3.set_xticks(range(1, 13)); ax3.set_xticklabels(MONTH_LABELS, fontsize=7)
    ax3.set_ylabel("PM₂.₅ (µg/m³)"); ax3b.set_ylabel("AQI")
    ax3.set_title("(c) 2019 vs 2020 by Month")
    lines = ax3.get_lines() + ax3b.get_lines()
    ax3.legend(lines, [l.get_label() for l in lines], fontsize=7, ncol=2)
    plt.tight_layout()
    return fig


# ── §9 — Meteorological Drivers ───────────────────────────────────────────────

def plot_rainfall_scatter(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 15. Seaborn scatter plots of rainfall vs each pollutant.
    """
    rain_df = df.dropna(subset=["norm_rain"]).copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, (col, label) in zip(axes.flat, list(zip(POL_COLS[:4], POL_LATEX[:4]))):
        sns.scatterplot(data=rain_df, x="norm_rain", y=col, hue="season",
                        hue_order=SEASON_ORDER, palette=SEASON_PAL,
                        s=55, alpha=0.85, edgecolors="none", ax=ax)
        slope, intercept, r, p, _ = stats.linregress(rain_df["norm_rain"], rain_df[col])
        x_fit = np.linspace(rain_df["norm_rain"].min(), rain_df["norm_rain"].max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "k--", lw=1.5, alpha=0.6)
        r_sp, _ = spearmanr(rain_df["norm_rain"], rain_df[col])
        p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.set_xlabel("Normalised Rainfall"); ax.set_ylabel(label)
        ax.set_title(f"Pearson r={r:.2f} ({p_str}), Spearman ρ={r_sp:.2f}", fontsize=8.5)
        if ax is axes[0, 0]:
            ax.legend(fontsize=7, ncol=2, title="Season")
        else:
            ax.get_legend().remove() if ax.get_legend() else None
    plt.tight_layout()
    return fig


def plot_rainfall_dual_axis(df: pd.DataFrame) -> plt.Figure:
    """
    Figure 16. Monthly rainfall (bars) with PM₂.₅ and AQI (lines), dual-axis.
    """
    rain_df = df.dropna(subset=["norm_rain"]).copy()
    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()
    ax1.bar(rain_df["month_start"], rain_df["norm_rain"],
            color=PAL["rain"], alpha=0.45, width=20, label="Norm. Rainfall")
    ax2.plot(df["month_start"], df["pm25_mean"], color=PAL["pm25"], lw=2, label=r"$\mathrm{PM}_{2.5}$")
    ax2.plot(df["month_start"], df["aqi_mean"],  color=PAL["aqi"], lw=1.5, ls="--", label="AQI")
    ax1.set_ylabel("Normalised Rainfall", color=PAL["rain"])
    ax2.set_ylabel(r"$\mathrm{PM}_{2.5}$ ($\mu$g m$^{-3}$) / AQI")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bars_leg = [mpatches.Patch(color=PAL["rain"], alpha=0.45, label="Norm. Rainfall")]
    ax1.legend(handles=bars_leg + ax2.get_lines(),
               labels=["Norm. Rainfall", r"$\mathrm{PM}_{2.5}$", "AQI"],
               loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


# ── §10 — Socioeconomic ───────────────────────────────────────────────────────

def plot_socioeconomic(annual: pd.DataFrame) -> plt.Figure:
    """
    Figure 17. Annual mean AQI vs HDI, urban share, and poverty rate.
    """
    socio_vars = [
        ("hdi",              "Human Development Index (HDI)", "#1A6B8A"),
        ("urban_share_pct",  "Urban Population Share (%)",    "#D35400"),
        ("poverty_rate_pct", "Poverty Rate (%)",              "#7D3C98"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (var, xlabel, color) in zip(axes, socio_vars):
        sc = ax.scatter(annual[var], annual["aqi_mean"],
                        c=annual["year"], cmap="viridis", s=90, zorder=5,
                        edgecolors="#333", lw=0.5)
        for _, row in annual.iterrows():
            ax.annotate(str(int(row["year"])), (row[var], row["aqi_mean"]),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, color="#333")
        slope, intercept, r, p, _ = stats.linregress(annual[var], annual["aqi_mean"])
        x_fit = np.linspace(annual[var].min(), annual[var].max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color=color, lw=1.5, alpha=0.7)
        p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.set_xlabel(xlabel); ax.set_ylabel("Annual Mean AQI")
        ax.set_title(f"Pearson r={r:.2f}, {p_str}", fontsize=9)
        plt.colorbar(sc, ax=ax, label="Year")
    plt.tight_layout()
    return fig


def plot_per_capita(annual: pd.DataFrame) -> plt.Figure:
    """
    Figure 18. Population-normalised AQI and PM₂.₅ per million residents.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, label, color in [
        (axes[0], "aqi_per_1m",  "AQI per million residents",                                      PAL["aqi"]),
        (axes[1], "pm25_per_1m", r"$\mathrm{PM}_{2.5}$ ($\mu$g m$^{-3}$) per million residents",  PAL["pm25"]),
    ]:
        sns.barplot(data=annual, x="year", y=col, color=color, alpha=0.78, ax=ax)
        ax.set_xlabel("Year"); ax.set_ylabel(label)
        ax.set_title(label.split(" per ")[0] + " per Million Residents")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


# ── §11 — WHO/EPA Exceedance ──────────────────────────────────────────────────

def plot_exceedance(df: pd.DataFrame, exc_df: pd.DataFrame) -> plt.Figure:
    """
    Figure 19. PM₂.₅ time series vs WHO/EPA thresholds + exceedance rate bar chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # (a) PM2.5 bar chart
    ax = axes[0]
    bar_colors = []
    for v in df["pm25_mean"]:
        if   v > EPA["pm25"]:        bar_colors.append("#C0392B")
        elif v > WHO["pm25"] * 5:    bar_colors.append("#E67E22")
        elif v > WHO["pm25"]:        bar_colors.append("#F1C40F")
        else:                         bar_colors.append("#27AE60")
    ax.bar(df["month_start"], df["pm25_mean"], color=bar_colors, width=25, alpha=0.85, zorder=3)
    ax.axhline(WHO["pm25"], color="darkgreen", lw=2, ls="-.", label=f"WHO ({WHO['pm25']}" + r" $\mu$g m$^{-3}$)")
    ax.axhline(EPA["pm25"], color="#E67E22",   lw=2, ls="--", label=f"US EPA ({EPA['pm25']}" + r" $\mu$g m$^{-3}$)")
    ax.set_ylabel(r"$\mathrm{PM}_{2.5}$ Monthly Mean ($\mu$g m$^{-3}$)", fontsize=13)
    ax.set_title("(a) PM₂.₅ vs WHO and EPA Thresholds", fontsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    from matplotlib.patches import Patch
    colour_legend = [
        Patch(color="#C0392B", label=f"> EPA ({EPA['pm25']})"),
        Patch(color="#E67E22", label="> 5× WHO"),
        Patch(color="#F1C40F", label=f"> WHO ({WHO['pm25']})"),
        Patch(color="#27AE60", label="≤ WHO"),
    ]
    ax.legend(handles=colour_legend + ax.get_lines(), fontsize=11, ncol=2)

    # (b) exceedance bar chart
    ax2 = axes[1]
    pols    = exc_df.index.tolist()
    who_exc = exc_df["% months > WHO"].values
    epa_exc = exc_df["% months > EPA"].values
    x = np.arange(len(pols)); w = 0.35
    exc_colors = [PAL["pm25"], PAL["pm10"], PAL["no2"], PAL["so2"]]
    ax2.bar(x - w/2, who_exc, w, label="% > WHO", color=exc_colors, alpha=0.85)
    ax2.bar(x + w/2, epa_exc, w, label="% > EPA", color=exc_colors, alpha=0.45)
    for xi, (wv, ev) in enumerate(zip(who_exc, epa_exc)):
        ax2.text(xi - w/2, wv + 1, f"{wv:.0f}%", ha="center", fontsize=11, fontweight="bold")
        ax2.text(xi + w/2, ev + 1, f"{ev:.0f}%", ha="center", fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(pols, fontsize=12)
    ax2.set_ylabel("% of Monthly Observations Exceeding Threshold", fontsize=12)
    ax2.set_title("(b) WHO & EPA Exceedance Rates", fontsize=13)
    ax2.legend(fontsize=11)
    plt.tight_layout()
    return fig


# ── §12 — Health Burden ───────────────────────────────────────────────────────

def plot_health_burden(health_df: pd.DataFrame) -> plt.Figure:
    """
    Figure 20. Estimated PM₂.₅-attributable mortality (absolute + per 100k).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(data=health_df.reset_index(), x="Year",
                y="PM₂.₅-attributable deaths", color=PAL["pm25"],
                alpha=0.78, ax=axes[0])
    axes[0].set_ylabel("Estimated Attributable Deaths")
    axes[0].set_title("(a) Attributable Deaths (absolute)")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=health_df.reset_index(), x="Year",
                y="Deaths per 100k pop", color=PAL["aqi"],
                alpha=0.78, ax=axes[1])
    axes[1].set_ylabel("Deaths per 100,000 Population")
    axes[1].set_title("(b) Attributable Deaths per 100k")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


# ── §13 — Regional Comparison ─────────────────────────────────────────────────

def plot_regional_comparison() -> plt.Figure:
    """
    Figure 21. Regional PM₂.₅ comparison with South/SE Asian cities.
    """
    regional = {
        "City":       ["Dhaka (this study)","Delhi","Karachi","Lahore",
                       "Kolkata","Mumbai","Jakarta","Ho Chi Minh City",
                       "Bangkok","Beijing","WHO Guideline"],
        "Country":    ["Bangladesh","India","Pakistan","Pakistan",
                       "India","India","Indonesia","Vietnam",
                       "Thailand","China","—"],
        "PM2.5_2017": [132,114, 62,117, 63,47,15,28,27,73, 5],
        "PM2.5_2019": [145, 98, 59,109, 55,45,16,24,26,42, 5],
        "PM2.5_2022": [119, 92, 55, 97, 49,43,15,23,20,28, 5],
        "Region":     ["South Asia","South Asia","South Asia","South Asia",
                       "South Asia","South Asia","SE Asia","SE Asia",
                       "SE Asia","East Asia","Reference"],
    }
    reg_df = pd.DataFrame(regional)
    cities = reg_df[reg_df["City"] != "WHO Guideline"]["City"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # (a) grouped bar
    ax = axes[0]
    x = np.arange(len(cities)); w = 0.25
    year_colors = ["#2980B9","#E67E22","#27AE60"]
    for i, (yr, col) in enumerate(zip(["2017","2019","2022"],
                                       ["PM2.5_2017","PM2.5_2019","PM2.5_2022"])):
        vals = reg_df[reg_df["City"] != "WHO Guideline"][col].values
        bars = ax.bar(x + i * w, vals, w, label=yr, color=year_colors[i], alpha=0.78)
        bars[0].set_edgecolor("red"); bars[0].set_linewidth(2)
    ax.axhline(5, color="red", lw=1.5, ls="-.", label="WHO guideline (5 µg/m³)")
    ax.set_xticks(x + w); ax.set_xticklabels(cities, fontsize=7.5, rotation=25, ha="right")
    ax.set_ylabel("Annual Mean PM₂.₅ (µg/m³)")
    ax.set_title("(a) City Comparison (2017, 2019, 2022)")
    ax.legend(fontsize=8)

    # (b) trend lines for selected cities
    ax2 = axes[1]
    years_plot = [2017, 2019, 2022]
    highlight = {
        "Dhaka (this study)": (PAL["pm25"], "Dhaka"),
        "Delhi":              ("#E67E22", "Delhi"),
        "Lahore":             ("#2980B9", "Lahore"),
        "Beijing":            ("#27AE60", "Beijing"),
        "Jakarta":            ("#8E44AD", "Jakarta"),
    }
    for city, (color, short) in highlight.items():
        row = reg_df[reg_df["City"] == city].iloc[0]
        vals = [row["PM2.5_2017"], row["PM2.5_2019"], row["PM2.5_2022"]]
        lw   = 3 if city == "Dhaka (this study)" else 1.8
        ax2.plot(years_plot, vals, "o-", color=color, lw=lw, ms=7, label=short)
    ax2.axhline(5, color="red", lw=1.5, ls="-.", alpha=0.7, label="WHO guideline")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Annual Mean PM₂.₅ (µg/m³)")
    ax2.set_title("(b) Trend Comparison — Selected Cities")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ── §14–15 — Forecasting Figures ──────────────────────────────────────────────

def plot_model_evaluation(df_test: pd.DataFrame, model_preds: dict) -> plt.Figure:
    """
    Figure 22. Out-of-sample test-period forecasts vs observed values.
    """
    model_colors = {
        "OLS":      "#555",
        "ETS":      "#E67E22",
        "SARIMA":   "#2980B9",
        "Prophet":  "#C0392B",
        "Ensemble": "#27AE60",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, col in zip(axes.flat, ["pm25_mean","aqi_mean","no2_mean","so2_mean"]):
        res    = model_preds[col]
        x_test = df_test["month_start"].values
        ax.plot(x_test, res["true"], "ko-", lw=2, ms=5, label="Observed", zorder=6)
        for mname, yhat in res["preds"].items():
            lw = 2.5 if mname == "Ensemble" else 1.5
            ls = "-" if mname == "Ensemble" else "--"
            ax.plot(x_test, yhat, ls, color=model_colors[mname], lw=lw, label=mname, alpha=0.85)
        ax.set_title(f"{res['label']} — Test Period 2024–2025")
        ax.set_ylabel(res["label"])
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=7.5, ncol=3)
    plt.tight_layout()
    return fig


def plot_forecasts_2030(df: pd.DataFrame,
                        future_df: pd.DataFrame,
                        forecasts: dict) -> plt.Figure:
    """
    Figure 23. Multi-model ensemble forecasts to 2030 with Prophet 90% PI.
    """
    mcolors = {
        "ols":      "#888", "ets":      "#E67E22",
        "sarima":   "#2980B9", "prophet": "#C0392B", "ensemble": "#27AE60",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, target in zip(axes.flat, ["pm25_mean","aqi_mean","no2_mean","so2_mean"]):
        res   = forecasts[target]
        color = PAL[target.replace("_mean", "")]
        obs_x = df["month_start"]; fut_x = future_df["month_start"]
        ax.plot(obs_x, df[target], color=color, lw=2, alpha=0.9, label="Observed (2017–2025)")
        ax.fill_between(fut_x, res["p_lo"], res["p_hi"],
                        alpha=0.12, color=color, label="Prophet 90% PI")
        ax.plot(fut_x, res["ols"],      "--", color=mcolors["ols"],      lw=1.3, label="OLS")
        ax.plot(fut_x, res["ets"],      "-.", color=mcolors["ets"],       lw=1.3, label="ETS")
        ax.plot(fut_x, res["sarima"],   ":",  color=mcolors["sarima"],    lw=1.5, label="SARIMA")
        ax.plot(fut_x, res["prophet"],  "-",  color=mcolors["prophet"],   lw=1.3, label="Prophet")
        ax.plot(fut_x, res["ensemble"], "-",  color=mcolors["ensemble"],  lw=2.5, label="Ensemble")
        ax.axvline(df["month_start"].max(), color="grey", lw=1, ls=":", alpha=0.8)
        who_key = target.replace("_mean", "")
        if who_key in WHO:
            ax.axhline(WHO[who_key], color="red", lw=1, ls="-.", alpha=0.6,
                       label=f"WHO ({WHO[who_key]})")
        ax.set_ylabel(f"{res['label']} (µg/m³)")
        ax.set_title(f"{res['label']}  |  OLS R²={res['r2_ols']:.3f}")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    plt.tight_layout()
    return fig


# ── Additional publication figures ────────────────────────────────────────────

def plot_violin_by_season(df: pd.DataFrame) -> plt.Figure:
    """
    Additional Figure A. Split violin plots of PM₂.₅, PM₁₀, NO₂, SO₂, and AQI
    by meteorological season.

    Violins reveal the full distribution shape (multi-modality, skewness) which
    box plots suppress. Inner quartile lines and the median point are overlaid.
    A horizontal WHO guideline is drawn where applicable.
    """
    pol_colors = [PAL["pm25"], PAL["pm10"], PAL["no2"], PAL["so2"], PAL["aqi"]]
    who_vals   = [WHO["pm25"], WHO["pm10"], WHO["no2"], WHO["so2"], None]
    pol_cfg = list(zip(POL_COLS, POL_LATEX, pol_colors, who_vals))

    fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=False)

    for ax, (col, label, color, who_val) in zip(axes, pol_cfg):
        sns.violinplot(
            data=df, x="season", y=col, order=SEASON_ORDER,
            palette=SEASON_PAL,
            inner="quartile",      # show IQR lines inside violin
            linewidth=0.9,
            cut=0,                 # do not extend beyond data range
            ax=ax,
        )
        # WHO guideline
        if who_val is not None:
            ax.axhline(who_val, color="red", lw=1.2, ls="--", alpha=0.8,
                       label=f"WHO ({who_val})")
            ax.legend(fontsize=7.5, loc="upper right")

        ax.set_xticklabels(["Winter", "Pre-\nMon.", "Monsoon", "Post-\nMon."],
                           fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=10)

    plt.tight_layout()
    return fig


def plot_interannual_boxplots(df: pd.DataFrame) -> plt.Figure:
    """
    Additional Figure B. Year-by-year box plots for PM₂.₅ and AQI showing
    the distribution of 12 monthly values per year.

    Complements the annual mean trend by showing whether variability is
    increasing or decreasing over the study period.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for ax, (col, label, color) in zip(axes, [
        ("pm25_mean", r"$\mathrm{PM}_{2.5}$ ($\mu$g m$^{-3}$)", PAL["pm25"]),
        ("aqi_mean",  "AQI",                                      PAL["aqi"]),
    ]):
        # Colour 2020 distinctly (COVID year); cast keys to str to match seaborn x
        years_sorted = sorted(df["year"].unique())
        palette = {str(yr): ("#DAA520" if yr == 2020 else color)
                   for yr in years_sorted}
        df_plot = df.copy()
        df_plot["year"] = df_plot["year"].astype(str)
        sns.boxplot(
            data=df_plot, x="year", y=col,
            palette=palette,
            width=0.6,
            flierprops={"marker": "o", "markersize": 3.5, "alpha": 0.6},
            medianprops={"color": "white", "lw": 2},
            ax=ax,
        )
        # WHO guideline
        who_key = col.replace("_mean", "")
        if who_key in WHO:
            ax.axhline(WHO[who_key], color="red", lw=1.2, ls="--", alpha=0.8,
                       label=f"WHO ({WHO[who_key]}" + r" $\mu$g m$^{-3}$)")
            ax.legend(fontsize=9)

        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=11)

        # Annotate COVID year
        covid_idx = years_sorted.index(2020)
        ax.text(covid_idx, ax.get_ylim()[1] * 0.97, "COVID-19",
                ha="center", fontsize=7.5, color="#8B6914", style="italic")

    axes[-1].set_xlabel("Year", fontsize=11)
    axes[-1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    return fig


def plot_model_performance_heatmap(eval_df: pd.DataFrame) -> plt.Figure:
    """
    Additional Figure C. Heatmap of model MAPE (%) across all four target
    variables (PM₂.₅, AQI, NO₂, SO₂) — a compact visual model leaderboard.

    Cells are annotated with the numeric MAPE value; the colour scale runs
    from low (good, green) to high (poor, red). The Ensemble row is outlined
    for emphasis.
    """
    variables = ["PM₂.₅", "AQI", "NO₂", "SO₂"]
    models    = ["OLS", "ETS", "SARIMA", "Prophet", "Ensemble"]

    # Build MAPE pivot
    mape_pivot = (
        eval_df.pivot_table(index="Model", columns="Variable", values="MAPE%")
        .reindex(index=models, columns=variables)
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    metric_titles = [("MAE", "MAE"), ("RMSE", "RMSE"), ("MAPE%", "MAPE (%)")]

    for ax, (metric, title) in zip(axes, metric_titles):
        pivot = (
            eval_df.pivot_table(index="Model", columns="Variable", values=metric)
            .reindex(index=models, columns=variables)
        )
        cmap = "RdYlGn_r" if metric != "MAPE%" else "RdYlGn_r"
        sns.heatmap(
            pivot, annot=True, fmt=".1f",
            cmap=cmap, ax=ax,
            linewidths=0.5, linecolor="#cccccc",
            annot_kws={"size": 11, "weight": "bold"},
            cbar_kws={"shrink": 0.8, "label": title},
        )
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Model" if ax is axes[0] else "")
        ax.tick_params(axis="x", labelsize=10, rotation=0)
        ax.tick_params(axis="y", labelsize=10, rotation=0)

        # Highlight Ensemble row with a rectangle
        ens_idx = models.index("Ensemble")
        ax.add_patch(plt.Rectangle(
            (0, ens_idx), len(variables), 1,
            fill=False, edgecolor="#2C3E50", lw=2.5, zorder=5,
        ))

    plt.tight_layout()
    return fig
