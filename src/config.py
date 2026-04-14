"""
config.py
---------
Shared constants, colour palettes, and rcParams for all modules.
Import this first in every notebook cell and src module.
"""

import matplotlib as mpl
import seaborn as sns

# ── Colour palettes ────────────────────────────────────────────────────────────
PAL = {
    "pm25": "#C0392B",
    "pm10": "#E67E22",
    "no2":  "#2980B9",
    "so2":  "#27AE60",
    "aqi":  "#8E44AD",
    "rain": "#1A9BC4",
}

SEASON_PAL = {
    "Winter":       "#2980B9",
    "Pre-monsoon":  "#E67E22",
    "Monsoon":      "#27AE60",
    "Post-monsoon": "#8E44AD",
}

SEASON_ORDER  = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
MONTH_LABELS  = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Pollutant / variable helpers ───────────────────────────────────────────────
POL_COLS  = ["pm25_mean","pm10_mean","no2_mean","so2_mean","aqi_mean"]
POL_NAMES = ["PM₂.₅ (µg/m³)","PM₁₀ (µg/m³)","NO₂ (µg/m³)","SO₂ (µg/m³)","AQI"]
POL_SHORT = ["PM₂.₅","PM₁₀","NO₂","SO₂","AQI"]

# Mathtext versions (matplotlib mathtext; no system LaTeX required)
POL_LATEX = [
    r"$\mathrm{PM}_{2.5}$ ($\mu$g m$^{-3}$)",
    r"$\mathrm{PM}_{10}$ ($\mu$g m$^{-3}$)",
    r"$\mathrm{NO}_2$ ($\mu$g m$^{-3}$)",
    r"$\mathrm{SO}_2$ ($\mu$g m$^{-3}$)",
    "AQI",
]
POL_SHORT_LATEX = [
    r"$\mathrm{PM}_{2.5}$",
    r"$\mathrm{PM}_{10}$",
    r"$\mathrm{NO}_2$",
    r"$\mathrm{SO}_2$",
    "AQI",
]

# ── Air quality guidelines ─────────────────────────────────────────────────────
WHO = {"pm25": 5,  "pm10": 15,  "no2": 10,  "so2": 40}   # WHO 2021 annual
EPA = {"pm25": 35, "pm10": 150, "no2": 100, "so2": 75}   # US EPA 24-h

# ── AQI category breakpoints (US EPA) ─────────────────────────────────────────
AQI_CATS = [
    (0,   50,  "Good",                   "#5A9E6A"),  # muted green
    (51,  100, "Moderate",               "#C4A94A"),  # muted amber
    (101, 150, "Unhealthy for Sensitive","#CC7A3A"),  # muted burnt orange
    (151, 200, "Unhealthy",              "#B54040"),  # muted brick red
    (201, 300, "Very Unhealthy",         "#7A4E82"),  # muted purple
    (301, 500, "Hazardous",              "#5C2E2E"),  # dark maroon
]

# ── Matplotlib publication style ───────────────────────────────────────────────
PUB_STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#111111",
    "axes.titlecolor":   "#111111",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.5,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "text.color":        "#111111",
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
}


def apply_style() -> None:
    """Apply publication-ready rcParams globally (seaborn theme + custom overrides)."""
    sns.set_theme(style="ticks")
    mpl.rcParams.update(PUB_STYLE)
