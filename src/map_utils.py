"""
map_utils.py
------------
Geospatial map of the Dhaka City District study area.

Produces a publication-quality figure similar to the reference image:
  - Cream/beige background fill for Dhaka district
  - Thick dark-maroon outer district boundary
  - Thin light-grey upazila boundaries
  - Selected upazila labels
  - Compass rose (N/S/E/W) in upper-right
  - Scale bar at bottom-centre
  - White surrounding area (outside Dhaka district)
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# Shapefile paths
SHP_DIR  = pathlib.Path("data/shapefiles")
ADM2_SHP = SHP_DIR / "bgd_admbnda_adm2_bbs_20201113.shp"
ADM3_SHP = SHP_DIR / "bgd_admbnda_adm3_bbs_20201113.shp"

# Upazilas to label (matching names in the reference image)
LABEL_UPAZILAS = [
    "Dhamrai", "Savar", "Uttara", "Turag", "Khilkhet",
    "Mirpur", "Keraniganj", "Nawabganj", "Dohar",
    "Demra", "Motijheel", "Jatrabari",
]

# Manual label offsets (dx, dy in degrees) for legibility
LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "Dhamrai":    (-0.05,  0.00),
    "Savar":      ( 0.02,  0.02),
    "Uttara":     ( 0.02,  0.00),
    "Turag":      (-0.02,  0.02),
    "Khilkhet":   ( 0.02,  0.00),
    "Mirpur":     (-0.04,  0.00),
    "Keraniganj": (-0.03, -0.02),
    "Nawabganj":  (-0.06,  0.00),
    "Dohar":      (-0.04, -0.02),
    "Demra":      ( 0.02,  0.00),
    "Motijheel":  ( 0.02, -0.01),
    "Jatrabari":  ( 0.02, -0.01),
}


def _add_compass_rose(ax: plt.Axes, x: float, y: float, size: float = 0.06) -> None:
    """
    Draw a classic 4-point star compass rose in axes-fraction space using an
    inset axes, matching the style in the reference image.

    Parameters
    ----------
    ax   : the map axes
    x, y : centre of the compass in axes-fraction coordinates
    size : half-width of the inset in axes-fraction units
    """
    # Create a small inset axes for the compass so it is always pixel-perfect
    inset = ax.inset_axes([x - size, y - size, size * 2, size * 2])
    inset.set_xlim(-1, 1)
    inset.set_ylim(-1, 1)
    inset.set_aspect("equal")
    inset.axis("off")

    # 4-point star: two overlapping diamond polygons (N/S = tall, E/W = wide)
    # North/South diamond (tall, black)
    ns = plt.Polygon(
        [(-0.18, 0), (0, 0.85), (0.18, 0), (0, -0.85)],
        closed=True, facecolor="black", edgecolor="black", linewidth=0.5, zorder=2,
    )
    # East/West diamond (wide, white with black border)
    ew = plt.Polygon(
        [(0, 0.18), (0.85, 0), (0, -0.18), (-0.85, 0)],
        closed=True, facecolor="white", edgecolor="black", linewidth=0.7, zorder=3,
    )
    inset.add_patch(ns)
    inset.add_patch(ew)

    # Small circle at centre
    circle = plt.Circle((0, 0), 0.12, color="white", ec="black", lw=0.8, zorder=4)
    inset.add_patch(circle)

    # Cardinal labels
    fs = 7
    inset.text( 0,    1.08, "N", ha="center", va="bottom", fontsize=fs, fontweight="bold", zorder=5)
    inset.text( 0,   -1.08, "S", ha="center", va="top",    fontsize=fs, zorder=5)
    inset.text( 1.08,  0,   "E", ha="left",   va="center", fontsize=fs, zorder=5)
    inset.text(-1.08,  0,   "W", ha="right",  va="center", fontsize=fs, zorder=5)


def _add_scale_bar(ax: plt.Axes, length_km: float = 20,
                   x0_frac: float = 0.15, y0_frac: float = 0.05) -> None:
    """
    Draw a simple scale bar in the lower-left region of the axes.

    Parameters
    ----------
    length_km : total bar length in kilometres
    x0_frac   : left edge in axes fraction
    y0_frac   : bottom edge in axes fraction
    """
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # 1 degree latitude ≈ 111 km at Dhaka's latitude (~23.8°N)
    deg_per_km = 1 / 111.0
    bar_deg    = length_km * deg_per_km

    x0 = xlim[0] + x0_frac * x_range
    y0 = ylim[0] + y0_frac * y_range
    bar_h = y_range * 0.008   # thin bar height

    # Split bar into 4 segments with alternating fill
    n_seg  = 4
    seg_km = length_km / n_seg
    seg_d  = bar_deg / n_seg
    for i in range(n_seg):
        color = "black" if i % 2 == 0 else "white"
        rect  = mpatches.FancyBboxPatch(
            (x0 + i * seg_d, y0), seg_d, bar_h,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="black", linewidth=0.7,
        )
        ax.add_patch(rect)

    # Tick labels at 0, half, full
    for km, x_pos in [(0, x0), (length_km / 2, x0 + bar_deg / 2), (length_km, x0 + bar_deg)]:
        ax.text(x_pos, y0 - y_range * 0.012, f"{int(km)}",
                ha="center", va="top", fontsize=7, color="black")

    ax.text(x0 + bar_deg / 2, y0 - y_range * 0.028, "km",
            ha="center", va="top", fontsize=7, color="black")


def plot_study_area(
    adm2_shp: str | pathlib.Path = ADM2_SHP,
    adm3_shp: str | pathlib.Path = ADM3_SHP,
    district:  str = "Dhaka",
) -> plt.Figure:
    """
    Generate the study area map for Dhaka City District, Bangladesh.

    Parameters
    ----------
    adm2_shp : path to ADM2 (district) shapefile
    adm3_shp : path to ADM3 (upazila) shapefile
    district  : district name as it appears in the ADM2_EN column

    Returns
    -------
    matplotlib Figure
    """
    import geopandas as gpd

    adm2_shp = pathlib.Path(adm2_shp)
    adm3_shp = pathlib.Path(adm3_shp)

    if not adm2_shp.exists():
        raise FileNotFoundError(f"ADM2 shapefile not found: {adm2_shp}")
    if not adm3_shp.exists():
        raise FileNotFoundError(f"ADM3 shapefile not found: {adm3_shp}")

    # Load and filter to Dhaka district
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adm2 = gpd.read_file(adm2_shp)
        adm3 = gpd.read_file(adm3_shp)

    dhaka_district = adm2[adm2["ADM2_EN"].str.contains(district, case=False, na=False)]
    dhaka_upazilas = adm3[adm3["ADM2_EN"].str.contains(district, case=False, na=False)]

    if dhaka_district.empty:
        raise ValueError(f"No ADM2 feature found matching '{district}'.")
    if dhaka_upazilas.empty:
        raise ValueError(f"No ADM3 features found within '{district}' district.")

    # ── Canvas ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect("equal")

    # White figure & axes background (outside district)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Plot layers ────────────────────────────────────────────────────────────
    # 1. Upazila fill (cream background for the district interior)
    dhaka_upazilas.plot(
        ax=ax,
        facecolor="#F5F0E4",   # warm cream / parchment
        edgecolor="#888888",   # medium grey upazila borders (visible but not dominant)
        linewidth=0.8,
        zorder=1,
    )

    # 2. District outer boundary (dark maroon, thick)
    dhaka_district.boundary.plot(
        ax=ax,
        edgecolor="#7B1E1E",   # dark maroon
        linewidth=2.8,
        zorder=3,
    )

    # ── Upazila labels ─────────────────────────────────────────────────────────
    for _, row in dhaka_upazilas.iterrows():
        name = row["ADM3_EN"]
        if name not in LABEL_UPAZILAS:
            continue
        centroid = row.geometry.centroid
        cx, cy   = centroid.x, centroid.y
        dx, dy   = LABEL_OFFSETS.get(name, (0, 0))
        ax.text(
            cx + dx, cy + dy, name,
            fontsize=6.5, ha="center", va="center",
            color="#333333",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            zorder=5,
        )

    # ── Compute extent with padding ────────────────────────────────────────────
    bounds   = dhaka_district.total_bounds   # [minx, miny, maxx, maxy]
    x_pad    = (bounds[2] - bounds[0]) * 0.06
    y_pad    = (bounds[3] - bounds[1]) * 0.06
    ax.set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
    ax.set_ylim(bounds[1] - y_pad, bounds[3] + y_pad)

    # ── Compass rose ───────────────────────────────────────────────────────────
    _add_compass_rose(ax, x=0.88, y=0.88, size=0.065)

    # ── Scale bar ──────────────────────────────────────────────────────────────
    _add_scale_bar(ax, length_km=20, x0_frac=0.12, y0_frac=0.04)

    # ── Axes cosmetics ─────────────────────────────────────────────────────────
    ax.set_title(
        "Study Area: Dhaka City, Bangladesh",
        fontsize=13, fontweight="bold", pad=14, color="#1a1a1a",
    )
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.8)
    ax.grid(False)

    plt.tight_layout()
    return fig
