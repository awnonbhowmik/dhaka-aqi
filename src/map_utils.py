"""
map_utils.py
------------
Geospatial map of the Dhaka City study area.

Produces a publication-quality figure:
  - Cream/beige background fill for Dhaka City thanas
  - Thick dark-maroon outer city boundary (41 urban thanas dissolved)
  - Thin light-grey thana (ADM3) boundaries
  - Selected thana labels
  - Compass rose (N/S/E/W) in upper-right
  - Scale bar at bottom-left

Note: Dhaka District (ADM2) contains 46 administrative units.  Five are
rural/peri-urban (Dhamrai, Savar, Keraniganj, Nawabganj, Dohar) and are
excluded here; the remaining 41 urban thanas constitute Dhaka City as
covered by the Dhaka City Corporation (DNCC + DSCC) and form the study area
consistent with the monitoring station data.
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

# Rural/peri-urban thanas that are part of Dhaka District but NOT Dhaka City
RURAL_THANAS = {"Dhamrai", "Savar", "Keraniganj", "Nawabganj", "Dohar"}

# Thanas to label on the map (urban core, spread across the city)
LABEL_THANAS = [
    "Uttara", "Mirpur", "Pallabi", "Gulshan",
    "Tejgaon", "Dhanmondi", "Mohammadpur",
    "Motijheel", "Lalbagh", "Jatrabari",
    "Demra", "Khilkhet",
]

# Manual label offsets (dx, dy in degrees) for legibility.
# Centroids (lon, lat): Uttara(90.394,23.872), Mirpur(90.362,23.794),
# Pallabi(90.368,23.825), Gulshan(90.412,23.790), Tejgaon(90.392,23.763),
# Dhanmondi(90.374,23.745), Mohammadpur(90.357,23.759), Motijheel(90.422,23.735),
# Lalbagh(90.378,23.721), Jatrabari(90.451,23.713), Demra(90.481,23.725),
# Khilkhet(90.448,23.835)
LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "Uttara":      ( 0.00, -0.01),   # near top edge — nudge down
    "Mirpur":      ( 0.03,  0.01),   # near left edge — push right
    "Pallabi":     ( 0.03,  0.01),   # near left edge — push right
    "Gulshan":     ( 0.02,  0.01),
    "Tejgaon":     ( 0.01,  0.00),
    "Dhanmondi":   ( 0.02, -0.01),
    "Mohammadpur": ( 0.03,  0.00),   # near left edge — push right
    "Motijheel":   ( 0.01, -0.01),
    "Lalbagh":     ( 0.01, -0.01),
    "Jatrabari":   ( 0.00, -0.01),
    "Demra":       (-0.03,  0.01),   # near right edge — push left
    "Khilkhet":    ( 0.01,  0.01),
}


def _add_compass_rose(ax: plt.Axes, x: float, y: float, size: float = 0.06) -> None:
    """
    Draw a classic 4-point star compass rose in axes-fraction space using an
    inset axes, matching the style in the reference image.
    """
    inset = ax.inset_axes([x - size, y - size, size * 2, size * 2])
    inset.set_xlim(-1, 1)
    inset.set_ylim(-1, 1)
    inset.set_aspect("equal")
    inset.axis("off")

    ns = plt.Polygon(
        [(-0.18, 0), (0, 0.85), (0.18, 0), (0, -0.85)],
        closed=True, facecolor="black", edgecolor="black", linewidth=0.5, zorder=2,
    )
    ew = plt.Polygon(
        [(0, 0.18), (0.85, 0), (0, -0.18), (-0.85, 0)],
        closed=True, facecolor="white", edgecolor="black", linewidth=0.7, zorder=3,
    )
    inset.add_patch(ns)
    inset.add_patch(ew)

    circle = plt.Circle((0, 0), 0.12, color="white", ec="black", lw=0.8, zorder=4)
    inset.add_patch(circle)

    fs = 7
    inset.text( 0,    1.08, "N", ha="center", va="bottom", fontsize=fs, fontweight="bold", zorder=5)
    inset.text( 0,   -1.08, "S", ha="center", va="top",    fontsize=fs, zorder=5)
    inset.text( 1.08,  0,   "E", ha="left",   va="center", fontsize=fs, zorder=5)
    inset.text(-1.08,  0,   "W", ha="right",  va="center", fontsize=fs, zorder=5)


def _add_scale_bar(ax: plt.Axes, length_km: float = 10,
                   x0_frac: float = 0.12, y0_frac: float = 0.05) -> None:
    """
    Draw a simple scale bar in the lower-left region of the axes.
    """
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # 1 degree latitude ≈ 111 km at Dhaka's latitude (~23.8°N)
    deg_per_km = 1 / 111.0
    bar_deg    = length_km * deg_per_km

    x0 = xlim[0] + x0_frac * x_range
    y0 = ylim[0] + y0_frac * y_range
    bar_h = y_range * 0.008

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
    Generate the study area map for Dhaka City, Bangladesh.

    Uses the 41 urban thanas of Dhaka District (excluding the five
    rural/peri-urban thanas: Dhamrai, Savar, Keraniganj, Nawabganj, Dohar)
    to represent the Dhaka City Corporation (DNCC + DSCC) study area,
    consistent with the monitoring station dataset.

    Parameters
    ----------
    adm2_shp : path to ADM2 (district) shapefile  (unused for boundary,
               kept for API compatibility)
    adm3_shp : path to ADM3 (thana/upazila) shapefile
    district  : district name as it appears in the ADM2_EN column

    Returns
    -------
    matplotlib Figure
    """
    import geopandas as gpd

    adm3_shp = pathlib.Path(adm3_shp)
    if not adm3_shp.exists():
        raise FileNotFoundError(f"ADM3 shapefile not found: {adm3_shp}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adm3 = gpd.read_file(adm3_shp)

    # All thanas in Dhaka District
    all_thanas = adm3[adm3["ADM2_EN"].str.contains(district, case=False, na=False)]

    # City thanas: exclude the five rural/peri-urban thanas
    city_thanas = all_thanas[~all_thanas["ADM3_EN"].isin(RURAL_THANAS)].copy()

    if city_thanas.empty:
        raise ValueError(f"No city thanas found for district '{district}'.")

    # Dissolve city thanas into a single outer boundary polygon
    city_boundary = city_thanas.dissolve()

    # ── Canvas ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 9))
    ax.set_aspect("equal")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Plot layers ────────────────────────────────────────────────────────────
    # 1. City thana fill (cream) with light internal boundaries
    city_thanas.plot(
        ax=ax,
        facecolor="#F5F0E4",
        edgecolor="#999999",
        linewidth=0.6,
        zorder=1,
    )

    # 2. Outer city boundary (dark maroon, thick)
    city_boundary.boundary.plot(
        ax=ax,
        edgecolor="#7B1E1E",
        linewidth=2.5,
        zorder=3,
    )

    # ── Thana labels ───────────────────────────────────────────────────────────
    for _, row in city_thanas.iterrows():
        name = row["ADM3_EN"]
        if name not in LABEL_THANAS:
            continue
        centroid = row.geometry.centroid
        cx, cy   = centroid.x, centroid.y
        dx, dy   = LABEL_OFFSETS.get(name, (0, 0))
        ax.text(
            cx + dx, cy + dy, name,
            fontsize=6, ha="center", va="center",
            color="#333333", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            zorder=5,
        )

    # ── Compute extent with padding ────────────────────────────────────────────
    bounds = city_boundary.total_bounds   # [minx, miny, maxx, maxy]
    x_pad  = (bounds[2] - bounds[0]) * 0.07
    y_pad  = (bounds[3] - bounds[1]) * 0.07
    ax.set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
    ax.set_ylim(bounds[1] - y_pad, bounds[3] + y_pad)

    # ── Compass rose ───────────────────────────────────────────────────────────
    _add_compass_rose(ax, x=0.88, y=0.88, size=0.065)

    # ── Scale bar ──────────────────────────────────────────────────────────────
    _add_scale_bar(ax, length_km=10, x0_frac=0.12, y0_frac=0.04)

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
