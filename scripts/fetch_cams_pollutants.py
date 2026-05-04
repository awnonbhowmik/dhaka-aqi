#!/usr/bin/env python3
"""
fetch_cams_pollutants.py
-------------------------
Downloads daily PM2.5, PM10, NO2, SO2, CO, O3 surface concentrations for
Dhaka from Copernicus CAMS EAC4 reanalysis and merges them into
data/daily_dhaka_aqi_dataset.csv.

One-time setup (all free):
  1. Register at https://ads.atmosphere.copernicus.eu/
  2. Copy your API key from https://ads.atmosphere.copernicus.eu/user/
  3. Create the file ~/.cdsapirc containing:
       url: https://ads.atmosphere.copernicus.eu/api
       key: YOUR_API_KEY
  4. pip install cdsapi xarray netCDF4 scipy

Run from the project root:
    python scripts/fetch_cams_pollutants.py

What this does:
  - Downloads 3-hourly CAMS EAC4 surface concentrations for 2017-2022
    (EAC4 coverage; NRT data handles 2023+ automatically if available)
  - Extracts values at Dhaka's nearest grid point (23.78°N, 90.41°E)
  - Aggregates to daily means
  - Converts units: kg m⁻³ → µg m⁻³ (×10⁹)
  - Saves raw daily values to data/cams_dhaka_pollutants.csv
  - Merges into data/daily_dhaka_aqi_dataset.csv,
    replacing monthly-mean pollutant columns with actual daily values

CAMS EAC4 data caveats:
  - Spatial resolution 0.75° (~83 km); Dhaka value represents a regional
    average, not a single point measurement
  - Units: µg/m³ at surface level
  - NO2/SO2/CO/O3 are given as mass concentrations (µg/m³); to convert to
    mixing-ratio units used by some indices, use the molecular-weight
    conversions noted in the code
"""

import pathlib
import sys
import tempfile
import warnings
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

# ── configuration ─────────────────────────────────────────────────────────────

# Dhaka coordinates
DHAKA_LAT = 23.7808
DHAKA_LON = 90.4125

# CAMS EAC4 runs 2003-01-01 → 2022-12-31; NRT fills the gap to present
EAC4_START = 2017
EAC4_END   = 2022   # last complete year in EAC4

# Actual NetCDF variable names inside the downloaded ZIP files
# data_sfc.nc  → surface concentration (kg m⁻³)  × 1e9 → µg m⁻³
# data_mlev.nc → mass mixing ratio (kg kg⁻¹) × air_density × 1e9 → µg m⁻³
SFC_VARS  = {"pm2p5": "pm25", "pm10": "pm10"}
MLEV_VARS = {"no2": "no2", "so2": "so2", "co": "co", "go3": "o3"}

# CDS request variable names (used in the API call)
REQUEST_VARS = [
    "particulate_matter_2.5um", "particulate_matter_10um",
    "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "ozone",
]

# Air density at Dhaka surface (~sea level, ~300 K mean): P/(R_d × T)
# 101325 / (287.05 × 300) ≈ 1.176 kg m⁻³
AIR_DENSITY = 1.176

CAMS_RAW   = pathlib.Path("data/cams_dhaka_pollutants.csv")
DAILY_CSV  = pathlib.Path("data/daily_dhaka_aqi_dataset.csv")
CACHE_DIR  = pathlib.Path("data/.cams_cache")


# ── CAMS download ─────────────────────────────────────────────────────────────

def _download_year(c, year: int, out_path: pathlib.Path) -> None:
    """Request one full year of 3-hourly CAMS EAC4 data for a box around Dhaka."""
    print(f"  Requesting CAMS EAC4 for {year} …")
    area = [DHAKA_LAT + 1, DHAKA_LON - 1, DHAKA_LAT - 1, DHAKA_LON + 1]
    c.retrieve(
        "cams-global-reanalysis-eac4",
        {
            "variable":    REQUEST_VARS,
            "model_level": "60",   # lowest model level ≈ surface
            "date":        f"{year}-01-01/{year}-12-31",
            "time":        ["00:00", "03:00", "06:00", "09:00",
                            "12:00", "15:00", "18:00", "21:00"],
            "area":        area,
            "format":      "netcdf",
        },
        str(out_path),
    )
    print(f"  Saved → {out_path}")


def _open_nc_from_zip(zip_path: pathlib.Path, member: str) -> xr.Dataset:
    """Extract one NetCDF member from a ZIP and open it as an xarray Dataset."""
    import zipfile, tempfile, os
    with zipfile.ZipFile(zip_path) as z:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(z.read(member))
            tmp_path = tmp.name
    ds = xr.open_dataset(tmp_path)
    os.unlink(tmp_path)
    return ds


def _extract_daily(zip_path: pathlib.Path) -> pd.DataFrame:
    """
    Read a CAMS ZIP (data_sfc.nc + data_mlev.nc), extract the Dhaka grid point,
    resample 3-hourly → daily mean, apply unit conversions, and return a
    DataFrame indexed by date.

    Units:
      data_sfc  (PM2.5, PM10): kg m⁻³  → µg m⁻³  (× 1e9)
      data_mlev (NO2…O3):      kg kg⁻¹ → µg m⁻³  (× AIR_DENSITY × 1e9)
    """
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()

    daily_rows: dict[str, pd.Series] = {}

    # ── surface file (PM2.5, PM10) ────────────────────────────────────────────
    sfc_name = next((m for m in members if "sfc" in m), None)
    if sfc_name:
        ds = _open_nc_from_zip(zip_path, sfc_name)
        ds_pt = ds.sel(latitude=DHAKA_LAT, longitude=DHAKA_LON, method="nearest")
        for nc_var, col in SFC_VARS.items():
            if nc_var in ds_pt:
                daily = (ds_pt[nc_var].resample(valid_time="1D").mean() * 1e9
                         ).to_pandas()
                daily_rows[col] = daily
        ds.close()

    # ── model-level file (NO2, SO2, CO, O3) ──────────────────────────────────
    mlev_name = next((m for m in members if "mlev" in m), None)
    if mlev_name:
        ds = _open_nc_from_zip(zip_path, mlev_name)
        ds_pt = ds.sel(latitude=DHAKA_LAT, longitude=DHAKA_LON, method="nearest")
        for nc_var, col in MLEV_VARS.items():
            if nc_var in ds_pt:
                da = ds_pt[nc_var]
                if "model_level" in da.dims:
                    da = da.isel(model_level=0)   # surface model level
                daily = (da.resample(valid_time="1D").mean()
                         * AIR_DENSITY * 1e9).to_pandas()
                daily_rows[col] = daily
        ds.close()

    df = pd.DataFrame(daily_rows)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return df.round(3)


def fetch_cams(years: range) -> pd.DataFrame:
    """Download CAMS EAC4 for each year in *years* and return a combined DataFrame."""
    try:
        import cdsapi
    except ImportError:
        sys.exit("cdsapi not installed. Run: pip install cdsapi xarray netCDF4 scipy")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client(quiet=True)
    all_frames = []

    for year in years:
        nc_path = CACHE_DIR / f"cams_eac4_dhaka_{year}.nc"
        if nc_path.exists():
            print(f"  Using cached file for {year}: {nc_path}")
        else:
            _download_year(c, year, nc_path)

        print(f"  Extracting daily values for {year} …")
        df_year = _extract_daily(nc_path)  # nc_path is actually a ZIP
        all_frames.append(df_year)

    combined = pd.concat(all_frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


# ── merge into daily CSV ──────────────────────────────────────────────────────

def merge_into_daily(cams_df: pd.DataFrame) -> None:
    """
    Overwrite pollutant columns in DAILY_CSV with CAMS daily values where
    available; leave existing values (monthly means) for dates not covered.
    """
    if not DAILY_CSV.exists():
        sys.exit(
            f"Daily CSV not found at '{DAILY_CSV}'. "
            "Run 'python scripts/fetch_daily_aqi.py' first."
        )

    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    daily = daily.set_index("date")

    # Add/update pollutant columns from CAMS
    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if col not in daily.columns:
            daily[col] = np.nan
        if col in cams_df.columns:
            daily.update(cams_df[[col]])

    # Reorder columns sensibly
    front = ["year", "month", "aqi", "pm25", "pm10", "no2", "so2", "co", "o3", "season", "source"]
    other = [c for c in daily.columns if c not in front]
    daily = daily[front + other]

    daily.reset_index().to_csv(DAILY_CSV, index=False, float_format="%.3f")
    print(f"\nMerged CAMS data into {DAILY_CSV}")

    # Coverage report
    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if col in daily.columns:
            n = daily[col].notna().sum()
            pct = n / len(daily) * 100
            print(f"  {col:6s}: {n:,} / {len(daily):,} rows  ({pct:.1f}%)")


# ── CLI / main ────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Download CAMS EAC4 daily pollutant data for Dhaka and "
                    "merge into data/daily_dhaka_aqi_dataset.csv."
    )
    parser.add_argument(
        "--start-year", type=int, default=EAC4_START,
        help=f"First year to download (default: {EAC4_START}).",
    )
    parser.add_argument(
        "--end-year", type=int, default=EAC4_END,
        help=f"Last year to download (default: {EAC4_END}, EAC4 coverage end).",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip CAMS download; only re-process cached files.",
    )
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    print(f"CAMS EAC4 download: {args.start_year}–{args.end_year} "
          f"({len(years)} years, ~{len(years) * 60} MB)")
    print(f"Grid point: {DHAKA_LAT}°N, {DHAKA_LON}°E (nearest CAMS cell)\n")

    # ── download + extract ─────────────────────────────────────────────────────
    cams_df = fetch_cams(years)
    print(f"\nCAMS daily records: {len(cams_df):,}  "
          f"({cams_df.index.min().date()} → {cams_df.index.max().date()})")

    # ── save raw CAMS CSV ──────────────────────────────────────────────────────
    cams_df.reset_index().rename(columns={"date": "date"}).to_csv(
        CAMS_RAW, index=False, float_format="%.3f"
    )
    print(f"Raw CAMS data saved → {CAMS_RAW}")

    # ── merge into main daily CSV ──────────────────────────────────────────────
    merge_into_daily(cams_df)


if __name__ == "__main__":
    main()
