#!/usr/bin/env python3
"""
fetch_daily_aqi.py
------------------
Builds a daily Dhaka AQI dataset by combining two sources:

  1. aqi.in scrape        — actual daily AQI, 2020-03-19 → present
  2. existing monthly CSV — expanded to per-day rows for 2017-01-01 → 2019-12-31
                            (and monthly-mean pollutant columns for 2020-2025)

Run from the project root:

    python scripts/fetch_daily_aqi.py

With a WAQI token (free at https://aqicn.org/api/) to replace monthly-mean
pollutant columns with actual daily concentrations (PM2.5, PM10, CO, NO₂,
SO₂, O₃) for all years:

    python scripts/fetch_daily_aqi.py --waqi-token YOUR_TOKEN_HERE

Output
------
data/daily_dhaka_aqi_dataset.csv

Columns (always present)
    date, year, month, aqi, pm25, pm10, no2, so2, season, source

Columns (with --waqi-token only)
    co, o3  (daily, all years)
    pm25, pm10, no2, so2 are replaced with actual daily values for 2017+

source values
    "monthly_avg"   — 2017-2019: AQI and pollutants from monthly mean
    "daily_aqi"     — 2020+: AQI is actual daily; pollutants are monthly
                      mean (or actual daily when --waqi-token supplied)
"""

import argparse
import calendar
import csv
import json
import pathlib
import ssl
import time
import urllib.request
from datetime import date, timedelta

# ── constants ─────────────────────────────────────────────────────────────────

AQI_URL = (
    "https://www.aqi.in/us/dashboard/bangladesh/"
    "dhaka-division/dhaka/historical-analysis"
)
WAQI_FEED   = "https://api.waqi.info/feed/dhaka/?token={token}"
WAQI_HIST   = (
    "https://api.waqi.info/api/feed/@{uid}/obs.day.json"
    "?token={token}&date={date}"
)
WAQI_SEARCH = "https://api.waqi.info/search/?token={token}&keyword=dhaka+bangladesh"

MONTHLY_CSV = pathlib.Path("data/final_dhaka_aqi_dataset_clean.csv")
OUTPUT      = pathlib.Path("data/daily_dhaka_aqi_dataset.csv")

SEASON_MAP = {
    12: "Winter",    1: "Winter",       2: "Winter",
    3:  "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6:  "Monsoon",   7: "Monsoon",      8: "Monsoon",    9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}


# ── HTTP helper ───────────────────────────────────────────────────────────────

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def _get(url: str, *, retries: int = 3, delay: float = 2.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, context=_ctx, timeout=30) as r:
                return r.read()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt + 1}/{retries} after error: {exc}")
            time.sleep(delay)


# ── monthly CSV → daily expansion ─────────────────────────────────────────────

def load_monthly_lookup(path: pathlib.Path) -> dict[tuple[int, int], dict]:
    """
    Read the existing monthly CSV and return a {(year, month): row_dict} map.
    Keeps only the mean pollutant and AQI columns needed here.
    """
    keep = {"pm25_mean", "pm10_mean", "no2_mean", "so2_mean", "aqi_mean"}
    lookup: dict[tuple[int, int], dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yr = int(row["year"])
            mo = int(row["month"])
            lookup[(yr, mo)] = {k: _float(row.get(k)) for k in keep}
    return lookup


def _float(v) -> float | None:
    try:
        return float(v) if v not in (None, "", "nan") else None
    except (TypeError, ValueError):
        return None


def expand_monthly_to_daily(
    lookup: dict[tuple[int, int], dict],
    year_range: range,
) -> list[dict]:
    """
    For each calendar day in year_range, emit one row whose AQI and pollutant
    values equal the monthly mean from the monthly CSV.
    Days with no entry in the lookup are skipped.
    """
    rows = []
    for yr in year_range:
        for mo in range(1, 13):
            entry = lookup.get((yr, mo))
            if entry is None:
                continue
            _, n_days = calendar.monthrange(yr, mo)
            for day in range(1, n_days + 1):
                d = date(yr, mo, day)
                rows.append({
                    "date":   d.isoformat(),
                    "year":   yr,
                    "month":  mo,
                    "aqi":    entry["aqi_mean"],
                    "pm25":   entry["pm25_mean"],
                    "pm10":   entry["pm10_mean"],
                    "no2":    entry["no2_mean"],
                    "so2":    entry["so2_mean"],
                    "season": SEASON_MAP[mo],
                    "source": "monthly_avg",
                })
    return rows


# ── aqi.in scraper ────────────────────────────────────────────────────────────

def _extract_annual_data(html: str) -> list[dict]:
    """
    Pull the RSC-embedded annual daily AQI dataset out of the page HTML.
    Structure (double-JSON-escaped in the RSC payload):
      [{year, yearAvg, months: [{month, monthAvg, days: [{day, value}]}]}]
    """
    bs = chr(92)
    q = '"'
    marker = bs + q + "data" + bs + q + ":["

    candidates: list[tuple[int, str]] = []
    start = 0
    while True:
        idx = html.find(marker, start)
        if idx < 0:
            break
        arr_open = html.find("[", idx + len(marker) - 1)
        if arr_open < 0:
            break
        pos = arr_open + 1
        depth = 1
        while pos < len(html) and depth > 0:
            c = html[pos]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            pos += 1
        raw = html[arr_open + 1 : pos - 1]
        candidates.append((len(raw), raw))
        start = idx + 1

    if not candidates:
        raise RuntimeError(
            "Could not locate embedded data in aqi.in HTML. "
            "The site layout may have changed."
        )

    _, raw = max(candidates)  # annual dataset is the largest array (~300 KB)
    cleaned = raw.replace(bs + q, q)
    return json.loads("[" + cleaned + "]")


def fetch_aqi_in() -> dict[str, int]:
    """Return {date_str: aqi_int} for all days on aqi.in (2020 → now)."""
    print("Fetching aqi.in historical page …")
    html = _get(AQI_URL).decode("utf-8", errors="ignore")
    print(f"  Page size : {len(html):,} bytes")

    annual = _extract_annual_data(html)
    years = sorted(b.get("year") for b in annual if isinstance(b, dict))
    print(f"  Years found : {years}")

    records: dict[str, int] = {}
    for year_block in annual:
        if not isinstance(year_block, dict) or "months" not in year_block:
            continue
        for month_block in year_block["months"]:
            for day_block in month_block.get("days", []):
                d = day_block.get("day")
                v = day_block.get("value")
                if d and v is not None:
                    records[d] = int(v)

    print(f"  Daily AQI records : {len(records):,}")
    return records


# ── WAQI API (actual daily pollutants for all years) ──────────────────────────

def _waqi_station_uid(token: str) -> str | None:
    """Find the numeric station UID for the Dhaka US-Consulate monitor."""
    url = WAQI_SEARCH.format(token=token)
    data = json.loads(_get(url))
    if data.get("status") != "ok":
        return None
    for entry in data.get("data", []):
        station = entry.get("station", {})
        name    = station.get("name", "").lower()
        country = station.get("country", "").upper()
        if country == "BD" and "dhaka" in name:
            return str(entry.get("uid"))
    return None


def fetch_waqi_historical(token: str, start_year: int = 2017) -> dict[str, dict]:
    """
    Fetch daily pollutant concentrations from the WAQI historical API.
    Returns {date_str: {pm25, pm10, co, no2, so2, o3}}.

    The /obs.day.json endpoint returns a window of observations around the
    requested date; we page through in 25-day steps to cover the full range.
    """
    print("\nFetching WAQI pollutant history …")

    uid = _waqi_station_uid(token)
    if uid is None:
        print("  Warning: Dhaka station not found via WAQI search.")
        print("  Trying current feed as a fallback (today only) …")
        resp = json.loads(_get(WAQI_FEED.format(token=token)))
        if resp.get("status") == "ok":
            iaqi  = resp["data"].get("iaqi", {})
            today = str(date.today())
            return {today: {k: iaqi.get(k, {}).get("v") for k in ("pm25","pm10","co","no2","so2","o3")}}
        return {}

    print(f"  Station UID : {uid}")

    results: dict[str, dict] = {}
    step   = timedelta(days=25)
    cursor = date(start_year, 1, 1)
    end    = date.today()
    total  = max(1, (end - cursor).days // step.days + 1)
    done   = 0

    while cursor <= end:
        url = WAQI_HIST.format(uid=uid, token=token, date=cursor.isoformat())
        try:
            payload = json.loads(_get(url))
            for entry in payload.get("rxs", {}).get("obs", []):
                msg     = entry.get("msg", {})
                day_str = msg.get("t", "")[:10]
                if not day_str:
                    continue
                iaqi = msg.get("iaqi", {})
                row  = {k: (iaqi.get(k, [None])[0]) for k in ("pm25","pm10","co","no2","so2","o3")}
                if any(v is not None for v in row.values()):
                    results[day_str] = row
        except Exception as exc:
            print(f"  Warning: {cursor} → {exc}")

        cursor += step
        done   += 1
        if done % 20 == 0:
            print(f"  Progress : {min(100, done * 100 // total)}%  ({len(results)} days)")
        time.sleep(0.4)

    print(f"  Total pollutant records : {len(results):,}")
    return results


# ── merge & write ─────────────────────────────────────────────────────────────

def build_dataset(
    aqi_in_data:    dict[str, int],
    monthly_lookup: dict[tuple[int, int], dict],
    waqi_data:      dict[str, dict] | None,
) -> list[dict]:
    """
    Merge all sources into a sorted list of daily rows.

    Priority / logic:
      AQI    : aqi.in daily value if available, else monthly mean
      pm25/pm10/no2/so2 :
               WAQI daily value if available, else monthly mean from CSV
      co/o3  : WAQI daily value only (not in existing monthly CSV)
      source : "monthly_avg"  if AQI came from monthly mean (2017-2019)
               "daily_aqi"    if AQI is an actual daily observation (2020+)
    """
    # All dates: actual daily (2020+) plus every calendar day in monthly data
    daily_dates: set[str] = set(aqi_in_data.keys())

    for (yr, mo), _ in monthly_lookup.items():
        _, n = calendar.monthrange(yr, mo)
        for day in range(1, n + 1):
            daily_dates.add(date(yr, mo, day).isoformat())

    # Only add co/o3 columns if the WAQI data actually contains those values
    _has_co_o3 = waqi_data is not None and any(
        row.get("co") is not None or row.get("o3") is not None
        for row in waqi_data.values()
    )

    rows = []
    for d in sorted(daily_dates):
        try:
            dt = date.fromisoformat(d)
        except ValueError:
            continue

        mon = monthly_lookup.get((dt.year, dt.month), {})
        waqi_row = (waqi_data or {}).get(d, {})

        # AQI: prefer actual daily (2020+), fall back to monthly mean
        aqi_val = aqi_in_data.get(d) or mon.get("aqi_mean")
        if aqi_val is None:
            continue  # skip days where we have absolutely no AQI estimate

        # Pollutants: prefer WAQI daily, fall back to monthly mean
        pm25 = waqi_row.get("pm25") or mon.get("pm25_mean")
        pm10 = waqi_row.get("pm10") or mon.get("pm10_mean")
        no2  = waqi_row.get("no2")  or mon.get("no2_mean")
        so2  = waqi_row.get("so2")  or mon.get("so2_mean")
        co   = waqi_row.get("co")   # only from WAQI
        o3   = waqi_row.get("o3")   # only from WAQI

        source = "daily_aqi" if d in aqi_in_data else "monthly_avg"

        row: dict = {
            "date":   d,
            "year":   dt.year,
            "month":  dt.month,
            "aqi":    round(float(aqi_val), 2) if aqi_val is not None else None,
            "pm25":   round(float(pm25), 2)    if pm25 is not None else None,
            "pm10":   round(float(pm10), 2)    if pm10 is not None else None,
            "no2":    round(float(no2), 2)     if no2 is not None  else None,
            "so2":    round(float(so2), 2)     if so2 is not None  else None,
            "season": SEASON_MAP[dt.month],
            "source": source,
        }
        # Only include co/o3 if the WAQI pull actually returned values
        if _has_co_o3:
            row["co"] = round(float(co), 2) if co is not None else None
            row["o3"] = round(float(o3), 2) if o3 is not None else None

        rows.append(row)

    return rows


def write_csv(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["date"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build daily Dhaka AQI dataset (aqi.in + monthly CSV expansion)."
    )
    parser.add_argument(
        "--waqi-token", metavar="TOKEN", default=None,
        help="Free WAQI token (https://aqicn.org/api/). Adds CO & O₃ columns "
             "and replaces monthly-mean pollutants with actual daily values.",
    )
    parser.add_argument(
        "--start-year", type=int, default=2017,
        help="Earliest year when using --waqi-token (default: 2017).",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT),
        help=f"Output path (default: {OUTPUT}).",
    )
    args = parser.parse_args()
    out  = pathlib.Path(args.output)

    # ── monthly CSV ────────────────────────────────────────────────────────────
    if not MONTHLY_CSV.exists():
        raise FileNotFoundError(f"Monthly CSV not found at '{MONTHLY_CSV}'.")
    print(f"Loading monthly CSV : {MONTHLY_CSV}")
    monthly_lookup = load_monthly_lookup(MONTHLY_CSV)
    print(f"  Monthly entries   : {len(monthly_lookup)}  "
          f"({min(k[0] for k in monthly_lookup)}–{max(k[0] for k in monthly_lookup)})")

    # ── aqi.in: daily AQI 2020 → present ──────────────────────────────────────
    aqi_in_data = fetch_aqi_in()

    # ── WAQI: actual daily pollutants (optional) ───────────────────────────────
    waqi_data = None
    if args.waqi_token:
        waqi_data = fetch_waqi_historical(args.waqi_token, args.start_year)

    # ── merge & save ───────────────────────────────────────────────────────────
    rows = build_dataset(aqi_in_data, monthly_lookup, waqi_data)
    write_csv(rows, out)

    n_daily   = sum(1 for r in rows if r["source"] == "daily_aqi")
    n_monthly = sum(1 for r in rows if r["source"] == "monthly_avg")

    print(f"\nWrote {len(rows):,} rows → {out}")
    print(f"  {n_daily:,} rows  — actual daily AQI  (2020+, from aqi.in)")
    print(f"  {n_monthly:,} rows  — monthly-mean expanded (2017–2019, from monthly CSV)")
    print(f"  Date range : {rows[0]['date']}  →  {rows[-1]['date']}")
    print(f"  Columns    : {', '.join(rows[0].keys())}")

    if not args.waqi_token:
        print(
            "\nTip: add --waqi-token TOKEN to replace monthly-mean pollutant\n"
            "     values with actual daily concentrations for all years, and\n"
            "     add CO and O₃ columns.  Free token: https://aqicn.org/api/"
        )


if __name__ == "__main__":
    main()
