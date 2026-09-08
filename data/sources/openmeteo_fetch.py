#!/usr/bin/env python3
"""Fetch hourly weather series from the Open-Meteo ERA5 archive (no API key).

Why this source: the benchmark has no hourly group otherwise, hourly is where
foundation models are claimed to be strongest, and ERA5 reanalysis is
published continuously — so a 2026 test window sits after every model's
release date (docs/dataset-plan.md §1).

Selection rule, fixed and stated so it cannot be tuned to a model: the capital
or largest city of each of the CITIES entries below — chosen once to span
every inhabited continent and the main Köppen climate zones — crossed with
three variables (temperature, wind speed, relative humidity). No station is
added or dropped after seeing results.

  python data/sources/openmeteo_fetch.py                 # 2024-01-01 -> yesterday
  python data/sources/openmeteo_fetch.py --start 2025-01-01
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, os, ssl, sys, time, urllib.parse, urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd

API = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["temperature_2m", "wind_speed_10m", "relative_humidity_2m"]
START = "2024-01-01"
BATCH = 20          # locations per request; the API accepts comma-separated lists
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openmeteo")
UA = "tsfm-bench/0.1 (research benchmark)"

# name, latitude, longitude
CITIES: list[tuple[str, float, float]] = [
    ("Abu_Dhabi", 24.45, 54.38), ("Accra", 5.60, -0.19), ("Addis_Ababa", 9.02, 38.75),
    ("Adelaide", -34.93, 138.60), ("Algiers", 36.75, 3.06), ("Amman", 31.95, 35.93),
    ("Amsterdam", 52.37, 4.90), ("Anchorage", 61.22, -149.90), ("Ankara", 39.93, 32.87),
    ("Astana", 51.17, 71.43), ("Athens", 37.98, 23.73), ("Atlanta", 33.75, -84.39),
    ("Auckland", -36.85, 174.76), ("Baghdad", 33.32, 44.36), ("Baku", 40.41, 49.87),
    ("Bangkok", 13.76, 100.50), ("Barcelona", 41.39, 2.17), ("Beijing", 39.90, 116.41),
    ("Beirut", 33.89, 35.50), ("Belgrade", 44.79, 20.45), ("Berlin", 52.52, 13.40),
    ("Bogota", 4.71, -74.07), ("Boston", 42.36, -71.06), ("Brasilia", -15.79, -47.88),
    ("Brisbane", -27.47, 153.03), ("Brussels", 50.85, 4.35), ("Bucharest", 44.43, 26.10),
    ("Budapest", 47.50, 19.04), ("Buenos_Aires", -34.60, -58.38), ("Cairo", 30.04, 31.24),
    ("Calgary", 51.05, -114.07), ("Cape_Town", -33.92, 18.42), ("Caracas", 10.48, -66.90),
    ("Casablanca", 33.57, -7.59), ("Chennai", 13.08, 80.27), ("Chicago", 41.88, -87.63),
    ("Colombo", 6.93, 79.86), ("Copenhagen", 55.68, 12.57), ("Dakar", 14.72, -17.47),
    ("Dallas", 32.78, -96.80), ("Damascus", 33.51, 36.28), ("Dar_es_Salaam", -6.79, 39.21),
    ("Delhi", 28.61, 77.21), ("Denver", 39.74, -104.98), ("Dhaka", 23.81, 90.41),
    ("Doha", 25.29, 51.53), ("Dubai", 25.20, 55.27), ("Dublin", 53.35, -6.26),
    ("Edinburgh", 55.95, -3.19), ("Frankfurt", 50.11, 8.68), ("Guatemala_City", 14.63, -90.51),
    ("Hanoi", 21.03, 105.85), ("Harare", -17.83, 31.05), ("Havana", 23.11, -82.37),
    ("Helsinki", 60.17, 24.94), ("Ho_Chi_Minh_City", 10.82, 106.63), ("Hong_Kong", 22.32, 114.17),
    ("Honolulu", 21.31, -157.86), ("Houston", 29.76, -95.37), ("Islamabad", 33.68, 73.05),
    ("Istanbul", 41.01, 28.98), ("Jakarta", -6.21, 106.85), ("Jeddah", 21.49, 39.19),
    ("Johannesburg", -26.20, 28.05), ("Kabul", 34.56, 69.21), ("Karachi", 24.86, 67.01),
    ("Khartoum", 15.50, 32.56), ("Kiev", 50.45, 30.52), ("Kinshasa", -4.44, 15.27),
    ("Kolkata", 22.57, 88.36), ("Kuala_Lumpur", 3.14, 101.69), ("Kuwait_City", 29.38, 47.99),
    ("Lagos", 6.52, 3.38), ("Lahore", 31.55, 74.34), ("Lima", -12.05, -77.04),
    ("Lisbon", 38.72, -9.14), ("London", 51.51, -0.13), ("Los_Angeles", 34.05, -118.24),
    ("Madrid", 40.42, -3.70), ("Manila", 14.60, 120.98), ("Melbourne", -37.81, 144.96),
    ("Mexico_City", 19.43, -99.13), ("Miami", 25.76, -80.19), ("Milan", 45.46, 9.19),
    ("Minneapolis", 44.98, -93.27), ("Montreal", 45.50, -73.57), ("Moscow", 55.76, 37.62),
    ("Mumbai", 19.08, 72.88), ("Munich", 48.14, 11.58), ("Nairobi", -1.29, 36.82),
    ("New_York", 40.71, -74.01), ("Osaka", 34.69, 135.50), ("Oslo", 59.91, 10.75),
    ("Ottawa", 45.42, -75.70), ("Panama_City", 8.98, -79.52), ("Paris", 48.86, 2.35),
    ("Perth", -31.95, 115.86), ("Phoenix", 33.45, -112.07), ("Prague", 50.08, 14.44),
    ("Pretoria", -25.75, 28.19), ("Reykjavik", 64.15, -21.94), ("Riga", 56.95, 24.11),
    ("Rio_de_Janeiro", -22.91, -43.17), ("Riyadh", 24.71, 46.68), ("Rome", 41.90, 12.50),
    ("Santiago", -33.45, -70.67), ("Sao_Paulo", -23.55, -46.63), ("Seattle", 47.61, -122.33),
    ("Seoul", 37.57, 126.98), ("Shanghai", 31.23, 121.47), ("Singapore", 1.35, 103.82),
    ("Sofia", 42.70, 23.32), ("Stockholm", 59.33, 18.07), ("Sydney", -33.87, 151.21),
    ("Taipei", 25.03, 121.57), ("Tashkent", 41.30, 69.24), ("Tehran", 35.69, 51.39),
    ("Tokyo", 35.68, 139.65), ("Toronto", 43.65, -79.38), ("Tunis", 36.81, 10.18),
    ("Ulaanbaatar", 47.89, 106.91), ("Vancouver", 49.28, -123.12), ("Vienna", 48.21, 16.37),
    ("Warsaw", 52.23, 21.01), ("Wellington", -41.29, 174.78), ("Winnipeg", 49.90, -97.14),
    ("Zurich", 47.38, 8.54),
]


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


_CTX = _ctx()


def _get(url: str, retries: int = 5) -> dict | list:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120, context=_CTX) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    print(f"  request failed: {type(last).__name__}: {str(last)[:140]}", file=sys.stderr)
    return {}


def fetch_batch(chunk, start: str, end: str) -> dict[str, pd.Series]:
    q = {
        "latitude": ",".join(f"{c[1]}" for c in chunk),
        "longitude": ",".join(f"{c[2]}" for c in chunk),
        "start_date": start, "end_date": end,
        "hourly": ",".join(VARIABLES), "timezone": "UTC",
    }
    data = _get(f"{API}?{urllib.parse.urlencode(q)}")
    if isinstance(data, dict):
        data = [data]
    out: dict[str, pd.Series] = {}
    for (name, _, _), loc in zip(chunk, data):
        h = loc.get("hourly") or {}
        idx = pd.to_datetime(h.get("time", []))
        for var in VARIABLES:
            vals = h.get(var)
            if vals is None or len(vals) != len(idx):
                continue
            out[f"{name}__{var}"] = pd.Series(vals, index=idx, dtype="float64")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    series: dict[str, pd.Series] = {}
    for i in range(0, len(CITIES), args.batch):
        chunk = CITIES[i:i + args.batch]
        series.update(fetch_batch(chunk, args.start, end))
        print(f"  {min(i + args.batch, len(CITIES))}/{len(CITIES)} cities, {len(series)} series", flush=True)
        time.sleep(1.0)

    df = pd.DataFrame(series).sort_index()
    df.index.name = "time"
    before = df.shape[1]
    df = df.dropna(axis=1, thresh=int(len(df) * 0.98))    # drop series with real gaps
    df = df.ffill(limit=3).dropna(axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "weather_hourly.csv.gz")
    with gzip.open(out, "wt") as fh:
        df.to_csv(fh)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    json.dump({
        "source": "Open-Meteo ERA5 archive (no API key)", "variables": VARIABLES,
        "n_cities": len(CITIES), "start": args.start, "end": end,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "n_series": int(df.shape[1]), "dropped_incomplete": before - int(df.shape[1]),
        "sha256": sha,
    }, open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=2)
    print(f"wrote {out}: {df.shape[1]} series x {df.shape[0]} hours; sha256 {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
