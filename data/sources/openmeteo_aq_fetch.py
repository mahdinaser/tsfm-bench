#!/usr/bin/env python3
"""Fetch hourly air-quality series from Open-Meteo's CAMS archive (no API key).

Why this source: weather is smooth and strongly periodic, which is the easy end
of hourly forecasting. Air quality shares the same sampling rate and the same
daily cycle but is spiky and heavy-tailed — pollution episodes arrive as sudden
level shifts that no amount of seasonal structure predicts. Having both lets the
paper separate "foundation models are good at hourly data" from "foundation
models are good at smooth data", which the two-domain version of this benchmark
could not distinguish.

Selection rule, fixed before any result was seen: the same city list as
openmeteo_fetch.py, crossed with three pollutants. No city is added or dropped
after seeing results.

  python data/sources/openmeteo_aq_fetch.py                # 2024-01-01 -> yesterday
  python data/sources/openmeteo_aq_fetch.py --start 2025-01-01
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, os, sys, time, urllib.parse
from datetime import date, datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openmeteo_fetch import CITIES, _get, OUT_DIR as WEATHER_OUT  # noqa: E402

API = "https://air-quality-api.open-meteo.com/v1/air-quality"
VARIABLES = ["pm10", "pm2_5", "nitrogen_dioxide"]
START = "2024-01-01"
BATCH = 20
OUT_DIR = os.path.join(os.path.dirname(WEATHER_OUT), "openmeteo_aq")


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
        h = (loc or {}).get("hourly") or {}
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
    df = df.dropna(axis=1, thresh=int(len(df) * 0.98))
    df = df.ffill(limit=3).dropna(axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "airquality_hourly.csv.gz")
    with gzip.open(out, "wt") as fh:
        df.to_csv(fh)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    json.dump({
        "source": "Open-Meteo air-quality archive, CAMS reanalysis (no API key)",
        "variables": VARIABLES, "n_cities": len(CITIES),
        "start": args.start, "end": end,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "n_series": int(df.shape[1]), "dropped_incomplete": before - int(df.shape[1]),
        "sha256": sha,
    }, open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=2)
    print(f"wrote {out}: {df.shape[1]} series x {df.shape[0]} hours; sha256 {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
