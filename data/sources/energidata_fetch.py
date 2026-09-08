#!/usr/bin/env python3
"""Fetch hourly Danish electricity series from Energi Data Service (no API key).

Why this source: electricity load is the domain the foundation-model papers
most often claim, usually on ETTh/ETTm — datasets from 2016-2018 that are in
every pretraining corpus, so a strong number there says little. Energinet
publishes the Danish grid openly and continuously, which gives the same kind of
series (strong daily and weekly cycles, weather-driven renewables, occasional
negative prices) with a 2026 hold-out that postdates every model release.

Selection rule, fixed in advance: both Danish price areas (DK1, DK2) crossed
with the settlement quantities below. Nothing is chosen after seeing results.

  python data/sources/energidata_fetch.py            # 2024-01-01 -> yesterday
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, os, ssl, sys, time, urllib.parse, urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd

API = "https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement"
START = "2024-01-01"
UA = "tsfm-bench/0.1 (research benchmark)"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "energidata")
PAGE = 20000

# Every settlement quantity the dataset reports, rather than a hand-picked
# few: total load, each generation technology, the cross-border flows and the
# grid losses. Picking six by hand both thinned the panel to ten series and
# quietly dropped one whose real column name differed from my guess. Taking the
# lot means the panel is defined by the publisher, not by me, and it spans the
# range of behaviour that makes electricity interesting — a smooth load curve,
# solar that is zero every night, wind that is pure weather, and flows that
# change sign.
SKIP = {"HourUTC", "HourDK", "PriceArea"}


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


_CTX = _ctx()


def _get(url: str, retries: int = 5) -> dict:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    args = ap.parse_args()

    end = date.today().strftime("%Y-%m-%d")
    records: list[dict] = []
    offset = 0
    while True:
        q = {"start": f"{args.start}T00:00", "end": f"{end}T00:00",
             "limit": PAGE, "offset": offset, "sort": "HourUTC asc"}
        page = _get(f"{API}?{urllib.parse.urlencode(q)}").get("records") or []
        if not page:
            break
        records.extend(page)
        offset += len(page)
        print(f"  {offset} records", flush=True)
        if len(page) < PAGE:
            break
        time.sleep(0.5)

    if not records:
        print("no records returned", file=sys.stderr)
        return 1

    raw = pd.DataFrame.from_records(records)
    raw["HourUTC"] = pd.to_datetime(raw["HourUTC"])
    series: dict[str, pd.Series] = {}
    for area, block in raw.groupby("PriceArea"):
        # A settlement hour can appear as several municipality rows; the grid
        # total is their sum, which is what a load forecast is actually for.
        numeric = [c for c in block.columns
                   if c not in SKIP and pd.api.types.is_numeric_dtype(block[c])]
        agg = block.groupby("HourUTC")[numeric].sum()
        for col in agg.columns:
            series[f"{area}__{col}"] = agg[col].astype("float64")

    df = pd.DataFrame(series).sort_index()
    df.index.name = "time"
    before = df.shape[1]
    df = df.dropna(axis=1, thresh=int(len(df) * 0.98))
    df = df.ffill(limit=3).dropna(axis=1)
    # A column that never moves carries no forecasting problem at all.
    df = df.loc[:, df.std(numeric_only=True) > 0]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "energy_hourly.csv.gz")
    with gzip.open(out, "wt") as fh:
        df.to_csv(fh)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    json.dump({
        "source": "Energi Data Service, ProductionConsumptionSettlement (no API key)",
        "columns": sorted(c.split("__", 1)[1] for c in df.columns[:len(df.columns)//2]) or None,
        "start": args.start, "end": end,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "n_series": int(df.shape[1]), "dropped_incomplete": before - int(df.shape[1]),
        "sha256": sha,
    }, open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=2)
    print(f"wrote {out}: {df.shape[1]} series x {df.shape[0]} hours; sha256 {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
