#!/usr/bin/env python3
"""Fetch daily reference exchange rates from the ECB's SDMX API (no API key).

Why this source: it is the adversarial case. Exchange rates are close to a
random walk, and the standing result in forecasting is that nothing reliably
beats a naive forecast on them. A benchmark made only of series with strong
seasonality tells you how well a model reads a daily cycle; adding a domain
where the honest answer is "no method helps" is what makes a claim about *when*
foundation models are worth their cost falsifiable rather than decorative.

The ECB publishes these at 16:00 CET every TARGET day and never revises them,
so the 2026 hold-out cannot have leaked into any 2024-2025 pretraining corpus.

Selection rule, fixed in advance: every currency in the ECB's daily reference
list, quoted against the euro. Nothing is dropped afterwards except series with
missing observations in the window.

  python data/sources/ecb_fx_fetch.py               # 2024-01-01 -> yesterday
"""
from __future__ import annotations

import argparse, csv, gzip, hashlib, io, json, os, ssl, sys, time, urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd

API = "https://data-api.ecb.europa.eu/service/data/EXR"
START = "2024-01-01"
UA = "tsfm-bench/0.1 (research benchmark)"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ecb")

# The ECB's daily reference currencies. Listed explicitly so the panel is
# reproducible even if the ECB adds or retires one later.
CURRENCIES = [
    "AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "GBP", "HKD",
    "HUF", "IDR", "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR", "NOK",
    "NZD", "PHP", "PLN", "RON", "SEK", "SGD", "THB", "TRY", "USD", "ZAR",
]


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


_CTX = _ctx()


def _get_csv(url: str, retries: int = 5) -> str:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/csv"})
            with urllib.request.urlopen(req, timeout=120, context=_CTX) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    print(f"  request failed: {type(last).__name__}: {str(last)[:140]}", file=sys.stderr)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    series: dict[str, pd.Series] = {}
    for i in range(0, len(CURRENCIES), args.batch):
        chunk = CURRENCIES[i:i + args.batch]
        key = f"D.{'+'.join(chunk)}.EUR.SP00.A"
        url = (f"{API}/{key}?startPeriod={args.start}&endPeriod={end}&format=csvdata")
        text = _get_csv(url)
        if not text:
            continue
        rows = list(csv.DictReader(io.StringIO(text)))
        frame: dict[str, dict[str, float]] = {}
        for row in rows:
            cur, when, val = row.get("CURRENCY"), row.get("TIME_PERIOD"), row.get("OBS_VALUE")
            if not cur or not when or not val:
                continue
            try:
                frame.setdefault(f"EURTO{cur}", {})[when] = float(val)
            except ValueError:
                continue
        for name, points in frame.items():
            s = pd.Series(points, dtype="float64")
            s.index = pd.to_datetime(s.index)
            series[name] = s.sort_index()
        print(f"  {min(i + args.batch, len(CURRENCIES))}/{len(CURRENCIES)} currencies, {len(series)} series", flush=True)
        time.sleep(1.0)

    df = pd.DataFrame(series).sort_index()
    df.index.name = "date"
    before = df.shape[1]
    # Business-day index with no weekend rows; a currency missing more than a
    # couple of fixings in the window is dropped rather than interpolated.
    df = df.dropna(axis=1, thresh=int(len(df) * 0.98))
    df = df.ffill(limit=3).dropna(axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fx_daily.csv.gz")
    with gzip.open(out, "wt") as fh:
        df.to_csv(fh)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    json.dump({
        "source": "ECB SDMX data API, daily euro reference rates (no API key)",
        "n_currencies": len(CURRENCIES), "start": args.start, "end": end,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "n_series": int(df.shape[1]), "dropped_incomplete": before - int(df.shape[1]),
        "sha256": sha,
    }, open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=2)
    print(f"wrote {out}: {df.shape[1]} series x {df.shape[0]} days; sha256 {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
