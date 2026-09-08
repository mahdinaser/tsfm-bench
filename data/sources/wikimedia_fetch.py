#!/usr/bin/env python3
"""Fetch a reproducible set of English-Wikipedia daily pageview series.

Selection rule (deterministic, no hand-picking): the top-1000 articles of a
fixed reference month by all-access user views, excluding Main_Page and
Special: pages. Popularity in that month is the only criterion, so the set
cannot be tuned to a model's strengths, and any article that lacks a
complete daily history over the fetch window is dropped and reported.

Daily views are then fetched from START through yesterday and written to
data/wikimedia/pageviews_daily.csv.gz (wide: date x article), plus a
manifest with the retrieval timestamp, the selection parameters, dropped
titles and a SHA-256 of the data file. Aggregation to weekly/monthly
happens in tsfm_bench.data, not here, so the raw daily file is the single
source of truth.

  python data/sources/wikimedia_fetch.py            # defaults below
  python data/sources/wikimedia_fetch.py --n 300    # smaller set
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, os, sys, time, urllib.parse, urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
PROJECT = "en.wikipedia"
REFERENCE_MONTH = ("2024", "01")     # selection month: fixed, cited in the paper
START = "20200101"                   # daily history start (YYYYMMDD)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wikimedia")
UA = "tsfm-bench/0.1 (research benchmark; contact via repo)"


def _ssl_context():
    # python.org builds of Python on macOS ship without system CA roots, so a
    # plain urlopen fails with CERTIFICATE_VERIFY_FAILED. certifi is present in
    # the venv (transformers depends on it); use it when available.
    try:
        import certifi, ssl
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


_CTX = _ssl_context()


def _get(url: str, retries: int = 5) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            if e.code == 429 or e.code >= 500:
                last = e
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:  # network / TLS; retry with backoff, then report
            last = e
            time.sleep(2 ** attempt)
    print(f"  request failed after {retries} attempts: {type(last).__name__}: {str(last)[:120]}", file=sys.stderr)
    return {}


def top_articles(n: int) -> list[str]:
    y, m = REFERENCE_MONTH
    data = _get(f"{API}/top/{PROJECT}/all-access/{y}/{m}/all-days")
    items = data.get("items", [{}])[0].get("articles", [])
    titles = []
    for it in items:
        t = it["article"]
        if t == "Main_Page" or t.startswith(("Special:", "Wikipedia:", "File:", "Portal:", "Help:", "Talk:")):
            continue
        titles.append(t)
        if len(titles) >= n:
            break
    return titles


def daily_views(title: str, start: str, end: str) -> pd.Series | None:
    t = urllib.parse.quote(title, safe="")
    data = _get(f"{API}/per-article/{PROJECT}/all-access/user/{t}/daily/{start}/{end}")
    items = data.get("items")
    if not items:
        return None
    s = pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items}, name=title)
    return s.sort_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", default=START)
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to sleep between article requests (the API rate-limits bursts with 429)")
    ap.add_argument("--only-dropped", action="store_true",
                    help="re-fetch just the titles listed as dropped in the manifest and merge into the existing file")
    args = ap.parse_args()

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    out = os.path.join(OUT_DIR, "pageviews_daily.csv.gz")
    prev = json.load(open(manifest_path)) if os.path.exists(manifest_path) else None

    if args.only_dropped:
        if not prev:
            print("no manifest to read dropped titles from", file=sys.stderr)
            return 1
        end = prev["end"]
        titles = list(prev["dropped"])
        existing = pd.read_csv(out, index_col="date", parse_dates=True)
        print(f"re-fetching {len(titles)} dropped titles at {args.delay}s spacing")
    else:
        end = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
        titles = top_articles(args.n)
        existing = None
        print(f"{len(titles)} candidate articles from {PROJECT} top-{args.n}, {REFERENCE_MONTH[0]}-{REFERENCE_MONTH[1]}")

    full_index = pd.date_range(pd.Timestamp(args.start), pd.Timestamp(end), freq="D")
    kept, dropped = {}, []
    for i, title in enumerate(titles, 1):
        s = daily_views(title, args.start, end)
        if s is None or len(s) < len(full_index) * 0.98:
            dropped.append(title)
        else:
            kept[title] = s.reindex(full_index)
        if i % 50 == 0:
            print(f"  {i}/{len(titles)} fetched, {len(kept)} kept", flush=True)
        if args.delay:
            time.sleep(args.delay)

    df = pd.DataFrame(kept)
    if existing is not None:
        df = pd.concat([existing, df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
    df.index.name = "date"
    os.makedirs(OUT_DIR, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        df.to_csv(fh)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    manifest = {
        "project": PROJECT, "reference_month": "-".join(REFERENCE_MONTH),
        "top_n": prev["top_n"] if (args.only_dropped and prev) else args.n,
        "start": args.start, "end": end, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "n_kept": int(df.shape[1]), "dropped": dropped, "sha256": sha,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {out}: {df.shape[1]} series x {df.shape[0]} days; still dropped {len(dropped)}; sha256 {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
