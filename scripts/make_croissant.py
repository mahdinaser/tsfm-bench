#!/usr/bin/env python3
"""Emit the Croissant record the NeurIPS Evaluations & Datasets track requires.

Generated rather than hand-written, because the interesting fields -- series
counts, date ranges, checksums -- already exist in the manifest each fetcher
writes, and a hand-copied checksum is a checksum that will be wrong by the next
re-fetch.

  python scripts/make_croissant.py
"""
from __future__ import annotations

import datetime, json, os, pathlib, sys

SOURCES = {
    "wiki_daily": ("data/wikimedia/manifest.json",
                   "Wikipedia pageviews, daily (weekly and monthly groups are aggregated from this file)",
                   "https://wikimedia.org/api/rest_v1/"),
    "weather_hourly": ("data/openmeteo/manifest.json",
                       "ERA5 reanalysis weather, hourly",
                       "https://archive-api.open-meteo.com/v1/archive"),
    "airquality_hourly": ("data/openmeteo_aq/manifest.json",
                          "CAMS air quality, hourly",
                          "https://air-quality-api.open-meteo.com/v1/air-quality"),
    "energy_hourly": ("data/energidata/manifest.json",
                      "Danish grid production and consumption settlement, hourly",
                      "https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement"),
    "fx_daily": ("data/ecb/manifest.json",
                 "European Central Bank euro reference exchange rates, daily",
                 "https://data-api.ecb.europa.eu/service/data/EXR"),
}

CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "sc": "https://schema.org/",
    "column": "cr:column",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileObject": "cr:fileObject",
    "recordSet": "cr:recordSet",
    "source": "cr:source",
}


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    dists, records, missing = [], [], []

    for name, (manifest_path, desc, api) in SOURCES.items():
        entry = {
            "@type": "cr:FileObject",
            "@id": f"{name}.csv.gz",
            "name": f"{name}.csv.gz",
            "description": desc,
            "contentUrl": api,
            "encodingFormat": "application/gzip",
        }
        manifest = root / manifest_path
        if manifest.exists():
            m = json.loads(manifest.read_text())
            if m.get("sha256"):
                entry["sha256"] = m["sha256"]
            entry["description"] += (
                f". {m.get('n_series', '?')} series covering {m.get('start', '?')} to "
                f"{m.get('end', '?')}; retrieved {str(m.get('retrieved_at', '?'))[:10]}"
            )
        else:
            missing.append(manifest_path)
        dists.append(entry)

        records.append({
            "@type": "cr:RecordSet",
            "@id": name,
            "name": name,
            "description": f"Wide format: one column per series, first column the timestamp. {desc}.",
            "field": [
                {"@type": "cr:Field", "@id": f"{name}/timestamp", "name": "timestamp",
                 "dataType": "sc:DateTime",
                 "description": "Observation time, UTC.",
                 "source": {"fileObject": {"@id": f"{name}.csv.gz"},
                            "extract": {"column": "time"}}},
                {"@type": "cr:Field", "@id": f"{name}/value", "name": "value",
                 "dataType": "sc:Float",
                 "description": "Observed value for one series at one timestamp.",
                 "source": {"fileObject": {"@id": f"{name}.csv.gz"}}},
            ],
        })

    croissant = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "tsfm-bench-holdout-2026",
        "description": (
            "A contamination-free forecasting benchmark: seven groups drawn from five "
            "domains, with every test observation published after the release of every "
            "model evaluated against it. Distributed as retrieval code rather than as a "
            "frozen archive, so that the panel is rebuilt from the publisher and the test "
            "window keeps moving forward -- a frozen file would be contaminated as soon as "
            "the next generation of models was trained on it."
        ),
        "license": ("Code: MIT. Data: not redistributed here -- each publisher's own "
                    "terms govern reuse of the observations its API returns."),
        "url": "https://github.com/mahdinaser/tsfm-bench",
        "version": "1.0.0",
        "datePublished": datetime.date.today().isoformat(),
        "citeAs": ("Naser Moghadasi, M. and Ghaderi, F. A Later Test Set Is Not a New "
                   "Domain: Pretraining Familiarity Survives a Contamination-Free "
                   "Hold-Out. 2026."),
        "distribution": dists,
        "recordSet": records,
    }

    out = root / "croissant.json"
    out.write_text(json.dumps(croissant, indent=2) + "\n")
    print(f"wrote {out} — {len(dists)} file objects, {len(records)} record sets")
    if missing:
        print("  manifests not found (run the matching fetcher): " + ", ".join(missing),
              file=sys.stderr)
    todos = [k for k, v in croissant.items() if isinstance(v, str) and v.startswith("TODO")]
    if todos:
        print(f"  fields still to fill before submission: {', '.join(todos)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
