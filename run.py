#!/usr/bin/env python3
"""Run models over benchmark groups and write per-series metrics + timings.

  python run.py --models SeasonalNaive Theta --groups m3_other
  python run.py --models ChronosBoltSmall --groups all
"""
from __future__ import annotations

import argparse, json, os, platform, sys, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsfm_bench.data import GROUPS, load_group
from tsfm_bench.metrics import QUANTILE_LEVELS, coverage, mase_scale, smape, wql
from tsfm_bench.models import get_model

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def score(group, fc, data) -> tuple[pd.DataFrame, dict]:
    """Per-series metrics plus group-level aggregates."""
    h, m = group.horizon, group.seasonality
    rows = []
    for i, sid in enumerate(data.ids):
        te, pt, tr = data.test[i], fc.point[i], data.train[i]
        sc = mase_scale(tr, m)
        rows.append({
            "group": group.name, "series_id": sid,
            "smape": float(np.mean(smape(te, pt))),
            "mase": float(np.mean(np.abs(te - pt)) / sc) if sc else np.nan,
        })
    per_series = pd.DataFrame(rows)

    agg = {
        "group": group.name, "n_series": data.n_series, "horizon": h,
        "seasonality": m,
        "smape": float(per_series["smape"].mean()),
        "mase": float(per_series["mase"].mean()),
    }
    if fc.quantiles is not None:
        agg["wql"] = wql(data.test, fc.quantiles)
        lo = fc.quantiles[..., 0]   # q = 0.1
        hi = fc.quantiles[..., -1]  # q = 0.9
        agg["coverage80"] = coverage(data.test, lo, hi)
    return per_series, agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--groups", nargs="+", required=True,
                    help='group names, or "all"')
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    groups = list(GROUPS) if args.groups == ["all"] else args.groups
    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    timing_path = os.path.join(args.out, "timing.jsonl")

    for model_name in args.models:
        for gname in groups:
            group = GROUPS[gname]
            tag = f"{model_name}__{gname}"
            out_csv = os.path.join(args.out, "metrics", f"{tag}.csv")
            if os.path.exists(out_csv):
                print(f"[skip] {tag} (already have {out_csv})", flush=True)
                continue
            print(f"[run ] {tag} ...", flush=True)
            try:
                data = load_group(gname)
                model = get_model(model_name)
                t0 = time.perf_counter()
                fc = model.run(data.train, group.horizon, group.seasonality)
                wall = time.perf_counter() - t0
                per_series, agg = score(group, fc, data)
                per_series.insert(0, "model", model_name)
                per_series.to_csv(out_csv, index=False)
                rec = {
                    "model": model_name, "group": gname,
                    "model_version": getattr(model, "version", None),
                    "n_series": data.n_series, "horizon": group.horizon,
                    "wall_seconds": wall,
                    "fit_seconds": fc.fit_seconds,
                    "predict_seconds": fc.predict_seconds,
                    "seconds_per_1000_forecasts":
                        wall / (data.n_series * group.horizon) * 1000.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "python": platform.python_version(),
                    **{k: v for k, v in agg.items() if k not in ("group",)},
                }
                with open(timing_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                msg = f"  MASE={agg['mase']:.4f} sMAPE={agg['smape']:.4f}"
                if "wql" in agg:
                    msg += f" WQL={agg['wql']:.4f} cov80={agg['coverage80']:.3f}"
                print(f"{msg}  [{wall:.1f}s]", flush=True)
            except Exception as exc:
                print(f"  FAILED {tag}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
