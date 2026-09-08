#!/usr/bin/env python3
"""Check that the models we call deterministic actually are.

run_seeds.sh re-runs only LightGBM, LSTM and NBEATS, on the claim that
everything else returns the same forecast every time. That claim is load
bearing — if a pretrained model quietly samples, its single reported number is
one draw and every significance test in the paper is understated — so it is
checked rather than asserted. Each model is run twice on one group, under
different seeds, and the point forecasts are compared exactly.

  python scripts/check_determinism.py --group fx_daily
  python scripts/check_determinism.py --group fx_daily --models Chronos2 Moirai2
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODELS = [
    "SeasonalNaive", "Theta", "AutoETS",
    "ChronosBoltSmall", "ChronosBoltBase", "Chronos2",
    "TimesFM", "TimesFM3", "Moirai2",
]


def run_once(model_name: str, group_name: str, seed: int) -> np.ndarray:
    os.environ["TSFM_BENCH_SEED"] = str(seed)
    # Re-import under the new seed: the model modules read it at import time.
    for mod in [m for m in list(sys.modules) if m.startswith("tsfm_bench")]:
        del sys.modules[mod]
    from tsfm_bench.data import GROUPS, load_group
    from tsfm_bench.models import get_model
    group = GROUPS[group_name]
    data = load_group(group_name)
    fc = get_model(model_name).run(data.train, group.horizon, group.seasonality)
    return np.asarray(fc.point, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="fx_daily")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs=2, type=int, default=[20260907, 20260909])
    args = ap.parse_args()
    os.environ.setdefault("TSFM_BENCH_ORIGIN", "2026-01-01")

    report = {}
    for name in args.models:
        try:
            a = run_once(name, args.group, args.seeds[0])
            b = run_once(name, args.group, args.seeds[1])
        except Exception as e:
            print(f"  {name:20} SKIPPED ({type(e).__name__}: {str(e)[:70]})")
            report[name] = {"status": "error", "detail": str(e)[:200]}
            continue
        if a.shape != b.shape:
            verdict, gap = "SHAPE MISMATCH", float("nan")
        else:
            gap = float(np.max(np.abs(a - b)))
            verdict = "deterministic" if gap == 0.0 else "STOCHASTIC"
        print(f"  {name:20} {verdict:14} max|Δ| = {gap:.3e}")
        report[name] = {"status": verdict, "max_abs_diff": gap}

    bad = [m for m, r in report.items() if r.get("status") == "STOCHASTIC"]
    out = os.path.join("results", "holdout_2026", "determinism.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"group": args.group, "seeds": args.seeds, "models": report},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    if bad:
        print(f"NOT deterministic: {', '.join(bad)} — these need seed repeats too", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
