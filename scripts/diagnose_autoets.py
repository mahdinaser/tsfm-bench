#!/usr/bin/env python3
"""Why AutoETS's WQL explodes to ~1e14 on wiki_daily while its MASE is fine.

Run:  TSFM_BENCH_ORIGIN=2026-01-01 python scripts/diagnose_autoets.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsfm_bench.data import load_group
from tsfm_bench.models import get_model


def main():
    d = load_group("wiki_daily")
    n = 60
    train, test, ids = d.train[:n], d.test[:n], d.ids[:n]
    fc = get_model("AutoETS").forecast(train, d.group.horizon, d.group.seasonality)
    q = np.asarray(fc.quantiles, dtype=float)
    point = np.asarray(fc.point, dtype=float)

    print(f"quantiles {q.shape} finite={np.isfinite(q).all()}  point finite={np.isfinite(point).all()}")
    mag = np.abs(q).max(axis=(1, 2))
    for i in np.argsort(mag)[-5:][::-1]:
        tr = np.asarray(train[i], dtype=float)
        print(f"  {ids[i][:26]:26} max|q|={mag[i]:.3e}  point_max={np.abs(point[i]).max():.3e}"
              f"  train[min,max]=[{tr.min():.0f},{tr.max():.0f}]  test_max={test[i].max():.0f}")
    print(f"\nmedian max|q| = {np.median(mag):.3e}")

    # Which decile is blowing up: the far tails or the whole fan?
    spread = q[..., -1] - q[..., 0]
    worst = int(np.argmax(spread.max(axis=1)))
    print(f"\nworst series {ids[worst]}: q0.1..q0.9 at h=1")
    print("  ", np.array2string(q[worst, 0], precision=2, max_line_width=200))
    print("   point:", point[worst, 0])


if __name__ == "__main__":
    main()
