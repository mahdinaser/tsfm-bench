#!/usr/bin/env python3
"""Is the gap between two models real, or is it 500 series of noise?

A table of mean MASE invites the reader to rank models by a third decimal
place. Almost every such gap is inside the sampling error of the panel it was
measured on, and the M4 and M5 organisers settled this question the same way
each time: compare average RANKS across series, and put an interval around
them. Two things are computed here, both on the per-series losses that run.py
already writes:

  1. MCB (Multiple Comparisons with the Best), as used to report the M5
     results. Each series is ranked across models, and each model gets its
     average rank with a Nemenyi interval derived from the Friedman statistic.
     Models whose intervals do not overlap the best model's are the ones that
     are actually distinguishable from it.

  2. A pairwise Wilcoxon signed-rank test of every model against the best model
     on that group, Holm-corrected across the comparisons. Signed-rank rather
     than a paired t-test because MASE across a panel is heavily right-skewed:
     a handful of near-flat series produce enormous ratios that a mean-based
     test reads as signal.

This is deliberately NOT called a Diebold-Mariano test. DM compares two
forecast error series over TIME for one series; what a panel benchmark has is
one loss per series, which is a different comparison and needs the paired
across-series machinery above. Labelling it DM would be borrowing authority
the design does not earn.

  python scripts/significance.py                      # all groups
  python scripts/significance.py --metric smape --results results/holdout_2026
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# Studentised range q_{alpha, k, inf} at alpha = 0.05, indexed by k models.
# Used for the Nemenyi critical distance; k > 20 falls back to the k = 20 value,
# which makes the interval conservative rather than optimistic.
Q05 = {
    2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


def load_panel(results: str, metric: str) -> dict[str, pd.DataFrame]:
    """group -> DataFrame of per-series loss, one column per model.

    Seed repeats are excluded. They belong in the variance estimate, not in the
    headline ranks: three copies of LightGBM in the panel would shift every
    other model's average rank simply by being there.
    """
    frames: dict[str, dict[str, pd.Series]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(results, "metrics", "*.csv"))):
        if "__seed" in os.path.basename(path):
            continue
        df = pd.read_csv(path)
        if metric not in df.columns:
            continue
        model = df["model"].iloc[0]
        group = df["group"].iloc[0]
        frames[group][model] = df.set_index("series_id")[metric]
    out = {}
    for group, cols in frames.items():
        panel = pd.DataFrame(cols)
        # Only series every model produced a forecast for; a model that skipped
        # the hard series would otherwise win by not attempting them.
        panel = panel.dropna(axis=0, how="any")
        panel = panel.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if not panel.empty:
            out[group] = panel
    return out


def mcb(panel: pd.DataFrame) -> pd.DataFrame:
    """Average ranks with Nemenyi half-intervals; lower rank is better."""
    ranks = panel.rank(axis=1, method="average")
    n, k = ranks.shape
    avg = ranks.mean(axis=0)
    q = Q05.get(k, Q05[20])
    # Nemenyi critical distance; halved so two models differ when their
    # intervals do not overlap, which is how MCB plots are read.
    half = q * np.sqrt(k * (k + 1) / (12.0 * n)) / 2.0
    friedman = stats.friedmanchisquare(*[panel[c].to_numpy() for c in panel.columns])
    res = pd.DataFrame({
        "avg_rank": avg,
        "lo": avg - half,
        "hi": avg + half,
        "mean_loss": panel.mean(axis=0),
    }).sort_values("avg_rank")
    best_hi = res["hi"].iloc[0]
    # "In the best group" = interval overlaps the best model's interval.
    res["indistinguishable_from_best"] = res["lo"] <= best_hi
    res.attrs["friedman_p"] = float(friedman.pvalue)
    res.attrs["n_series"] = int(n)
    res.attrs["half_interval"] = float(half)
    return res


def wilcoxon_vs_best(panel: pd.DataFrame, best: str) -> pd.DataFrame:
    rows = []
    for model in panel.columns:
        if model == best:
            continue
        d = panel[model] - panel[best]
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(panel[model], panel[best],
                                     zero_method="zsplit").pvalue)
        rows.append({"model": model, "p_raw": p,
                     "median_delta": float(np.median(d))})
    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    # Holm: the p-values are ordered, so the correction is a running maximum.
    m = len(out)
    out["p_holm"] = np.minimum.accumulate(
        np.minimum(1.0, out["p_raw"].to_numpy() * (m - np.arange(m)))[::-1])[::-1]
    out["p_holm"] = np.maximum.accumulate(out["p_holm"].to_numpy())
    out["worse_than_best"] = out["p_holm"] < 0.05
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/holdout_2026")
    ap.add_argument("--metric", default="mase", choices=["mase", "smape"])
    ap.add_argument("--out", default=None, help="directory for CSV output")
    args = ap.parse_args()

    panels = load_panel(args.results, args.metric)
    if not panels:
        print(f"no per-series {args.metric} found under {args.results}", file=sys.stderr)
        return 1
    out_dir = args.out or os.path.join(args.results, "significance")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    for group, panel in sorted(panels.items()):
        res = mcb(panel)
        best = res.index[0]
        wil = wilcoxon_vs_best(panel, best)
        res.to_csv(os.path.join(out_dir, f"mcb__{group}__{args.metric}.csv"))
        wil.to_csv(os.path.join(out_dir, f"wilcoxon__{group}__{args.metric}.csv"), index=False)

        tied = list(res.index[res["indistinguishable_from_best"]])
        summary[group] = {
            "n_series": res.attrs["n_series"], "n_models": int(panel.shape[1]),
            "friedman_p": res.attrs["friedman_p"], "best": best,
            "indistinguishable_from_best": tied,
        }
        print(f"\n=== {group}  ({res.attrs['n_series']} series, {panel.shape[1]} models, "
              f"Friedman p={res.attrs['friedman_p']:.2e}) ===")
        show = res.head(8).copy()
        for name, row in show.iterrows():
            flag = "=" if row["indistinguishable_from_best"] else " "
            print(f"  {flag} {name:20} rank {row['avg_rank']:5.2f} "
                  f"[{row['lo']:5.2f},{row['hi']:5.2f}]  mean {args.metric.upper()} {row['mean_loss']:8.4f}")
        print(f"  best: {best}; statistically indistinguishable from it: "
              f"{', '.join(t for t in tied if t != best) or 'none'}")

    json.dump(summary, open(os.path.join(out_dir, f"summary__{args.metric}.json"), "w"), indent=2)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
