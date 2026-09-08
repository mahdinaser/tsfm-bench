#!/usr/bin/env python3
"""Measure the series properties that were supposed to explain the results.

This script exists to test a hypothesis, and it refuted it. The hypothesis was
that pretraining pays where a series is periodic AND noisy: a repeating
structure worth transferring, too obscured by noise to estimate locally. Two
numbers per group settle it, and they say no. Seasonal strength runs the wrong
way (Wikipedia has almost none at the tested period and the largest gains;
weather has the most and a middling one), and spectral entropy fails the other
way (exchange rates are as high-entropy as Wikipedia and show no advantage).

The output is kept, and printed in the paper, because the negative result is
what motivates the corpus-overlap explanation that replaced it. A story fitted
to seven observations after the fact is worth less than a measurement that
rules one out.

Both measures are the standard ones from the forecasting feature literature
(Hyndman's tsfeatures, as used to characterise the M4 panel):

  seasonal strength   1 - Var(remainder) / Var(remainder + seasonal) from an
                      STL decomposition at the group's seasonal period. 0 means
                      no repeating structure, 1 means the season explains
                      everything.

  spectral entropy    Shannon entropy of the normalised periodogram, divided by
                      log(n) so it lands in [0, 1]. Low means the series is
                      concentrated at a few frequencies and therefore easy to
                      extrapolate; high means it is close to white noise. This
                      is the standard operationalisation of "forecastability".

Computed on the TRAINING window only. Using the test window would let the
characterisation peek at the thing it is supposed to predict.

  python scripts/features.py
  python scripts/features.py --groups wiki_daily fx_daily --max-series 100
"""
from __future__ import annotations

import argparse, json, os, sys, warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsfm_bench.data import GROUPS, load_group  # noqa: E402

warnings.filterwarnings("ignore")


def spectral_entropy(x: np.ndarray) -> float:
    from scipy import signal
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.allclose(x, x[0]):
        return np.nan
    # Differencing first: a trend puts almost all power at frequency zero and
    # would report every trending series as perfectly predictable.
    x = np.diff(x)
    if np.allclose(x, 0):
        return np.nan
    _, pxx = signal.periodogram(x - x.mean(), scaling="density")
    pxx = pxx[1:]                      # drop the zero-frequency bin
    total = pxx.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan
    p = pxx / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def seasonal_strength(x: np.ndarray, period: int) -> float:
    if period < 2:
        return 0.0
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 * period or np.allclose(x, x[0]):
        return np.nan
    from statsmodels.tsa.seasonal import STL
    try:
        res = STL(x, period=period, robust=True).fit()
    except Exception:
        return np.nan
    rem, seas = res.resid, res.seasonal
    denom = np.var(rem + seas)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(max(0.0, 1.0 - np.var(rem) / denom))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--max-series", type=int, default=150,
                    help="cap per group; STL on thousands of hourly series is slow "
                         "and the group medians are stable well before that")
    ap.add_argument("--out", default="paper/features.json")
    args = ap.parse_args()
    os.environ.setdefault("TSFM_BENCH_ORIGIN", "2026-01-01")

    names = args.groups or ["wiki_daily", "wiki_weekly", "wiki_monthly",
                            "weather_hourly", "airquality_hourly",
                            "energy_hourly", "fx_daily"]
    rows = []
    for name in names:
        g = GROUPS[name]
        data = load_group(name)
        rng = np.random.default_rng(20260907)
        idx = rng.permutation(len(data.train))[:args.max_series]
        ent, seas = [], []
        for i in idx:
            s = np.asarray(data.train[i], dtype=float)
            ent.append(spectral_entropy(s))
            seas.append(seasonal_strength(s, g.seasonality))
        row = {
            "group": name, "n_used": int(len(idx)), "seasonality": g.seasonality,
            "spectral_entropy": float(np.nanmedian(ent)),
            "seasonal_strength": float(np.nanmedian(seas)),
        }
        rows.append(row)
        print(f"  {name:20} entropy {row['spectral_entropy']:.3f}   "
              f"seasonal strength {row['seasonal_strength']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    # Join to the outcome, if the results are already computed.
    facts_path = "paper/figures.json"
    if os.path.exists(facts_path):
        facts = json.load(open(facts_path))["groups"]
        df["foundation_gain_pct"] = [facts.get(g, {}).get("foundation_gain_pct") for g in df["group"]]
        df["friedman_p"] = [facts.get(g, {}).get("friedman_p") for g in df["group"]]
        df["label"] = [facts.get(g, {}).get("label", g) for g in df["group"]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_json(args.out, orient="records", indent=2)

    # The table the paper prints, so the negative result is shown rather than
    # asserted: both measured properties beside the advantage they fail to
    # order.
    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Group & Seasonal strength & Spectral entropy & Pretrained gain \\",
           r"\midrule"]
    for _, r in df.iterrows():
        gain = r.get("foundation_gain_pct")
        gain_s = "--" if gain is None or pd.isna(gain) else f"{gain:+.1f}\\%"
        if r.get("friedman_p") is not None and not pd.isna(r.get("friedman_p")) \
                and r["friedman_p"] >= 0.05:
            gain_s = r"n.s."
        tex.append(f"{r.get('label', r['group'])} & {r['seasonal_strength']:.2f} & "
                   f"{r['spectral_entropy']:.2f} & {gain_s} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    out_tex = os.path.join(os.path.dirname(args.out), "tables", "features.tex")
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    open(out_tex, "w").write("\n".join(tex) + "\n")
    print(f"\nwrote {args.out} and {out_tex}")

    if "foundation_gain_pct" in df.columns and df["foundation_gain_pct"].notna().sum() >= 3:
        from scipy import stats
        sub = df.dropna(subset=["foundation_gain_pct"])
        # The claim is a conjunction, so the product is the natural single
        # predictor: either factor near zero should predict no advantage.
        sub = sub.assign(product=sub["seasonal_strength"] * sub["spectral_entropy"])
        for col in ("seasonal_strength", "spectral_entropy", "product"):
            r = stats.pearsonr(sub[col], sub["foundation_gain_pct"])
            rho = stats.spearmanr(sub[col], sub["foundation_gain_pct"])
            print(f"  {col:18} pearson r={r.statistic:+.3f} (p={r.pvalue:.3f})   "
                  f"spearman rho={rho.statistic:+.3f} (p={rho.pvalue:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
