#!/usr/bin/env python3
"""The two figures that carry arguments a table cannot.

1. MCB rank intervals per group. The paper's central claim is about which gaps
   are real, and a reader checks that by looking for overlap. In a table of
   numbers that is arithmetic; drawn, it is immediate -- and the exchange-rate
   panel, where every interval overlaps every other, makes the null result
   visible in a way no p-value does.

2. Cost against accuracy. The practical finding is that the pretrained models
   are clustered tightly on both axes while the automatic classical search sits
   two orders of magnitude to the right for no accuracy gain. That is a shape,
   not a number.

  python scripts/make_figures.py
"""
from __future__ import annotations

import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.significance import load_panel, mcb  # noqa: E402
from scripts.make_tables import CLASSICAL, FOUNDATION, GROUP_LABEL, LEARNED, ORDER  # noqa: E402

FAMILY_COLOUR = {"classical": "#b45309", "learned": "#0f766e", "pretrained": "#4338ca"}


def family(model: str) -> str:
    if model in FOUNDATION:
        return "pretrained"
    if model in CLASSICAL:
        return "classical"
    return "learned"


def fig_mcb(panels: dict[str, pd.DataFrame], out: str) -> None:
    groups = [g for g in ORDER if g in panels]
    # Three columns, not two: at two the figure is taller than a page, LaTeX
    # floats it past the references, and the reader meets it after the
    # conclusion. Wider and shorter keeps it on the page it is discussed on.
    ncol = 3
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.35 * nrow), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, g in zip(axes, groups):
        res = mcb(panels[g]).sort_values("avg_rank", ascending=False)
        y = np.arange(len(res))
        for i, (name, row) in enumerate(res.iterrows()):
            c = FAMILY_COLOUR[family(name)]
            ax.plot([row["lo"], row["hi"]], [i, i], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(row["avg_rank"], i, "o", color=c, ms=5)
        best_hi = res["hi"].iloc[-1]
        # Everything left of this line is indistinguishable from the winner.
        ax.axvline(best_hi, color="#64748b", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(res.index, fontsize=7)
        p = res.attrs["friedman_p"]
        note = "  (Friedman n.s.)" if p >= 0.05 else ""
        ax.set_title(f"{GROUP_LABEL.get(g, g)}{note}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.25)
    for ax in axes[len(groups):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, lw=2.2, label=k) for k, c in FAMILY_COLOUR.items()]
    axes[len(groups) - 1].legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)
    fig.supxlabel("average rank across series (lower is better)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cost(results: str, panels: dict[str, pd.DataFrame], out: str) -> None:
    t = pd.DataFrame([json.loads(l) for l in open(os.path.join(results, "timing.jsonl"))])
    if "seed" in t.columns:
        t = t[t["seed"].isna() | (t["seed"] == 20260907)]
    cost = t.groupby("model")["seconds_per_series"].median()
    # Mean rank across groups: one number per model that is comparable across
    # domains, unlike mean MASE.
    ranks = pd.DataFrame({g: mcb(p)["avg_rank"] for g, p in panels.items()}).mean(axis=1)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    pts = [(m, max(cost[m], 1e-5), ranks[m]) for m in ranks.index if m in cost.index]
    # Labels collide where the pretrained models cluster, and three of them
    # overlapped into unreadable soup in the first render. Alternating the
    # offset for neighbours that are close in both axes is enough to separate
    # them without a layout library.
    pts.sort(key=lambda r: (r[2], r[1]))
    prev = None
    for i, (m, x, y) in enumerate(pts):
        c = FAMILY_COLOUR[family(m)]
        ax.scatter(x, y, color=c, s=46, zorder=3)
        close = prev is not None and abs(y - prev[2]) < 0.18 and abs(np.log10(x) - np.log10(prev[1])) < 0.5
        dx, dy = (6, 4) if not close else (6, -11)
        label = f"{m} (free)" if m == "SeasonalNaive" else m
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, zorder=4)
        prev = (m, x, y)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("median seconds per series (log scale)")
    ax.set_ylabel("mean rank across the seven groups\n(lower is better)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in FAMILY_COLOUR.items()]
    # Not lower left: seasonal naive sits there, and the legend covered it.
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/holdout_2026")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()

    panels = load_panel(args.results, "mase")
    if not panels:
        print("no results", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    fig_mcb(panels, os.path.join(args.out, "mcb.png"))
    fig_cost(args.results, panels, os.path.join(args.out, "cost_accuracy.png"))
    print(f"wrote {args.out}/mcb.png and {args.out}/cost_accuracy.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
