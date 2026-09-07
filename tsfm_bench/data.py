"""Dataset definitions and loaders for the 13 benchmark groups.

Groups follow the official competition structure:
  M4      - Yearly, Quarterly, Monthly, Weekly, Daily, Hourly
  M3      - Yearly, Quarterly, Monthly, Other
  Tourism - Yearly, Quarterly, Monthly

Horizons and seasonal periods are the official competition values; every
count reported by describe_groups() is measured from the data on disk, never
hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get(
    "TSFM_BENCH_DATA",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)

M4_DIR = os.path.join(DATA_ROOT, "M4-methods", "Dataset")
M3_RDA = os.path.join(DATA_ROOT, "Mcomp", "data", "M3.rda")
TOURISM_RDA = os.path.join(DATA_ROOT, "Tcomp", "pkg", "data", "tourism.rda")

SUBSAMPLE_SEED = 20260907
SUBSAMPLE_N = 500


@dataclass
class Group:
    name: str
    source: str          # "M4" | "M3" | "Tourism"
    frequency: str       # competition frequency label
    horizon: int         # official forecast horizon
    seasonality: int     # seasonal period m used by MASE and seasonal naive
    subsample: int | None = None   # None = use all series


GROUPS: dict[str, Group] = {
    # --- M4: 500-series random subsample per frequency (seed 20260907);
    #     weekly and hourly used in full because they are already small.
    "m4_yearly":     Group("m4_yearly",    "M4", "Yearly",    6,  1,  SUBSAMPLE_N),
    "m4_quarterly":  Group("m4_quarterly", "M4", "Quarterly", 8,  4,  SUBSAMPLE_N),
    "m4_monthly":    Group("m4_monthly",   "M4", "Monthly",   18, 12, SUBSAMPLE_N),
    "m4_weekly":     Group("m4_weekly",    "M4", "Weekly",    13, 1,  None),
    "m4_daily":      Group("m4_daily",     "M4", "Daily",     14, 1,  SUBSAMPLE_N),
    "m4_hourly":     Group("m4_hourly",    "M4", "Hourly",    48, 24, None),
    # --- M3: all 3,003 series
    "m3_yearly":     Group("m3_yearly",    "M3", "YEARLY",    6,  1,  None),
    "m3_quarterly":  Group("m3_quarterly", "M3", "QUARTERLY", 8,  4,  None),
    "m3_monthly":    Group("m3_monthly",   "M3", "MONTHLY",   18, 12, None),
    "m3_other":      Group("m3_other",     "M3", "OTHER",     8,  1,  None),
    # --- Tourism: all 1,311 series
    "tourism_yearly":    Group("tourism_yearly",    "Tourism", "YEARLY",    4,  1,  None),
    "tourism_quarterly": Group("tourism_quarterly", "Tourism", "QUARTERLY", 8,  4,  None),
    "tourism_monthly":   Group("tourism_monthly",   "Tourism", "MONTHLY",   24, 12, None),
}


@dataclass
class GroupData:
    group: Group
    ids: list[str]
    train: list[np.ndarray]      # variable-length in-sample series
    test: np.ndarray             # (n_series, horizon) hold-out

    @property
    def n_series(self) -> int:
        return len(self.ids)


def _read_m4_csv(path: str) -> tuple[list[str], list[np.ndarray]]:
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str).tolist()
    values = df.iloc[:, 1:].to_numpy(dtype=float)
    series = [row[~np.isnan(row)] for row in values]
    return ids, series


@lru_cache(maxsize=None)
def _load_m4(frequency: str) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    tr_ids, train = _read_m4_csv(os.path.join(M4_DIR, "Train", f"{frequency}-train.csv"))
    te_ids, test = _read_m4_csv(os.path.join(M4_DIR, "Test", f"{frequency}-test.csv"))
    if tr_ids != te_ids:
        order = {sid: i for i, sid in enumerate(te_ids)}
        test = [test[order[sid]] for sid in tr_ids]
    return tuple(tr_ids), tuple(train), np.asarray(test, dtype=float)


def load_group(name: str) -> GroupData:
    g = GROUPS[name]
    if g.source == "M4":
        ids, train, test = _load_m4(g.frequency)
        ids, train = list(ids), list(train)
    elif g.source == "M3":
        ids, train, test = _load_rda_group(M3_RDA, "M3", g.frequency)
    elif g.source == "Tourism":
        ids, train, test = _load_rda_group(TOURISM_RDA, "tourism", g.frequency)
    else:
        raise ValueError(g.source)

    test = np.asarray(test, dtype=float)
    if test.shape[1] != g.horizon:
        raise ValueError(
            f"{name}: hold-out width {test.shape[1]} != official horizon {g.horizon}"
        )

    if g.subsample is not None and len(ids) > g.subsample:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        pick = np.sort(rng.choice(len(ids), size=g.subsample, replace=False))
        ids = [ids[i] for i in pick]
        train = [train[i] for i in pick]
        test = test[pick]

    return GroupData(group=g, ids=ids, train=train, test=test)


def _load_rda_group(path: str, obj: str, period: str):
    """Load an M3 / Tourism group from the packaged .rda.

    Requires the `rdata` package; raises a clear error if unavailable.
    """
    try:
        import rdata
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "reading .rda requires the `rdata` package "
            "(pip install rdata xarray)"
        ) from exc

    parsed = rdata.parser.parse_file(path)
    converted = rdata.conversion.convert(parsed)
    collection = converted[obj]

    ids, train, test = [], [], []
    for sid, entry in collection.items():
        if str(entry["period"]).upper() != period.upper():
            continue
        ids.append(str(sid))
        train.append(np.asarray(entry["x"], dtype=float))
        test.append(np.asarray(entry["xx"], dtype=float))
    return ids, train, test


def describe_groups() -> pd.DataFrame:
    """Measured summary of every group: counts, horizon, series lengths."""
    rows = []
    for name, g in GROUPS.items():
        try:
            d = load_group(name)
            lengths = np.array([len(s) for s in d.train])
            rows.append({
                "group": name, "source": g.source, "frequency": g.frequency,
                "n_series": d.n_series, "horizon": g.horizon,
                "seasonality": g.seasonality,
                "min_len": int(lengths.min()), "median_len": int(np.median(lengths)),
                "max_len": int(lengths.max()),
            })
        except Exception as exc:
            rows.append({
                "group": name, "source": g.source, "frequency": g.frequency,
                "n_series": f"ERROR: {type(exc).__name__}", "horizon": g.horizon,
                "seasonality": g.seasonality,
                "min_len": None, "median_len": None, "max_len": None,
            })
    return pd.DataFrame(rows)
