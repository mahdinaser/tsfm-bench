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
    # Uniform context cap, applied identically to every model. The ERA5 hourly
    # series run to 17k points — longer than anything in M4 — which no
    # foundation model can read (their windows stop at 512-2048) and which
    # makes a seasonal AutoARIMA search intractable. Truncating in the loader
    # keeps the comparison fair rather than letting each model choose.
    max_context: int | None = None


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
    # --- Wikimedia pageviews (post-cutoff holdout; see docs/dataset-plan.md).
    #     Fetched by data/sources/wikimedia_fetch.py; horizons/seasonality
    #     mirror M4's daily/weekly/monthly so results sit next to prior work.
    #     Monthly horizon is 8, not M4's 18: with the test origin at 1 Jan 2026
    #     (docs/dataset-plan.md §1) only eight complete months exist yet.
    "wiki_daily":    Group("wiki_daily",    "Wikimedia", "D",  14, 7,  SUBSAMPLE_N),
    #     Weekly seasonality is 1, as in M4's official weekly setting (m4_weekly
    #     above): a seasonal ARIMA search at m=52 is what the M4 organisers
    #     avoided, and it exhausts memory here too.
    "wiki_weekly":   Group("wiki_weekly",   "Wikimedia", "W",  13, 1,  SUBSAMPLE_N),
    "wiki_monthly":  Group("wiki_monthly",  "Wikimedia", "MS", 8,  12, SUBSAMPLE_N),
    # --- Open-Meteo ERA5 hourly weather (post-cutoff holdout, second domain).
    #     Horizon 48 and seasonality 24 are M4's official hourly settings.
    "weather_hourly": Group("weather_hourly", "OpenMeteo", "H", 48, 24, SUBSAMPLE_N,
                            max_context=2016),   # 12 weeks: covers daily (24) and weekly (168) cycles
    # --- Open-Meteo CAMS hourly air quality (third domain). Same sampling rate
    #     and same daily cycle as weather, but spiky and heavy-tailed: pollution
    #     episodes are level shifts no seasonal structure anticipates. Having
    #     both separates "good at hourly" from "good at smooth".
    "airquality_hourly": Group("airquality_hourly", "OpenMeteoAQ", "H", 48, 24, SUBSAMPLE_N,
                               max_context=2016),
    # --- Danish grid settlement, hourly (fourth domain). Electricity is the
    #     domain foundation-model papers claim most often, but always on
    #     ETTh/ETTm, which predate every model and sit in their pretraining.
    "energy_hourly":  Group("energy_hourly",  "Energinet", "H", 48, 24, None,
                            max_context=2016),
    # --- ECB daily reference rates (fifth domain, and the adversarial one).
    #     Seasonality 1: exchange rates have no weekly cycle, and the standing
    #     result is that nothing reliably beats a naive forecast on them. A
    #     benchmark with no such domain cannot show where the advantage stops.
    "fx_daily":       Group("fx_daily",       "ECB", "D", 14, 1, None),
}

WIKI_CSV = os.path.join(DATA_ROOT, "wikimedia", "pageviews_daily.csv.gz")
WEATHER_CSV = os.path.join(DATA_ROOT, "openmeteo", "weather_hourly.csv.gz")
AQ_CSV = os.path.join(DATA_ROOT, "openmeteo_aq", "airquality_hourly.csv.gz")
ENERGY_CSV = os.path.join(DATA_ROOT, "energidata", "energy_hourly.csv.gz")
FX_CSV = os.path.join(DATA_ROOT, "ecb", "fx_daily.csv.gz")

# Forecast origin for the post-cutoff groups: the first day of the hold-out.
# Set once the pretraining-cutoff table in docs/dataset-plan.md is filled
# (t* = latest cutoff + 1 month). Unset = hold out the last `horizon` points,
# which is fine for smoke runs but is NOT the contamination-free protocol.
TEST_ORIGIN = os.environ.get("TSFM_BENCH_ORIGIN")


def _split_at_origin(df: pd.DataFrame, horizon: int, label: str
                     ) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    """Hold out `horizon` points from TEST_ORIGIN (or the tail if unset)."""
    if TEST_ORIGIN:
        origin = pd.Timestamp(TEST_ORIGIN)
        pos = df.index.searchsorted(origin)
        if pos + horizon > len(df):
            raise ValueError(f"{label}: origin {origin.date()} leaves fewer than {horizon} points to hold out")
    else:
        pos = len(df) - horizon
    ids, train, test = [], [], []
    for col in df.columns:
        s = df[col].to_numpy(dtype=float)
        tr, te = s[:pos], s[pos:pos + horizon]
        if np.isnan(tr).any() or np.isnan(te).any() or len(tr) < 2 * horizon:
            continue
        ids.append(col); train.append(tr); test.append(te)
    return tuple(ids), tuple(train), np.asarray(test, dtype=float)


@lru_cache(maxsize=None)
def _load_weather(horizon: int) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    if not os.path.exists(WEATHER_CSV):
        raise FileNotFoundError(f"{WEATHER_CSV} missing — run data/sources/openmeteo_fetch.py")
    df = pd.read_csv(WEATHER_CSV, index_col="time", parse_dates=True)
    return _split_at_origin(df, horizon, "weather_hourly")


@lru_cache(maxsize=None)
def _load_wide(path: str, index_col: str, horizon: int, label: str
               ) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    """Load a wide CSV of one column per series and split it at the origin."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run the matching fetcher in data/sources/")
    df = pd.read_csv(path, index_col=index_col, parse_dates=True)
    return _split_at_origin(df, horizon, label)


@lru_cache(maxsize=None)
def _load_wikimedia(freq: str, horizon: int) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    if not os.path.exists(WIKI_CSV):
        raise FileNotFoundError(f"{WIKI_CSV} missing — run data/sources/wikimedia_fetch.py")
    df = pd.read_csv(WIKI_CSV, index_col="date", parse_dates=True)
    if freq != "D":
        # Sum views within each week (Mon-anchored) / month; drop a partial last period.
        df = df.resample("W-MON" if freq == "W" else "MS").sum()
        last_full = df.index[-1] + pd.tseries.frequencies.to_offset("W-MON" if freq == "W" else "MS")
        if last_full > pd.Timestamp.today().normalize():
            df = df.iloc[:-1]
    return _split_at_origin(df, horizon, f"wikimedia/{freq}")


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


# Sources that arrive as one wide CSV, one column per series.
WIDE_SOURCES: dict[str, tuple[str, str]] = {
    "OpenMeteoAQ": (AQ_CSV, "time"),
    "Energinet":   (ENERGY_CSV, "time"),
    "ECB":         (FX_CSV, "date"),
}


def load_group(name: str) -> GroupData:
    g = GROUPS[name]
    if g.source == "M4":
        ids, train, test = _load_m4(g.frequency)
        ids, train = list(ids), list(train)
    elif g.source == "M3":
        ids, train, test = _load_rda_group(M3_RDA, "M3", g.frequency)
    elif g.source == "Tourism":
        ids, train, test = _load_rda_group(TOURISM_RDA, "tourism", g.frequency)
    elif g.source == "Wikimedia":
        ids, train, test = _load_wikimedia(g.frequency, g.horizon)
        ids, train = list(ids), list(train)
    elif g.source == "OpenMeteo":
        ids, train, test = _load_weather(g.horizon)
        ids, train = list(ids), list(train)
    elif g.source in WIDE_SOURCES:
        path, index_col = WIDE_SOURCES[g.source]
        ids, train, test = _load_wide(path, index_col, g.horizon, g.name)
        ids, train = list(ids), list(train)
    else:
        raise ValueError(g.source)

    widths = {len(np.asarray(t).ravel()) for t in test}
    if widths != {g.horizon}:
        raise ValueError(
            f"{name}: hold-out widths {sorted(widths)} != official horizon {g.horizon}"
        )
    test = np.asarray([np.asarray(t, dtype=float).ravel() for t in test], dtype=float)

    if g.max_context is not None:
        train = [t[-g.max_context:] for t in train]

    if g.subsample is not None and len(ids) > g.subsample:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        pick = np.sort(rng.choice(len(ids), size=g.subsample, replace=False))
        ids = [ids[i] for i in pick]
        train = [train[i] for i in pick]
        test = test[pick]

    return GroupData(group=g, ids=ids, train=train, test=test)


def _as_scalar_str(value) -> str:
    """R length-1 character vectors reach Python in several shapes.

    Depending on the rdata version and whether xarray is installed, a field
    like ``period`` can arrive as ``str``, ``bytes``, a 0-d/1-element numpy
    array, or an xarray ``DataArray``.  ``str()`` on the array shapes yields
    ``"['YEARLY']"``, which silently matches nothing -- so normalise first.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    arr = np.asarray(getattr(value, "values", value)).ravel()
    if arr.size == 0:
        return ""
    item = arr[0]
    if isinstance(item, (bytes, bytearray)):
        return item.decode("utf-8", "replace")
    return str(item)


def _as_float_1d(value) -> np.ndarray:
    """Coerce an R ts / numeric vector (possibly an xarray DataArray) to 1-D float."""
    return np.asarray(getattr(value, "values", value), dtype=float).ravel()


def _entry_field(entry, key: str):
    """Read a field from an Mdata record however rdata chose to represent it."""
    try:
        return entry[key]
    except (TypeError, KeyError, IndexError):
        pass
    if hasattr(entry, key):
        return getattr(entry, key)
    raise KeyError(f"record has no field {key!r} (available: {_entry_keys(entry)})")


def _entry_keys(entry) -> list[str]:
    if hasattr(entry, "keys"):
        return [str(k) for k in entry.keys()]
    return [a for a in dir(entry) if not a.startswith("_")]


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

    collection = _read_rda_collection(path, obj)

    want = period.upper()
    ids, train, test, seen = [], [], [], set()
    for sid, entry in collection.items():
        label = _as_scalar_str(_entry_field(entry, "period")).strip().upper()
        seen.add(label)
        if label != want:
            continue
        ids.append(str(sid))
        train.append(_as_float_1d(_entry_field(entry, "x")))
        test.append(_as_float_1d(_entry_field(entry, "xx")))

    if not ids:
        raise ValueError(
            f"{os.path.basename(path)}: no series with period {want!r}; "
            f"labels present: {sorted(seen)}"
        )
    return ids, train, test


@lru_cache(maxsize=None)
def _read_rda_collection(path: str, obj: str):
    """Parse an .rda once and return the named collection inside it."""
    import rdata

    converted = rdata.conversion.convert(rdata.parser.parse_file(path))
    if obj not in converted:
        raise KeyError(
            f"{os.path.basename(path)} contains {sorted(converted)}, not {obj!r}"
        )
    return converted[obj]


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
