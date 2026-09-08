"""Model wrapper interface.

Every wrapper returns point forecasts and, where the model supports it,
quantile forecasts at the 9 deciles. Wrappers must not look at the test
hold-out for anything except final scoring.

Conventions shared by every wrapper (these are stated in the paper):
  * point forecast of a probabilistic model = its 0.5 quantile (median)
  * quantile levels = 0.1, 0.2, ..., 0.9 in that order on the last axis
  * the whole in-sample series is offered as context; each model applies
    its own maximum-context truncation
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metrics import QUANTILE_LEVELS

# One number decides LightGBM's bagging, the neural models' initialisation and
# their batch order. Reporting a single run of those three and calling the gap
# to a competitor a result assumes the seed did not matter, which is exactly
# what a reader should not have to take on trust — so the seed is settable and
# the paper reports the spread across several. Deterministic methods (Theta,
# AutoETS, AutoARIMA, and the pretrained models at fixed weights) ignore it.
RANDOM_SEED = int(os.environ.get("TSFM_BENCH_SEED", "20260907"))


@dataclass
class Forecast:
    point: np.ndarray                      # (n_series, h)
    quantiles: np.ndarray | None = None    # (n_series, h, 9) or None
    fit_seconds: float = 0.0               # training / model-selection time
    predict_seconds: float = 0.0           # inference time
    load_seconds: float = 0.0              # one-time weight loading (foundation models)
    device: str = "cpu"
    n_params: int | None = None            # trainable + frozen parameter count
    extra: dict = field(default_factory=dict)   # any other measured facts


class Model:
    """Base class. Subclasses implement forecast()."""

    name: str = "Model"
    probabilistic: bool = False
    #: set by subclasses once the underlying library is imported
    version: str | None = None

    def __init__(self, **kwargs):
        self.params = kwargs

    def forecast(self, train: list[np.ndarray], horizon: int,
                 seasonality: int) -> Forecast:
        raise NotImplementedError

    def run(self, train, horizon, seasonality) -> Forecast:
        t0 = time.perf_counter()
        fc = self.forecast(train, horizon, seasonality)
        elapsed = time.perf_counter() - t0
        if fc.predict_seconds == 0.0 and fc.fit_seconds == 0.0:
            fc.predict_seconds = elapsed
        return fc


class SeasonalNaive(Model):
    """Repeat the last seasonal cycle. Matches the official M4 definition."""

    name = "SeasonalNaive"
    probabilistic = False
    version = "harness"

    def forecast(self, train, horizon, seasonality):
        out = np.empty((len(train), horizon), dtype=float)
        for i, tr in enumerate(train):
            if seasonality > 1 and len(tr) >= seasonality:
                out[i] = np.resize(tr[-seasonality:], horizon)
            else:
                out[i] = tr[-1]
        return Forecast(point=out)


# --------------------------------------------------------------------- helpers

def pick_device() -> str:
    """cpu | mps | cuda. Override with TSFM_BENCH_DEVICE."""
    forced = os.environ.get("TSFM_BENCH_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def to_long_df(train: list[np.ndarray]) -> pd.DataFrame:
    """Nixtla long format with integer timestamps: unique_id, ds, y."""
    parts = []
    for i, tr in enumerate(train):
        n = len(tr)
        parts.append(pd.DataFrame({
            "unique_id": np.full(n, i, dtype=np.int64),
            "ds": np.arange(1, n + 1, dtype=np.int64),
            "y": np.asarray(tr, dtype=float),
        }))
    return pd.concat(parts, ignore_index=True)


def wide_from_long(df: pd.DataFrame, cols: list[str], n_series: int,
                   horizon: int) -> np.ndarray:
    """(n_series, h, len(cols)) from a Nixtla forecast frame ordered by ds."""
    df = df.sort_values(["unique_id", "ds"])
    out = np.empty((n_series, horizon, len(cols)), dtype=float)
    for i, (_, g) in enumerate(df.groupby("unique_id", sort=True)):
        out[i] = g[cols].to_numpy(dtype=float)[:horizon]
    return out


def interval_levels_to_deciles(point: np.ndarray, lo: dict[int, np.ndarray],
                               hi: dict[int, np.ndarray]) -> np.ndarray:
    """Assemble deciles from symmetric interval bounds.

    lo/hi keyed by level in {20, 40, 60, 80}:  lo[80] is q0.1, ..., lo[20] is
    q0.4, the point forecast stands in for q0.5, hi[20] is q0.6, ..., hi[80]
    is q0.9.  Used for models whose libraries expose intervals, not quantiles.
    """
    stack = [lo[80], lo[60], lo[40], lo[20], point, hi[20], hi[40], hi[60], hi[80]]
    q = np.stack(stack, axis=-1)
    return np.sort(q, axis=-1)          # guard against crossing


def count_params(module) -> int | None:
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except Exception:
        return None
