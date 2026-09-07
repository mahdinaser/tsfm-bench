"""Model wrapper interface.

Every wrapper returns point forecasts and, where the model supports it,
quantile forecasts at the 9 deciles. Wrappers must not look at the test
hold-out for anything except final scoring.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..metrics import QUANTILE_LEVELS


@dataclass
class Forecast:
    point: np.ndarray                      # (n_series, h)
    quantiles: np.ndarray | None = None    # (n_series, h, 9) or None
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0


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
