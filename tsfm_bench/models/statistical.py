"""Classical baselines via statsforecast.

Theta      - classic Theta (Assimakopoulos & Nikolopoulos), the M4 benchmark
             method: multiplicative classical decomposition when seasonal.
AutoETS    - automatic exponential smoothing (Hyndman & Khandakar).
AutoARIMA  - automatic ARIMA (Hyndman & Khandakar).

All three are fitted per series, in parallel across cores.  Prediction
intervals at levels 20/40/60/80 are converted to the nine deciles, with the
point forecast standing in for the median (exact under the Gaussian error
assumption these models make).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .base import Forecast, Model, interval_levels_to_deciles, to_long_df, wide_from_long

LEVELS = [20, 40, 60, 80]


class _StatsForecastModel(Model):
    probabilistic = True
    sf_name: str = ""          # column name statsforecast writes

    def make(self, seasonality: int):
        raise NotImplementedError

    def forecast(self, train, horizon, seasonality):
        import statsforecast
        from statsforecast import StatsForecast
        self.version = statsforecast.__version__

        df = to_long_df(train)
        n_jobs = int(os.environ.get("TSFM_BENCH_JOBS", os.cpu_count() or 1))
        sf = StatsForecast(models=[self.make(seasonality)], freq=1,
                           n_jobs=n_jobs, fallback_model=None)
        t0 = time.perf_counter()
        out = sf.forecast(df=df, h=horizon, level=LEVELS)
        elapsed = time.perf_counter() - t0
        out = out.reset_index() if "unique_id" not in out.columns else out

        n = len(train)
        cols = [self.sf_name] + [f"{self.sf_name}-lo-{l}" for l in LEVELS] \
                              + [f"{self.sf_name}-hi-{l}" for l in LEVELS]
        wide = wide_from_long(out, cols, n, horizon)
        point = wide[..., 0]
        lo = {l: wide[..., 1 + i] for i, l in enumerate(LEVELS)}
        hi = {l: wide[..., 1 + len(LEVELS) + i] for i, l in enumerate(LEVELS)}
        q = interval_levels_to_deciles(point, lo, hi)
        # statsforecast fits and predicts in one call; the whole cost is "fit"
        return Forecast(point=point, quantiles=q, fit_seconds=elapsed,
                        device="cpu", extra={"n_jobs": n_jobs})


class Theta(_StatsForecastModel):
    name = "Theta"
    sf_name = "Theta"

    def make(self, m):
        from statsforecast.models import Theta as _Theta
        return _Theta(season_length=m)


class AutoETS(_StatsForecastModel):
    name = "AutoETS"
    sf_name = "AutoETS"

    def make(self, m):
        from statsforecast.models import AutoETS as _AutoETS
        return _AutoETS(season_length=m)


class AutoARIMA(_StatsForecastModel):
    name = "AutoARIMA"
    sf_name = "AutoARIMA"

    def make(self, m):
        from statsforecast.models import AutoARIMA as _AutoARIMA
        return _AutoARIMA(season_length=m)
