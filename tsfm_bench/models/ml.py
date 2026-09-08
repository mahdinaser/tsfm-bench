"""Gradient-boosted trees baseline: one global LightGBM per group via mlforecast.

Features: lags 1..L and the rolling mean over one season of lag 1, where
L = max(h, 2m) capped so the shortest series in the group still yields at
least one training row.  Forecasts are produced recursively.  Point only.
Hyper-parameters are fixed (no per-group tuning) and listed in `LGBM_PARAMS`.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .base import Forecast, Model, RANDOM_SEED, to_long_df, wide_from_long

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    verbosity=-1,
)


def lag_window(train, horizon, seasonality) -> int:
    min_len = min(len(t) for t in train)
    L = max(horizon, 2 * seasonality)
    return int(max(1, min(L, min_len - 1)))


class LightGBMModel(Model):
    name = "LightGBM"
    probabilistic = False

    def forecast(self, train, horizon, seasonality):
        import lightgbm
        import mlforecast
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import RollingMean
        from mlforecast.target_transforms import LocalStandardScaler
        self.version = f"mlforecast {mlforecast.__version__} / lightgbm {lightgbm.__version__}"

        L = lag_window(train, horizon, seasonality)
        threads = int(os.environ.get("TSFM_BENCH_JOBS", os.cpu_count() or 1))
        lag_tf = {1: [RollingMean(window_size=seasonality, min_samples=1)]} if seasonality > 1 else None

        fcst = MLForecast(
            models={"LightGBM": lightgbm.LGBMRegressor(n_jobs=threads, **LGBM_PARAMS)},
            freq=1,
            lags=list(range(1, L + 1)),
            lag_transforms=lag_tf,
            target_transforms=[LocalStandardScaler()],
            num_threads=threads,
        )
        df = to_long_df(train)
        t0 = time.perf_counter()
        fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = fcst.predict(h=horizon)
        pred_s = time.perf_counter() - t0

        point = wide_from_long(pred, ["LightGBM"], len(train), horizon)[..., 0]
        return Forecast(point=point, fit_seconds=fit_s, predict_seconds=pred_s,
                        device="cpu", extra={"lags": L, "threads": threads, **LGBM_PARAMS})
