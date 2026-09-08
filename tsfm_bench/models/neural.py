"""Trained neural baselines via neuralforecast: LSTM and N-BEATS.

Both are trained as one global model per group (all series of the group in
one training set), with:
  * input window  = max(2h, 2m), left-padded for series shorter than that
  * loss          = multi-quantile (MQLoss) at the nine deciles;
                    point forecast = the 0.5 quantile
  * validation    = the last h points of every training series, used for
                    early stopping (the test hold-out is never seen)
  * fixed seed, fixed architecture; no per-group hyper-parameter search
The full configuration is recorded in Forecast.extra.
"""
from __future__ import annotations

import logging
import os
import time
import warnings

import numpy as np

from .base import Forecast, Model, RANDOM_SEED, count_params, pick_device, to_long_df, wide_from_long

LEVELS = [20, 40, 60, 80]
# MQLoss(level=LEVELS) output suffixes in decile order
DECILE_SUFFIX = ["-lo-80", "-lo-60", "-lo-40", "-lo-20", "-median",
                 "-hi-20", "-hi-40", "-hi-60", "-hi-80"]

COMMON = dict(
    max_steps=1000,
    val_check_steps=50,
    early_stop_patience_steps=5,
    learning_rate=1e-3,
    batch_size=32,
    windows_batch_size=256,
    scaler_type="robust",
    start_padding_enabled=True,
    random_seed=RANDOM_SEED,
)


def _quiet():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


def _accelerator(device: str) -> dict:
    if device == "mps":
        return dict(accelerator="mps", devices=1)
    if device == "cuda":
        return dict(accelerator="gpu", devices=1)
    return dict(accelerator="cpu", devices=1)


class _NeuralForecastModel(Model):
    probabilistic = True
    nf_name: str = ""

    def build(self, horizon, input_size, loss, trainer_kwargs):
        raise NotImplementedError

    def forecast(self, train, horizon, seasonality):
        _quiet()
        import neuralforecast
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        self.version = neuralforecast.__version__

        device = pick_device()
        input_size = int(max(2 * horizon, 2 * seasonality))
        loss = MQLoss(level=LEVELS)
        trainer_kwargs = dict(_accelerator(device), enable_progress_bar=False,
                              enable_model_summary=False, logger=False,
                              enable_checkpointing=False)
        model = self.build(horizon, input_size, loss, trainer_kwargs)
        nf = NeuralForecast(models=[model], freq=1)

        df = to_long_df(train)
        t0 = time.perf_counter()
        nf.fit(df=df, val_size=horizon)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = nf.predict()
        pred_s = time.perf_counter() - t0
        pred = pred.reset_index() if "unique_id" not in pred.columns else pred

        cols = [f"{self.nf_name}{s}" for s in DECILE_SUFFIX]
        q = wide_from_long(pred, cols, len(train), horizon)
        q = np.sort(q, axis=-1)
        point = q[..., 4]
        return Forecast(point=point, quantiles=q, fit_seconds=fit_s,
                        predict_seconds=pred_s, device=device,
                        n_params=count_params(nf.models[0]),
                        extra={"input_size": input_size, "val_size": horizon, **COMMON,
                               **self.arch})


class LSTMModel(_NeuralForecastModel):
    name = "LSTM"
    nf_name = "LSTM"
    arch = dict(encoder_n_layers=2, encoder_hidden_size=128,
                decoder_hidden_size=128, decoder_layers=2)

    def build(self, horizon, input_size, loss, trainer_kwargs):
        from neuralforecast.models import LSTM
        return LSTM(h=horizon, input_size=input_size, loss=loss,
                    **self.arch, **COMMON, **trainer_kwargs)


class NBEATSModel(_NeuralForecastModel):
    name = "NBEATS"
    nf_name = "NBEATS"
    arch = dict(stack_types=["identity", "trend", "seasonality"],
                n_blocks=[1, 1, 1], mlp_units=[[512, 512]] * 3)

    def build(self, horizon, input_size, loss, trainer_kwargs):
        from neuralforecast.models import NBEATS
        return NBEATS(h=horizon, input_size=input_size, loss=loss,
                      **self.arch, **COMMON, **trainer_kwargs)
