"""Zero-shot time-series foundation models.

Every wrapper: loads weights once (timed separately as load_seconds), offers
the full in-sample series as context (the model applies its own maximum
context), forecasts the nine deciles, and uses the 0.5 quantile as the point
forecast.  No fine-tuning, no covariates, no per-dataset settings.

HF repositories used (recorded in Forecast.extra["repo"]):
  amazon/chronos-bolt-small, amazon/chronos-bolt-base, amazon/chronos-2
  google/timesfm-2.5-200m-pytorch, google/timesfm-3.0-pytorch
  Salesforce/moirai-2.0-R-small
  thuml/sundial-base-128m
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .base import Forecast, Model, RANDOM_SEED, count_params, pick_device
from ..metrics import QUANTILE_LEVELS

QL = [float(q) for q in QUANTILE_LEVELS]        # 0.1 ... 0.9


def _batches(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _torch_device(device: str):
    import torch
    return torch.device(device)


# ------------------------------------------------------------------ Chronos

class _Chronos(Model):
    probabilistic = True
    repo: str = ""
    batch_size: int = 64

    def _load(self, device):
        import chronos, torch
        from chronos import BaseChronosPipeline
        self.version = chronos.__version__
        t0 = time.perf_counter()
        pipe = BaseChronosPipeline.from_pretrained(self.repo, device_map=device,
                                                   torch_dtype=torch.float32)
        return pipe, time.perf_counter() - t0

    def forecast(self, train, horizon, seasonality):
        import torch
        device = pick_device()
        pipe, load_s = self._load(device)
        n = len(train)
        q_out = np.empty((n, horizon, len(QL)), dtype=float)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, n, self.batch_size):
                chunk = [torch.tensor(np.asarray(s, dtype=np.float32)) for s in train[start:start + self.batch_size]]
                q, _mean = pipe.predict_quantiles(chunk, prediction_length=horizon, quantile_levels=QL)
                q = self._to_array(q)                      # (b, h, 9)
                q_out[start:start + len(chunk)] = q[:, :horizon, :]
        pred_s = time.perf_counter() - t0
        q_out = np.sort(q_out, axis=-1)
        return Forecast(point=q_out[..., 4], quantiles=q_out, load_seconds=load_s,
                        predict_seconds=pred_s, device=device,
                        n_params=count_params(pipe.model),
                        extra={"repo": self.repo, "batch_size": self.batch_size})

    @staticmethod
    def _to_array(q):
        import torch
        if isinstance(q, torch.Tensor):
            return q.float().cpu().numpy()
        # Chronos-2 returns a list of (n_variates, h, 9) tensors
        return np.stack([t[0].float().cpu().numpy() for t in q], axis=0)


class ChronosBoltSmall(_Chronos):
    name = "ChronosBoltSmall"
    repo = "amazon/chronos-bolt-small"


class ChronosBoltBase(_Chronos):
    name = "ChronosBoltBase"
    repo = "amazon/chronos-bolt-base"


class Chronos2(_Chronos):
    name = "Chronos2"
    repo = "amazon/chronos-2"
    batch_size = 32


# ------------------------------------------------------------------ TimesFM

class TimesFM25(Model):
    """TimesFM 2.5 (200M, PyTorch).  quantile output layout: [mean, q0.1..q0.9]."""
    name = "TimesFM"
    probabilistic = True
    repo = "google/timesfm-2.5-200m-pytorch"
    batch_size = 32

    def forecast(self, train, horizon, seasonality):
        import torch, timesfm
        self.version = getattr(timesfm, "__version__", "3.0.1")
        device = pick_device()
        t0 = time.perf_counter()
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo, torch_compile=False)
        max_len = max(len(s) for s in train)
        max_context = int(min(16384, max(64, math.ceil(max_len / 32) * 32)))
        max_horizon = int(max(64, math.ceil(horizon / 64) * 64))
        model.compile(timesfm.ForecastConfig(
            max_context=max_context, max_horizon=max_horizon,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=True,
            fix_quantile_crossing=True, per_core_batch_size=self.batch_size))
        try:
            dev = _torch_device(device)
            model.model.to(dev)
            model.model.device = dev          # the module caches its device
        except Exception:
            device = "cpu"
            model.model.to("cpu"); model.model.device = _torch_device("cpu")
        load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        inputs = [np.asarray(s, dtype=np.float32) for s in train]
        try:
            point, quant = model.forecast(horizon=horizon, inputs=inputs)
        except TypeError as e:
            # TimesFM's decode casts to float64 internally, which MPS cannot
            # hold. Rerun on CPU and record it as a CPU result so the timing
            # column is honest about where it ran.
            if "float64" not in str(e) or device == "cpu":
                raise
            device = "cpu"
            model.model.to("cpu"); model.model.device = _torch_device("cpu")
            t0 = time.perf_counter()
            point, quant = model.forecast(horizon=horizon, inputs=inputs)
        pred_s = time.perf_counter() - t0
        # TimesFM pads the batch up to a multiple of per_core_batch_size and
        # returns the padded rows too; keep only the real series.
        quant = np.asarray(quant, dtype=float)[:len(train), :horizon, 1:10]      # drop the mean slot
        quant = np.sort(quant, axis=-1)
        return Forecast(point=quant[..., 4], quantiles=quant, load_seconds=load_s,
                        predict_seconds=pred_s, device=device,
                        n_params=count_params(model.model),
                        extra={"repo": self.repo, "max_context": max_context,
                               "batch_size": self.batch_size})


class TimesFM3(Model):
    """TimesFM 3.0 (PyTorch), released 2026-08-31."""
    name = "TimesFM3"
    probabilistic = True
    repo = "google/timesfm-3.0-pytorch"
    batch_size = 16

    def forecast(self, train, horizon, seasonality):
        import timesfm
        from timesfm3 import TimesFM3Forecaster
        self.version = getattr(timesfm, "__version__", "3.0.1")
        device = pick_device()
        t0 = time.perf_counter()
        try:
            fm = TimesFM3Forecaster.from_pretrained(self.repo, device=device,
                                                    per_core_batch_size=self.batch_size)
        except Exception:
            device = "cpu"
            fm = TimesFM3Forecaster.from_pretrained(self.repo, device=device,
                                                    per_core_batch_size=self.batch_size)
        load_s = time.perf_counter() - t0
        levels = list(fm.config.quantiles)
        idx = [levels.index(q) for q in QL]

        t0 = time.perf_counter()
        outs = list(fm.predict_batch(contexts=[np.asarray(s, dtype=np.float32) for s in train],
                                     horizon=horizon, return_quantiles=True,
                                     make_positive=False, sort_quantiles=True))
        pred_s = time.perf_counter() - t0
        quant = np.stack([np.asarray(o.quantiles, dtype=float)[:horizon][:, idx] for o in outs], axis=0)
        quant = np.sort(quant, axis=-1)
        return Forecast(point=quant[..., 4], quantiles=quant, load_seconds=load_s,
                        predict_seconds=pred_s, device=device,
                        n_params=count_params(fm.model),
                        extra={"repo": self.repo, "batch_size": self.batch_size,
                               "model_quantiles": levels})


# ------------------------------------------------------------------ Moirai 2

class Moirai2(Model):
    name = "Moirai2"
    probabilistic = True
    repo = "Salesforce/moirai-2.0-R-small"
    batch_size = 64
    max_context = 2048

    def forecast(self, train, horizon, seasonality):
        import torch
        import uni2ts
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        self.version = getattr(uni2ts, "__version__", "2.0.0")
        device = pick_device()
        max_len = max(len(s) for s in train)
        context = int(min(self.max_context, max(32, max_len)))
        t0 = time.perf_counter()
        module = Moirai2Module.from_pretrained(self.repo)
        model = Moirai2Forecast(module=module, prediction_length=horizon,
                                context_length=context, target_dim=1,
                                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        try:
            model = model.to(_torch_device(device))
        except Exception:
            device = "cpu"
        model.eval()
        load_s = time.perf_counter() - t0
        levels = [float(q) for q in module.quantile_levels]
        idx = [levels.index(q) for q in QL]

        n = len(train)
        q_out = np.empty((n, horizon, len(QL)), dtype=float)
        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                chunk = [np.asarray(s, dtype=np.float32) for s in train[start:start + self.batch_size]]
                try:
                    pred = np.asarray(model.predict(past_target=chunk))      # (b, nq, h, 1)
                except TypeError as e:
                    # uni2ts builds a float64 time index that MPS cannot hold.
                    # Fall back to CPU for the rest of the run and say so.
                    if "float64" not in str(e) or device == "cpu":
                        raise
                    device = "cpu"
                    model = model.to(_torch_device("cpu"))
                    pred = np.asarray(model.predict(past_target=chunk))
                pred = pred.reshape(pred.shape[0], pred.shape[1], pred.shape[2], -1)[..., 0]
                q_out[start:start + len(chunk)] = np.transpose(pred[:, idx, :horizon], (0, 2, 1))
        pred_s = time.perf_counter() - t0
        q_out = np.sort(q_out, axis=-1)
        return Forecast(point=q_out[..., 4], quantiles=q_out, load_seconds=load_s,
                        predict_seconds=pred_s, device=device,
                        n_params=count_params(module),
                        extra={"repo": self.repo, "context_length": context,
                               "batch_size": self.batch_size})


# ------------------------------------------------------------------ Sundial

class Sundial(Model):
    """Sundial base 128M via transformers remote code; generative, so deciles
    are empirical quantiles of `num_samples` sampled trajectories."""
    name = "Sundial"
    probabilistic = True
    repo = "thuml/sundial-base-128m"
    batch_size = 32
    num_samples = 100
    max_context = 2880

    def forecast(self, train, horizon, seasonality):
        import torch, transformers
        from transformers import AutoModelForCausalLM
        from transformers.cache_utils import DynamicCache
        # Sundial's remote code was written against the transformers-4 cache
        # API: `seen_tokens`, `get_usable_length()` and `to_legacy_cache()` are
        # all gone in transformers 5. Shim the three rather than pin the whole
        # environment to transformers 4 for one model.
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
        if not hasattr(DynamicCache, "to_legacy_cache"):
            def _to_legacy(self):
                layers = getattr(self, "layers", None)
                if layers is not None:
                    return tuple((l.keys, l.values) for l in layers)
                return tuple(zip(self.key_cache, self.value_cache))
            DynamicCache.to_legacy_cache = _to_legacy
        if not hasattr(DynamicCache, "get_max_length"):
            # A dynamic cache has no fixed size; the old API returned None.
            DynamicCache.get_max_length = lambda self: (
                self.get_max_cache_shape() if hasattr(self, "get_max_cache_shape") else None)
        self.version = transformers.__version__
        device = pick_device()
        torch.manual_seed(RANDOM_SEED)
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(self.repo, trust_remote_code=True)
        try:
            model = model.to(_torch_device(device))
        except Exception:
            device = "cpu"
        model.eval()
        load_s = time.perf_counter() - t0

        n = len(train)
        q_out = np.empty((n, horizon, len(QL)), dtype=float)
        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                chunk = train[start:start + self.batch_size]
                L = min(self.max_context, max(len(s) for s in chunk))
                x = np.zeros((len(chunk), L), dtype=np.float32)
                for i, s in enumerate(chunk):                 # left-pad with first value
                    s = np.asarray(s, dtype=np.float32)[-L:]
                    x[i, L - len(s):] = s
                    x[i, :L - len(s)] = s[0]
                xt = torch.tensor(x, device=_torch_device(device))
                out = model.generate(xt, max_new_tokens=horizon, num_samples=self.num_samples)
                out = out.float().cpu().numpy()                # (b, num_samples, h)
                q_out[start:start + len(chunk)] = np.transpose(
                    np.quantile(out[:, :, :horizon], QL, axis=1), (1, 2, 0))
        pred_s = time.perf_counter() - t0
        return Forecast(point=q_out[..., 4], quantiles=q_out, load_seconds=load_s,
                        predict_seconds=pred_s, device=device,
                        n_params=count_params(model),
                        extra={"repo": self.repo, "num_samples": self.num_samples,
                               "batch_size": self.batch_size, "max_context": self.max_context})
