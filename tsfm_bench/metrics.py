"""Forecast accuracy metrics.

MASE and sMAPE follow the official M4 evaluation code
(Mcompetitions/M4-methods, "Benchmarks and Evaluation.R"):

    smape = mean( |y - f| * 200 / (|y| + |f|) )
    masep = mean_{t=m+1..n} |y_t - y_{t-m}|      (in-sample, seasonal period m)
    mase  = mean( |y - f| ) / masep

WQL follows the Chronos definition: the weighted quantile loss summed over
the 9 deciles, normalised by the sum of absolute target values.
"""
from __future__ import annotations

import numpy as np

QUANTILE_LEVELS = np.round(np.arange(0.1, 0.95, 0.1), 1)  # 0.1 ... 0.9


def smape(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Per-step sMAPE, M4 definition. Returns an array of the same shape."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.abs(actual) + np.abs(forecast)
    out = np.zeros_like(denom, dtype=float)
    nz = denom != 0
    out[nz] = np.abs(actual[nz] - forecast[nz]) * 200.0 / denom[nz]
    return out


def mase_scale(insample: np.ndarray, seasonality: int) -> float:
    """MASE denominator: mean in-sample seasonal-naive absolute error."""
    insample = np.asarray(insample, dtype=float)
    if len(insample) <= seasonality:
        raise ValueError(
            f"series of length {len(insample)} too short for seasonality {seasonality}"
        )
    return float(np.mean(np.abs(insample[seasonality:] - insample[:-seasonality])))


def mase(actual, forecast, insample, seasonality: int) -> np.ndarray:
    """Per-step MASE, M4 definition."""
    scale = mase_scale(insample, seasonality)
    if scale == 0:
        return np.full(len(np.asarray(actual)), np.nan)
    return np.abs(np.asarray(actual, float) - np.asarray(forecast, float)) / scale


def quantile_loss(actual: np.ndarray, pred_q: np.ndarray, q: float) -> np.ndarray:
    """Pinball loss at level q. pred_q has the same shape as actual."""
    actual = np.asarray(actual, dtype=float)
    pred_q = np.asarray(pred_q, dtype=float)
    diff = actual - pred_q
    return 2.0 * np.where(diff >= 0, q * diff, (q - 1.0) * diff)


def wql(actual: np.ndarray, quantile_forecasts: np.ndarray,
        levels: np.ndarray = QUANTILE_LEVELS) -> float:
    """Weighted quantile loss (Chronos definition), aggregated over a group.

    actual:              (n_series, h)
    quantile_forecasts:  (n_series, h, n_levels)
    """
    actual = np.asarray(actual, dtype=float)
    qf = np.asarray(quantile_forecasts, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return float("nan")
    total = 0.0
    for i, q in enumerate(levels):
        total += np.sum(quantile_loss(actual, qf[..., i], float(q)))
    return float(total / (len(levels) * denom))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of a prediction interval."""
    actual = np.asarray(actual, dtype=float)
    inside = (actual >= np.asarray(lower, float)) & (actual <= np.asarray(upper, float))
    return float(np.mean(inside))
