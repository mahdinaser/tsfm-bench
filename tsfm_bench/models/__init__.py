"""Model registry. Wrappers that need optional dependencies import lazily."""
from __future__ import annotations

from .base import Model, Forecast, SeasonalNaive

#: name -> zero-argument factory
REGISTRY: dict[str, callable] = {
    "SeasonalNaive": SeasonalNaive,
}


def _register_optional(name: str, module: str, attr: str) -> None:
    def factory(**kw):
        import importlib
        mod = importlib.import_module(f".{module}", __package__)
        return getattr(mod, attr)(**kw)
    REGISTRY[name] = factory


# Populated as each wrapper is implemented and smoke-tested.
for _n, _m, _a in [
    ("Theta",             "statistical", "Theta"),
    ("AutoETS",           "statistical", "AutoETS"),
    ("AutoARIMA",         "statistical", "AutoARIMA"),
    ("LightGBM",          "ml",          "LightGBMModel"),
    ("LSTM",              "neural",      "LSTMModel"),
    ("NBEATS",            "neural",      "NBEATSModel"),
    ("ChronosBoltSmall",  "foundation",  "ChronosBoltSmall"),
    ("ChronosBoltBase",   "foundation",  "ChronosBoltBase"),
    ("Chronos2",          "foundation",  "Chronos2"),
    ("TimesFM",           "foundation",  "TimesFM"),
    ("Moirai2",           "foundation",  "Moirai2"),
    ("Sundial",           "foundation",  "Sundial"),
]:
    _register_optional(_n, _m, _a)


def get_model(name: str, **kw) -> Model:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
