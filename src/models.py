# src/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    models: Dict[str, Dict[str, Any]] = None  # provided from YAML


def build_models(model_cfg: Dict[str, Dict[str, Any]], random_state: int = 42) -> Dict[str, Any]:
    """
    model_cfg example:
      {"Random_Forest": {"type": "random_forest", "n_estimators": 300, ...}, ...}
    """
    out: Dict[str, Any] = {}

    for name, spec in model_cfg.items():
        mtype = spec.get("type")

        if mtype == "linear_regression":
            out[name] = LinearRegression()

        elif mtype == "random_forest":
            out[name] = RandomForestRegressor(
                n_estimators=spec.get("n_estimators", 300),
                max_depth=spec.get("max_depth", None),
                min_samples_leaf=spec.get("min_samples_leaf", 50),
                random_state=random_state,
                n_jobs=spec.get("n_jobs", -1),
            )

        elif mtype == "gradient_boosting":
            out[name] = GradientBoostingRegressor(
                n_estimators=spec.get("n_estimators", 300),
                learning_rate=spec.get("learning_rate", 0.05),
                max_depth=spec.get("max_depth", 3),
                random_state=random_state,
            )

        else:
            raise ValueError(f"Unknown model type: {mtype} for {name}")

    return out
