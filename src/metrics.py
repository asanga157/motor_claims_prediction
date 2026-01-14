# src/metrics.py
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(y_true, eps)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def overall_mape(y_true, y_pred, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = max(float(y_true.sum()), eps)
    return (np.abs(y_true - y_pred).sum() / denom) * 100


def directional_mape(y_true, y_pred, direction: str, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if direction == "over":
        mask = y_pred > y_true
    elif direction == "under":
        mask = y_pred < y_true
    else:
        raise ValueError("direction must be 'over' or 'under'")

    if mask.sum() == 0:
        return float("nan")

    denom = np.maximum(y_true[mask], eps)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom) * 100


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "Overall_MAPE_%": overall_mape(y_true, y_pred),
        "OverForecast_MAPE_%": directional_mape(y_true, y_pred, "over"),
        "UnderForecast_MAPE_%": directional_mape(y_true, y_pred, "under"),
    }
