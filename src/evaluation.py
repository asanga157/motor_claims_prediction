# src/evaluation.py
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.metrics import regression_metrics


def evaluate_models(models: Dict[str, Any], X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        tr = regression_metrics(y_train, y_train_pred)
        te = regression_metrics(y_test, y_test_pred)

        rows.append({
            "Model": name,

            "Train_MAE": tr["MAE"],
            "Train_RMSE": tr["RMSE"],
            "Train_R2": tr["R2"],
            "Train_MAPE_%": tr["MAPE_%"],
            "Train_Overall_MAPE_%": tr["Overall_MAPE_%"],
            "Train_OverForecast_MAPE_%": tr["OverForecast_MAPE_%"],
            "Train_UnderForecast_MAPE_%": tr["UnderForecast_MAPE_%"],

            "Test_MAE": te["MAE"],
            "Test_RMSE": te["RMSE"],
            "Test_R2": te["R2"],
            "Test_MAPE_%": te["MAPE_%"],
            "Test_Overall_MAPE_%": te["Overall_MAPE_%"],
            "Test_OverForecast_MAPE_%": te["OverForecast_MAPE_%"],
            "Test_UnderForecast_MAPE_%": te["UnderForecast_MAPE_%"],
        })

    return pd.DataFrame(rows).sort_values("Test_Overall_MAPE_%", ascending=True)
