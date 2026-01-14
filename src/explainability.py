# src/explainability.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_model_based_feature_importance(model: Any, feature_names: list[str]) -> Optional[pd.DataFrame]:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        coef = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        return (
            pd.DataFrame({"feature": feature_names, "importance": coef})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return None


def get_permutation_importance_df(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "neg_mean_absolute_error",
) -> pd.DataFrame:
    r = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )
    return (
        pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "importance_std": r.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_top_feature_importance(imp_df: pd.DataFrame, top_n: int = 20, title: str = "Feature Importance") -> None:
    import matplotlib.pyplot as plt

    d = imp_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(4, top_n * 0.25)))
    plt.barh(d["feature"].astype(str), d["importance"].astype(float))
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


# ---------- SHAP (optional dependency) ----------
def _lazy_import_shap():
    try:
        import shap  # type: ignore
        return shap
    except Exception as e:
        raise RuntimeError("SHAP is not installed. Run: pip install shap") from e


def get_shap_background(X_train: pd.DataFrame, max_rows: int = 2000, random_state: int = 42) -> pd.DataFrame:
    if len(X_train) > max_rows:
        return X_train.sample(max_rows, random_state=random_state)
    return X_train


def build_shap_explainer(model: Any, X_train: pd.DataFrame):
    shap = _lazy_import_shap()

    if hasattr(model, "feature_importances_"):
        return shap.TreeExplainer(model)

    if hasattr(model, "coef_"):
        bg = get_shap_background(X_train)
        return shap.LinearExplainer(model, bg)

    bg = get_shap_background(X_train, max_rows=500)
    return shap.Explainer(model.predict, bg)


def shap_global_explanations(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, max_test_rows: int = 3000):
    shap = _lazy_import_shap()

    explainer = build_shap_explainer(model, X_train)
    X_vis = X_test.sample(max_test_rows, random_state=42) if len(X_test) > max_test_rows else X_test

    shap_values = explainer(X_vis)

    shap.summary_plot(shap_values, X_vis, plot_type="bar", show=True)
    shap.summary_plot(shap_values, X_vis, show=True)

    return explainer, shap_values, X_vis


def shap_individual_explanation(explainer: Any, X_row: pd.DataFrame, title: str = "Individual claim explanation"):
    shap = _lazy_import_shap()
    import matplotlib.pyplot as plt

    shap_values_row = explainer(X_row)
    plt.figure()
    shap.plots.waterfall(shap_values_row[0], show=True)
    plt.title(title)
    plt.show()
    return shap_values_row
