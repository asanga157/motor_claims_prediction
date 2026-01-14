
"""
End-to-end model training and evaluation pipeline.

This pipeline orchestrates feature engineering, train/test splitting,
model training, evaluation, and explainability in a reproducible,
config-driven manner.

It is intended to demonstrate a production-style workflow suitable
for batch execution and interview walkthroughs.
"""


from __future__ import annotations

import argparse
import logging
import logging.config
from pathlib import Path

import pandas as pd
import yaml

from src.config import feature_config_from_yaml, model_config_from_yaml
from src.feature_engineering import FNOLFeatureEngineer
from src.models import build_models
from src.evaluation import evaluate_models
from src.explainability import (
    get_model_based_feature_importance,
    get_permutation_importance_df,
)

logger = logging.getLogger(__name__)


def setup_logging(logging_yaml_path: str | Path) -> None:
    cfg = yaml.safe_load(Path(logging_yaml_path).read_text(encoding="utf-8"))
    logging.config.dictConfig(cfg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model training + evaluation pipeline (FNOL-safe)")
    p.add_argument("--input", required=True, help="Input CSV (cleaned/filtered dataset)")
    p.add_argument("--feature_config", default="conf/feature_config.yaml")
    p.add_argument("--model_config", default="conf/model_config.yaml")
    p.add_argument("--logging", default="conf/logging.yaml")
    p.add_argument("--outdir", default="analysis/outputs_model")
    p.add_argument("--save_feature_importance", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    logger.info("Loaded input: %s shape=%s", args.input, df.shape)

    feat_cfg = feature_config_from_yaml(args.feature_config)
    mcfg = model_config_from_yaml(args.model_config)

    # Split (use sklearn directly for simplicity)
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df, test_size=float(mcfg.get("test_size", 0.2)), random_state=int(mcfg.get("random_state", 42))
    )

    # Feature engineering (fit on train only)
    fe = FNOLFeatureEngineer(cfg=feat_cfg, dayfirst_dates=True)
    fe.fit(df_train)

    X_train = fe.transform(df_train)
    y_train = df_train[feat_cfg.target_col].astype(float)

    X_test = fe.transform(df_test)
    y_test = df_test[feat_cfg.target_col].astype(float)

    logger.info("X_train=%s X_test=%s", X_train.shape, X_test.shape)

    # Models
    models = build_models(mcfg["models"], random_state=int(mcfg.get("random_state", 42)))

    # Evaluate
    results = evaluate_models(models, X_train, y_train, X_test, y_test)
    results_path = outdir / "model_comparison.csv"
    results.to_csv(results_path, index=False)
    logger.info("Saved model comparison: %s", results_path)

    # Pick best
    best_name = results.iloc[0]["Model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    logger.info("Best model: %s", best_name)

    # Feature importance (model-based or permutation fallback)
    if args.save_feature_importance:
        feature_names = list(X_train.columns)
        imp = get_model_based_feature_importance(best_model, feature_names)
        if imp is None:
            imp = get_permutation_importance_df(best_model, X_test, y_test, n_repeats=5, scoring="neg_mean_absolute_error")
        imp_path = outdir / f"feature_importance__{best_name}.csv"
        imp.to_csv(imp_path, index=False)
        logger.info("Saved feature importance: %s", imp_path)


if __name__ == "__main__":
    main()
