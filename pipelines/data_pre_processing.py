"""
Data preprocessing and quality checks for motor insurance claims.

This module adds non-destructive data quality flags to identify
duplicates, domain violations, cost inconsistencies, outliers,
and other anomalies commonly found in claims data.

The logic is designed to support configurable filtering strategies
(e.g. hard vs strict) while preserving transparency and auditability.
"""


from __future__ import annotations

import argparse
import logging
import logging.config
from pathlib import Path

import pandas as pd
import yaml

from src.config import cleaning_config_from_yaml
from src.data_preprocessing import filter_rows, run_cleaning_pipeline


logger = logging.getLogger(__name__)


def setup_logging(logging_yaml_path: str | Path) -> None:
    cfg = yaml.safe_load(Path(logging_yaml_path).read_text(encoding="utf-8"))
    logging.config.dictConfig(cfg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data Pre-Processing Pipeline (enrich + optional filtering)")
    p.add_argument("--input", required=True, help="Path to input CSV (e.g., motor_repair_costs.csv)")
    p.add_argument("--config", default="conf/cleaning_config.yaml", help="Path to cleaning config YAML")
    p.add_argument("--logging", default="conf/logging.yaml", help="Path to logging config YAML")
    p.add_argument("--outdir", default="analysis/outputs", help="Output directory")
    p.add_argument(
        "--mode",
        choices=["enrich_only", "hard", "strict"],
        default="hard",
        help="Filtering mode: enrich_only (no drops), hard, or strict",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading input: %s", args.input)
    df_raw = pd.read_csv(args.input)

    cfg = cleaning_config_from_yaml(args.config)
    logger.info("Running cleaning pipeline with config: %s", args.config)

    result = run_cleaning_pipeline(df_raw, cfg=cfg)
    df_enriched = result["df_enriched"]
    reports = result["reports"]

    # Save enriched
    enriched_path = outdir / "df_enriched.csv"
    df_enriched.to_csv(enriched_path, index=False)
    logger.info("Saved enriched dataset: %s (shape=%s)", enriched_path, df_enriched.shape)

    # Save reports
    missing_path = outdir / "report_missingness.csv"
    flag_counts_path = outdir / "report_flag_counts.csv"
    reports["missingness"].to_csv(missing_path, header=["missing_count"])
    reports["flag_counts"].to_csv(flag_counts_path, header=["flag_count"])
    logger.info("Saved reports: %s, %s", missing_path, flag_counts_path)

    if args.mode == "enrich_only":
        logger.info("Mode=enrich_only; no filtering applied.")
        return

    drop_cols = ["invalid_any_domain", "invalid_any_cost_consistency", "is_duplicate_row", "is_duplicate_id"]
    if args.mode == "strict":
        drop_cols.append("suspicious_low_cost")

    df_filtered = filter_rows(df_enriched, drop_if_true=drop_cols)

    filtered_path = outdir / f"df_filtered_{args.mode}.csv"
    df_filtered.to_csv(filtered_path, index=False)
    logger.info("Saved filtered dataset: %s (shape=%s)", filtered_path, df_filtered.shape)

    # Quick log stats for interview narration
    logger.info("Raw shape: %s", df_raw.shape)
    logger.info("Enriched shape: %s", df_enriched.shape)
    logger.info("Filtered shape: %s", df_filtered.shape)
    logger.info("Suspicious low-cost rate: %.4f", float(df_enriched.get("suspicious_low_cost", pd.Series([0])).mean()))


if __name__ == "__main__":
    main()
