"""
Configuration loading utilities.

This module provides helper functions to load YAML configuration files
from the conf/ directory and convert them into Python dictionaries or
dataclass-based configuration objects used across the project.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

# add to src/config.py (below cleaning_config_from_yaml)
from src.feature_engineering import FeatureConfig
from src.split import SplitConfig


@dataclass(frozen=True)
class CleaningConfig:
    # Parsing / typing
    date_columns: Tuple[str, ...] = ("Claim_Date",)
    dayfirst_dates: bool = True

    numeric_columns: Tuple[str, ...] = (
        "Year_of_Manufacture",
        "Vehicle_Age",
        "Odometer_km",
        "Labour_Hours",
        "Parts_Cost",
        "Paint_Cost",
        "Shop_Rate_per_Hour",
        "Total_Repair_Cost",
        "Initial_Estimate",
        "Settlement_Time_Days",
    )

    # Duplicate rules
    id_column: str = "Claim_ID"

    # Domain checks
    year_min: int = 1980
    year_max: int = 2026
    vehicle_age_min: int = 0
    vehicle_age_max: int = 60
    odometer_min: int = 0
    odometer_max: int = 1_000_000

    # Outlier detection
    iqr_k: float = 1.5
    outlier_columns: Tuple[str, ...] = (
        "Total_Repair_Cost",
        "Parts_Cost",
        "Labour_Hours",
        "Shop_Rate_per_Hour",
        "Paint_Cost",
    )

    # Consistency checks tolerance
    component_total_tolerance_ratio: float = 0.95

    # Component columns
    parts_col: str = "Parts_Cost"
    paint_col: str = "Paint_Cost"
    labour_col: str = "Labour_Hours"
    rate_col: str = "Shop_Rate_per_Hour"
    total_col: str = "Total_Repair_Cost"

    # Text fields
    text_columns: Tuple[str, ...] = ("FNOL_Notes", "Repair_Shop_Report")

    # Low-cost sanity thresholds
    parts_min_abs: float = 20.0
    paint_min_abs: float = 30.0
    low_cost_rel_min_ratio: float = 0.01
    low_cost_high_total_threshold: float = 1500.0


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cleaning_config_from_yaml(path: str | Path) -> CleaningConfig:
    raw = load_yaml(path)

    # Convert list->tuple for tuple-typed fields
    def _tuple(key: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
        v = raw.get(key, list(default))
        return tuple(v) if isinstance(v, (list, tuple)) else default

    kwargs: Dict[str, Any] = dict(raw)
    kwargs["date_columns"] = _tuple("date_columns", CleaningConfig().date_columns)
    kwargs["numeric_columns"] = _tuple("numeric_columns", CleaningConfig().numeric_columns)
    kwargs["outlier_columns"] = _tuple("outlier_columns", CleaningConfig().outlier_columns)
    kwargs["text_columns"] = _tuple("text_columns", CleaningConfig().text_columns)

    return CleaningConfig(**kwargs)

def feature_config_from_yaml(path: str | Path) -> FeatureConfig:
    raw = load_yaml(path)

    def _tuple(key: str, default):
        v = raw.get(key, list(default))
        return tuple(v) if isinstance(v, (list, tuple)) else default

    kwargs = dict(raw)
    # convert list to tuple
    kwargs["cat_cols"] = _tuple("cat_cols", FeatureConfig().cat_cols)
    kwargs["num_cols"] = _tuple("num_cols", FeatureConfig().num_cols)
    kwargs["excluded_cols"] = _tuple("excluded_cols", FeatureConfig().excluded_cols)
    kwargs["vehicle_age_bins"] = tuple(raw.get("vehicle_age_bins", FeatureConfig().vehicle_age_bins))
    kwargs["vehicle_age_labels"] = tuple(raw.get("vehicle_age_labels", FeatureConfig().vehicle_age_labels))
    kwargs["tfidf_ngram_range"] = tuple(raw.get("tfidf_ngram_range", FeatureConfig().tfidf_ngram_range))
    kwargs["severity_keywords"] = _tuple("severity_keywords", FeatureConfig().severity_keywords)
    kwargs["impact_keywords"] = _tuple("impact_keywords", FeatureConfig().impact_keywords)
    kwargs["part_keywords"] = _tuple("part_keywords", FeatureConfig().part_keywords)

    return FeatureConfig(**kwargs)


def model_config_from_yaml(path: str | Path) -> dict:
    return load_yaml(path)