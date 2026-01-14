# src/data_preprocessing.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import CleaningConfig


# -------------------------
# Utility / helpers
# -------------------------
def _safe_has_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def _iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    s = series.dropna()
    if s.empty:
        return (np.nan, np.nan)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _parse_dates(df: pd.DataFrame, cols: Iterable[str], dayfirst: bool = True) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", dayfirst=dayfirst)
    return out


# -------------------------
# Profiling / reporting
# -------------------------
def profile_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return a missingness profile (count + pct) per column."""
    miss_cnt = df.isna().sum()
    miss_pct = (miss_cnt / len(df)).replace([np.inf, np.nan], 0.0)
    prof = (
        pd.DataFrame({"missing_count": miss_cnt, "missing_pct": miss_pct})
        .sort_values(["missing_count", "missing_pct"], ascending=False)
    )
    return prof


def profile_cardinality(df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Return unique counts for selected cols (or all object/category/bool cols if not provided)."""
    if cols is None:
        cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    rows = []
    for c in cols:
        if c in df.columns:
            rows.append({"column": c, "n_unique": int(df[c].nunique(dropna=True))})
    return pd.DataFrame(rows).sort_values("n_unique", ascending=False)


def profile_text_fields(df: pd.DataFrame, text_cols: Iterable[str]) -> pd.DataFrame:
    """Return null counts and average length for text fields."""
    rows = []
    for c in text_cols:
        if c in df.columns:
            nulls = int(df[c].isna().sum())
            avg_len = float(df[c].fillna("").astype(str).str.len().mean())
            rows.append({"column": c, "nulls": nulls, "avg_len": avg_len})
    return pd.DataFrame(rows)


# -------------------------
# Cleaning steps (non-destructive: add columns)
# -------------------------
def add_duplicate_flags(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """
    Adds:
      - is_duplicate_row: fully duplicated row
      - is_duplicate_id: duplicated Claim_ID (or chosen id column)
    """
    out = df.copy()
    out["is_duplicate_row"] = out.duplicated(keep="first")
    if id_column in out.columns:
        out["is_duplicate_id"] = out[id_column].duplicated(keep="first")
    else:
        out["is_duplicate_id"] = False
    return out


def add_domain_validity_flags(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Adds domain sanity flags:
      - invalid_negative_costs
      - invalid_nonpositive_hours_or_rate
      - invalid_vehicle_year
      - invalid_vehicle_age
      - invalid_odometer
      - invalid_any_domain
    """
    out = df.copy()

    # Costs non-negative
    cost_cols = [c for c in [cfg.parts_col, cfg.paint_col, cfg.total_col, "Initial_Estimate"] if c in out.columns]
    neg_cost = np.zeros(len(out), dtype=bool)
    for c in cost_cols:
        neg_cost |= (out[c] < 0)
    out["invalid_negative_costs"] = neg_cost

    # Hours / rate > 0
    nonpos = np.zeros(len(out), dtype=bool)
    for c in [cfg.labour_col, cfg.rate_col]:
        if c in out.columns:
            nonpos |= (out[c] <= 0)
    out["invalid_nonpositive_hours_or_rate"] = nonpos

    # Year / age / odometer ranges
    if "Year_of_Manufacture" in out.columns:
        out["invalid_vehicle_year"] = (out["Year_of_Manufacture"] < cfg.year_min) | (out["Year_of_Manufacture"] > cfg.year_max)
    else:
        out["invalid_vehicle_year"] = False

    if "Vehicle_Age" in out.columns:
        out["invalid_vehicle_age"] = (out["Vehicle_Age"] < cfg.vehicle_age_min) | (out["Vehicle_Age"] > cfg.vehicle_age_max)
    else:
        out["invalid_vehicle_age"] = False

    if "Odometer_km" in out.columns:
        out["invalid_odometer"] = (out["Odometer_km"] < cfg.odometer_min) | (out["Odometer_km"] > cfg.odometer_max)
    else:
        out["invalid_odometer"] = False

    out["invalid_any_domain"] = (
        out["invalid_negative_costs"]
        | out["invalid_nonpositive_hours_or_rate"]
        | out["invalid_vehicle_year"]
        | out["invalid_vehicle_age"]
        | out["invalid_odometer"]
    )
    return out


def add_cost_consistency_flags(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Adds:
      - invalid_total_lt_parts_paint
      - suspicious_total_lt_components  (Total < tolerance * (Parts + Paint + Labour*Rate))
      - invalid_any_cost_consistency
    """
    out = df.copy()

    # Total >= Parts + Paint
    if _safe_has_cols(out, [cfg.total_col, cfg.parts_col, cfg.paint_col]):
        parts_paint = out[cfg.parts_col].fillna(0) + out[cfg.paint_col].fillna(0)
        out["invalid_total_lt_parts_paint"] = out[cfg.total_col] < parts_paint
    else:
        out["invalid_total_lt_parts_paint"] = False

    # Total >= tolerance*(Parts + Paint + Labour*Rate)
    if _safe_has_cols(out, [cfg.total_col, cfg.parts_col, cfg.paint_col, cfg.labour_col, cfg.rate_col]):
        comp_total = (
            out[cfg.parts_col].fillna(0)
            + out[cfg.paint_col].fillna(0)
            + out[cfg.labour_col].fillna(0) * out[cfg.rate_col].fillna(0)
        )
        out["suspicious_total_lt_components"] = (out[cfg.total_col] + 1e-6) < (cfg.component_total_tolerance_ratio * comp_total)
    else:
        out["suspicious_total_lt_components"] = False

    out["invalid_any_cost_consistency"] = out["invalid_total_lt_parts_paint"] | out["suspicious_total_lt_components"]
    return out


def add_outlier_flags_iqr(df: pd.DataFrame, columns: Iterable[str], k: float = 1.5) -> pd.DataFrame:
    """
    Adds per-column IQR outlier flags, plus:
      - any_outlier_iqr
    """
    out = df.copy()
    flags: List[str] = []

    for c in columns:
        if c not in out.columns:
            continue
        lo, hi = _iqr_bounds(out[c], k=k)
        col_flag = f"outlier_iqr__{c}"
        out[col_flag] = (out[c] < lo) | (out[c] > hi)
        flags.append(col_flag)

    out["any_outlier_iqr"] = out[flags].any(axis=1) if flags else False
    return out


def add_text_quality_features(df: pd.DataFrame, text_cols: Iterable[str]) -> pd.DataFrame:
    """
    Adds simple text readiness features:
      - text_len__<col>
      - text_is_missing__<col>
    """
    out = df.copy()
    for c in text_cols:
        if c not in out.columns:
            continue
        s = out[c].astype("string")
        out[f"text_is_missing__{c}"] = s.isna()
        out[f"text_len__{c}"] = s.fillna("").str.len().astype(int)
    return out


# -------------------------
# Filtering step (configurable)
# -------------------------
def filter_rows(
    df: pd.DataFrame,
    *,
    drop_if_true: Optional[List[str]] = None,
    drop_duplicates: bool = False,
    drop_duplicate_id: bool = False,
) -> pd.DataFrame:
    """
    Returns a filtered dataframe based on boolean flag columns.
    Separated from flag-creation.

    drop_if_true:
        List of boolean columns; if any are True, row is dropped.
    drop_duplicates:
        Drop fully duplicated rows using pandas drop_duplicates()
    drop_duplicate_id:
        Drop rows where is_duplicate_id == True (keeps first)
    """
    out = df.copy()

    if drop_duplicates:
        out = out.drop_duplicates()

    if drop_duplicate_id and "is_duplicate_id" in out.columns:
        out = out.loc[~out["is_duplicate_id"]].copy()

    if drop_if_true:
        missing_cols = [c for c in drop_if_true if c not in out.columns]
        if missing_cols:
            raise ValueError(f"filter_rows: missing flag columns: {missing_cols}")

        mask_drop = np.zeros(len(out), dtype=bool)
        for c in drop_if_true:
            mask_drop |= out[c].fillna(False).astype(bool).to_numpy()

        out = out.loc[~mask_drop].copy()

    return out


# -------------------------
# Low-cost sanity flags
# -------------------------
def add_low_cost_sanity_flags(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Flags suspiciously low Parts_Cost / Paint_Cost using:
      - absolute floors
      - relative ratio to Total_Repair_Cost
      - severity-conditioned rule (only problematic when Total exceeds threshold)

    Adds:
      - low_parts_abs, low_paint_abs
      - low_parts_rel, low_paint_rel
      - low_parts_high_total, low_paint_high_total
      - suspicious_low_cost
    """
    out = df.copy()

    required = [cfg.parts_col, cfg.paint_col, cfg.total_col]
    if not _safe_has_cols(out, required):
        out["suspicious_low_cost"] = False
        return out

    total = out[cfg.total_col].replace(0, np.nan)  # avoid divide-by-zero
    parts = out[cfg.parts_col]
    paint = out[cfg.paint_col]

    out["low_parts_abs"] = parts < cfg.parts_min_abs
    out["low_paint_abs"] = paint < cfg.paint_min_abs

    out["low_parts_rel"] = (parts / total) < cfg.low_cost_rel_min_ratio
    out["low_paint_rel"] = (paint / total) < cfg.low_cost_rel_min_ratio

    out["low_parts_high_total"] = (out[cfg.total_col] > cfg.low_cost_high_total_threshold) & (parts < cfg.parts_min_abs)
    out["low_paint_high_total"] = (out[cfg.total_col] > cfg.low_cost_high_total_threshold) & (paint < cfg.paint_min_abs)

    out["suspicious_low_cost"] = (
        out["low_parts_high_total"]
        | out["low_paint_high_total"]
        | ((out[cfg.total_col] > cfg.low_cost_high_total_threshold) & out["low_parts_rel"])
        | ((out[cfg.total_col] > cfg.low_cost_high_total_threshold) & out["low_paint_rel"])
    ).fillna(False)

    return out


# -------------------------
# Pipeline runner
# -------------------------
def run_cleaning_pipeline(df: pd.DataFrame, cfg: Optional[CleaningConfig] = None) -> Dict[str, Any]:
    """
    Runs the data pre-processing pipeline and returns:
      - df_enriched: original rows + flags/features
      - reports: lightweight summaries for quick inspection
    """
    cfg = cfg or CleaningConfig()

    # Step 1: types
    df1 = _parse_dates(df, cfg.date_columns, dayfirst=cfg.dayfirst_dates)
    df1 = _coerce_numeric(df1, cfg.numeric_columns)

    # Step 2: flags/features
    df2 = add_duplicate_flags(df1, cfg.id_column)
    df2 = add_domain_validity_flags(df2, cfg)
    df2 = add_cost_consistency_flags(df2, cfg)
    df2 = add_low_cost_sanity_flags(df2, cfg)
    df2 = add_outlier_flags_iqr(df2, cfg.outlier_columns, k=cfg.iqr_k)
    df2 = add_text_quality_features(df2, cfg.text_columns)

    # Reports (keep interview-friendly)
    reports = {
        "missingness": df2.isna().sum().sort_values(ascending=False),
        "flag_counts": df2.filter(regex="^(invalid_|suspicious_|is_duplicate|any_outlier)").sum().sort_values(ascending=False),
    }

    return {"df_enriched": df2, "reports": reports}
