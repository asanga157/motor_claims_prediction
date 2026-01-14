"""
Feature engineering for motor repair cost prediction.

Implements a scikit-learn compatible transformer that creates
model-ready features using only information available at
First Notice of Loss (FNOL).
"""



from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class FeatureConfig:
    claim_date_col: str = "Claim_Date"
    target_col: str = "Total_Repair_Cost"
    initial_estimate_col: str = "Initial_Estimate"
    fnol_notes_col: str = "FNOL_Notes"
    photos_col: str = "Photos_Available"

    include_initial_estimate: bool = True

    use_external_text_features: bool = True
    external_text_features_path: str = "data/text_features_with_parts.csv"

    cat_cols: Tuple[str, ...] = (
        "Vehicle_Make",
        "Vehicle_Model",
        "Loss_Cause",
        "Repair_Shop_ID",
        "Region",
        "Liability_Assessment",
    )

    num_cols: Tuple[str, ...] = (
        "Year_of_Manufacture",
        "Vehicle_Age",
        "Odometer_km",
        "Initial_Estimate",
    )

    excluded_cols: Tuple[str, ...] = (
        "Labour_Hours",
        "Parts_Cost",
        "Paint_Cost",
        "Shop_Rate_per_Hour",
        "Repair_Shop_Report",
        "Settlement_Time_Days",
    )

    odometer_bucket_size: int = 10_000
    vehicle_age_bins: Tuple[float, ...] = (-1, 3, 7, 12, 20, np.inf)
    vehicle_age_labels: Tuple[str, ...] = ("0_3", "4_7", "8_12", "13_20", "20_plus")

    min_freq_for_common: int = 50
    smoothing_k: float = 20.0

    use_tfidf: bool = False
    tfidf_max_features: int = 5000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)

    severity_keywords: Tuple[str, ...] = ("minor", "moderate", "severe", "total", "write-off", "airbag")
    impact_keywords: Tuple[str, ...] = ("rear", "front", "side", "head-on", "collision", "impact")
    part_keywords: Tuple[str, ...] = (
        "bumper", "door", "hood", "bonnet", "fender", "quarter", "trunk", "boot",
        "windshield", "windscreen", "mirror", "wheel", "rim", "headlight", "taillight",
        "grille", "roof"
    )


def _to_datetime_safe(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def _safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")


def _clip_lower(s: pd.Series, lower: float = 0.0) -> pd.Series:
    return s.where(s >= lower, lower)


def _bucket_odometer(odometer: pd.Series, bucket_size: int) -> pd.Series:
    o = pd.to_numeric(odometer, errors="coerce")
    b = (np.floor(o / bucket_size) * bucket_size).astype("Int64")
    return b


def _bin_vehicle_age(age: pd.Series, bins: Tuple[float, ...], labels: Tuple[str, ...]) -> pd.Series:
    a = pd.to_numeric(age, errors="coerce")
    return pd.cut(a, bins=bins, labels=labels)


class FNOLFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    FNOL-safe feature engineering transformer.
    - fit(): learns train-only aggregates (smoothed mean target encodings) + TFIDF vocab (optional)
    - transform(): outputs numeric feature matrix aligned to training schema
    """

    def __init__(self, cfg: Optional[FeatureConfig] = None, dayfirst_dates: bool = True):
        self.cfg = cfg or FeatureConfig()
        self.dayfirst_dates = dayfirst_dates

        self.global_cost_mean_: Optional[float] = None
        self.cost_index_maps_: Dict[str, pd.Series] = {}
        self.freq_maps_: Dict[str, pd.Series] = {}

        self.tfidf_: Optional[TfidfVectorizer] = None
        self.tfidf_feature_names_: Optional[List[str]] = None

        self.external_text_features_: Optional[pd.DataFrame] = None
        self.ohe_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        df = X.copy()
        cfg = self.cfg

        # Global mean
        if cfg.target_col in df.columns:
            self.global_cost_mean_ = float(pd.to_numeric(df[cfg.target_col], errors="coerce").mean())
        elif y is not None:
            self.global_cost_mean_ = float(pd.to_numeric(y, errors="coerce").mean())
        else:
            raise ValueError("fit requires target via df[target_col] or y.")

        # Smoothed mean target encoding maps (train-only)
        for col in ["Region", "Repair_Shop_ID", "Vehicle_Make", "Vehicle_Model", "Loss_Cause"]:
            if col in df.columns:
                self.cost_index_maps_[col] = self._fit_smoothed_mean(df, group_col=col)

        # Frequency maps (rarity)
        for col in ["Vehicle_Make", "Vehicle_Model", "Repair_Shop_ID"]:
            if col in df.columns:
                self.freq_maps_[col] = df[col].value_counts(dropna=False)

        # TF-IDF (optional) on FNOL notes
        if cfg.use_tfidf:
            text = _safe_str_series(df.get(cfg.fnol_notes_col, pd.Series("", index=df.index)))
            self.tfidf_ = TfidfVectorizer(
                max_features=cfg.tfidf_max_features,
                ngram_range=cfg.tfidf_ngram_range,
                lowercase=True,
                stop_words="english",
            )
            self.tfidf_.fit(text.tolist())
            self.tfidf_feature_names_ = [f"tfidf__{t}" for t in self.tfidf_.get_feature_names_out().tolist()]

        # External features (train-time load)
        if cfg.use_external_text_features:
            p = Path(cfg.external_text_features_path)
            ext = pd.read_csv(p)
            if "Claim_ID" not in ext.columns:
                raise ValueError("External feature file must contain Claim_ID")
            drop_cols = [c for c in ["FNOL_Notes", "Repair_Shop_Report"] if c in ext.columns]
            ext = ext.drop(columns=drop_cols, errors="ignore").set_index("Claim_ID")
            self.external_text_features_ = ext

        # Learn final schema on train
        feats = self._build_features(df, fit_mode=True)
        self.ohe_columns_ = feats.columns.tolist()
        return self

    def _fit_smoothed_mean(self, df: pd.DataFrame, group_col: str) -> pd.Series:
        cfg = self.cfg
        y = pd.to_numeric(df[cfg.target_col], errors="coerce")
        stats = pd.DataFrame({"y": y, "g": df[group_col]})
        ag = stats.groupby("g")["y"].agg(["sum", "count"])

        k = float(cfg.smoothing_k)
        mu = float(self.global_cost_mean_ or 0.0)
        return (ag["sum"] + k * mu) / (ag["count"] + k)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.ohe_columns_ is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        feats = self._build_features(X.copy(), fit_mode=False)

        # Align schema
        feats = feats.reindex(columns=self.ohe_columns_, fill_value=0.0)

        # Ensure numeric
        for c in feats.columns:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)
        return feats

    def _build_features(self, df: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        cfg = self.cfg
        out = pd.DataFrame(index=df.index)

        if "Claim_ID" not in df.columns:
            raise ValueError("Claim_ID is required in input data.")
        out["Claim_ID"] = df["Claim_ID"].values

        # 1) Time features
        if cfg.claim_date_col in df.columns:
            dt = _to_datetime_safe(df[cfg.claim_date_col], dayfirst=self.dayfirst_dates)
            out["time__year"] = dt.dt.year.fillna(0).astype(int)
            out["time__month"] = dt.dt.month.fillna(0).astype(int)
            out["time__dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
            out["time__is_weekend"] = dt.dt.dayofweek.isin([5, 6]).fillna(False).astype(int)
        else:
            out["time__year"] = 0
            out["time__month"] = 0
            out["time__dayofweek"] = 0
            out["time__is_weekend"] = 0

        # 2) Numeric FNOL features
        for col in cfg.num_cols:
            if (not cfg.include_initial_estimate) and (col == cfg.initial_estimate_col):
                continue
            out[f"num__{col}"] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)

        # estimate transforms (optional)
        if cfg.include_initial_estimate and cfg.initial_estimate_col in df.columns:
            est = pd.to_numeric(df[cfg.initial_estimate_col], errors="coerce").fillna(0.0)
            out["est__log1p"] = np.log1p(_clip_lower(est, 0.0))
            out["est__is_zero"] = (est <= 0).astype(int)
            out["est__rank_pct"] = est.rank(pct=True).fillna(0.0)

        # 3) Buckets
        out["veh__age_bin"] = (
            _bin_vehicle_age(df.get("Vehicle_Age", pd.Series(np.nan, index=df.index)), cfg.vehicle_age_bins, cfg.vehicle_age_labels)
            .astype("string")
            .fillna("unknown")
        )
        out["veh__odometer_bucket"] = (
            _bucket_odometer(df.get("Odometer_km", pd.Series(np.nan, index=df.index)), cfg.odometer_bucket_size)
            .astype("string")
            .fillna("unknown")
        )

        yom = pd.to_numeric(df.get("Year_of_Manufacture", np.nan), errors="coerce")
        out["veh__yom_missing"] = yom.isna().astype(int)
        out["veh__yom"] = yom.fillna(0).astype(int)

        # 4) Photos
        out["fnol__photos_available"] = df.get(cfg.photos_col, False).astype(bool).fillna(False).astype(int)

        # 5) Train-learned indices + rarity
        for col in ["Region", "Repair_Shop_ID", "Vehicle_Make", "Vehicle_Model", "Loss_Cause"]:
            key = f"idx__{col.lower()}_mean_cost"
            if col in df.columns and col in self.cost_index_maps_:
                out[key] = df[col].map(self.cost_index_maps_[col]).fillna(self.global_cost_mean_ or 0.0)
            else:
                out[key] = self.global_cost_mean_ or 0.0

        for col in ["Vehicle_Make", "Vehicle_Model", "Repair_Shop_ID"]:
            cnt_key = f"freq__{col.lower()}_count"
            rare_key = f"freq__{col.lower()}_is_rare"
            if col in df.columns and col in self.freq_maps_:
                freq = self.freq_maps_[col]
                cnt = df[col].map(freq).fillna(0).astype(int)
                out[cnt_key] = cnt
                out[rare_key] = (cnt < cfg.min_freq_for_common).astype(int)
            else:
                out[cnt_key] = 0
                out[rare_key] = 1

        # 6) FNOL notes lightweight NLP
        text = _safe_str_series(df.get(cfg.fnol_notes_col, pd.Series("", index=df.index))).str.lower()
        out["txt__fnol_len_chars"] = text.str.len().astype(int)
        out["txt__fnol_len_words"] = text.str.split().map(len).astype(int)
        out["txt__fnol_is_missing"] = (text.str.len() == 0).astype(int)

        out["txt__severity_kw_count"] = text.apply(lambda s: sum(1 for w in cfg.severity_keywords if w in s)).astype(int)
        out["txt__impact_kw_count"] = text.apply(lambda s: sum(1 for w in cfg.impact_keywords if w in s)).astype(int)
        out["txt__parts_kw_count"] = text.apply(lambda s: sum(1 for w in cfg.part_keywords if w in s)).astype(int)

        out["txt__mentions_airbag"] = text.str.contains(r"\bairbag\b", regex=True).astype(int)
        out["txt__mentions_tow"] = text.str.contains(r"\btow|towed\b", regex=True).astype(int)
        out["txt__mentions_drivable"] = text.str.contains(r"\bdrivable\b", regex=True).astype(int)
        out["txt__mentions_not_drivable"] = text.str.contains(r"\bnot drivable|non[-\s]?drivable\b", regex=True).astype(int)

        # 7) One-hot for selected categoricals + engineered bins
        cat_for_ohe = ["Vehicle_Make", "Loss_Cause", "Region", "Liability_Assessment", "veh__age_bin", "veh__odometer_bucket"]
        ohe_input = pd.DataFrame(index=df.index)
        for c in cat_for_ohe:
            ohe_input[c] = (df[c].astype("string") if c in df.columns else out[c].astype("string"))
        out_ohe = pd.get_dummies(ohe_input, prefix=[f"cat__{c}" for c in cat_for_ohe], dummy_na=True)

        if cfg.use_tfidf and self.tfidf_ is not None:
            X_tfidf = self.tfidf_.transform(text.tolist())
            tfidf_df = pd.DataFrame(X_tfidf.toarray(), index=df.index, columns=self.tfidf_feature_names_)
            out = pd.concat([out, out_ohe, tfidf_df], axis=1)
        else:
            out = pd.concat([out, out_ohe], axis=1)

        # External features join
        if cfg.use_external_text_features:
            if self.external_text_features_ is None:
                raise RuntimeError("External text features enabled but not loaded. Ensure fit() was called.")
            ext = self.external_text_features_.add_prefix("ext__").reset_index()
            out = out.merge(ext, on="Claim_ID", how="left", validate="1:1")
            ext_cols = [c for c in out.columns if c.startswith("ext__")]
            out[ext_cols] = out[ext_cols].fillna(0.0)

        # Drop Claim_ID at the end
        out = out.drop(columns=["Claim_ID"])

        # Ensure numeric
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        return out