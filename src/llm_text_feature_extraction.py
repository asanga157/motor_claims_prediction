
"""
LLM-based text feature extraction for motor insurance claims.

Uses a schema-constrained Large Language Model (LLM) to extract
structured damage information from free-text claim descriptions
and convert it into numeric, model-ready features.

Designed to be FNOL-safe, deterministic, and modular.
"""



from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Requires: pip install openai
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration for OpenAI extraction calls.
    """
    model: str = "gpt-4o-mini"
    max_retries: int = 3
    retry_backoff_s: float = 0.8
    temperature: float = 0.0
    timeout_s: float = 30.0
    cache_enabled: bool = True


# -----------------------------
# JSON schema builder
# -----------------------------
def build_damage_extraction_json_schema(include_make_model: bool = True) -> Dict[str, Any]:
    """
    Builds a Structured Outputs json_schema for damage extraction.

    Output keys (if include_make_model=True):
      make, model, components, damage_types, severity, location

    Output keys (if include_make_model=False):
      components, damage_types, severity, location
    """
    base_props: Dict[str, Any] = {
        "components": {"type": "array", "items": {"type": "string"}},
        "damage_types": {"type": "array", "items": {"type": "string"}},
        "severity": {
            "type": "array",
            "items": {"type": "string", "enum": ["minor", "medium", "severe"]},
        },
        "location": {"type": "array", "items": {"type": "string"}},
    }

    if include_make_model:
        props = {
            "make": {"type": "string"},
            "model": {"type": "string"},
            **base_props,
        }
        required = ["make", "model", "components", "damage_types", "severity", "location"]
    else:
        props = base_props
        required = ["components", "damage_types", "severity", "location"]

    return {
        "name": "damage_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": props,
            "required": required,
        },
    }


# -----------------------------
# Extractor
# -----------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class OpenAIJSONSchemaExtractor:
    """
    Calls OpenAI and returns a Python dict matching the json_schema.

    Uses Structured Outputs (json_schema) for schema-constrained JSON responses.
    """

    def __init__(self, cfg: LLMConfig, json_schema: Dict[str, Any], include_make_model: bool = True):
        self.cfg = cfg
        self.client = OpenAI()
        self.json_schema = json_schema
        self.include_make_model = include_make_model
        self._cache: Dict[str, Dict[str, Any]] = {}

    def extract(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return self._empty()

        key = _sha(text)
        if self.cfg.cache_enabled and key in self._cache:
            return self._cache[key]

        prompt = self._build_prompt(text)

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    messages=[
                        {"role": "system", "content": "You are a precise information extraction engine."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_schema", "json_schema": self.json_schema},
                    timeout=self.cfg.timeout_s,
                )

                content = resp.choices[0].message.content
                obj = json.loads(content)
                obj = self._normalize(obj)

                if self.cfg.cache_enabled:
                    self._cache[key] = obj
                return obj

            except Exception as e:
                last_err = e
                time.sleep(self.cfg.retry_backoff_s * (attempt + 1))

        raise RuntimeError(f"LLM extraction failed after retries. Last error: {last_err}")

    def _build_prompt(self, text: str) -> str:
        schema_str = json.dumps(self.json_schema["schema"], indent=2)
        return f"""
You are a senior motor insurance claims assessor with deep experience in
vehicle damage assessment, repair estimation, and claims triage.

Your task is to read a free-text claim description and extract structured,
factual information that would help estimate repair cost and complexity.
Focus only on information that is explicitly stated or clearly implied.
Do NOT guess or infer details that are not supported by the text.

Return your response as VALID JSON only, strictly following the schema below.
Do not include any explanation, commentary, or additional fields.

SCHEMA (do not change keys or structure):
{schema_str}

EXTRACTION GUIDELINES:
- Only extract vehicle make or model if clearly mentioned in the text.
- Components should refer to physical vehicle parts, not repair actions.
- Damage types should describe the nature of the damage, not repair steps.
- Severity must be one of: minor, medium, severe.
- Location should capture spatial context (front/rear/left/right/etc.).
- If information is missing, return empty strings or empty lists.
- Use lower-case values for list items.
- Do not duplicate items.

CLAIM DESCRIPTION:
\"\"\"
{text}
\"\"\"
""".strip()

    def _empty(self) -> Dict[str, Any]:
        if self.include_make_model:
            return {"make": "", "model": "", "components": [], "damage_types": [], "severity": [], "location": []}
        return {"components": [], "damage_types": [], "severity": [], "location": []}

    def _normalize(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        base = self._empty()

        # Ensure keys exist
        for k in base.keys():
            if k not in obj:
                obj[k] = base[k]

        # Ensure lists are lists + normalize tokens
        for lk in ["components", "damage_types", "severity", "location"]:
            if not isinstance(obj.get(lk), list):
                obj[lk] = []
            obj[lk] = [str(x).strip().lower() for x in obj.get(lk, []) if str(x).strip()]

        if self.include_make_model:
            obj["make"] = str(obj.get("make", "")).strip()
            obj["model"] = str(obj.get("model", "")).strip()

        return obj


# -----------------------------
# Extraction -> numeric features
# -----------------------------
def extraction_to_features(
    ex: Dict[str, Any],
    *,
    prefix: str,
    include_make_model: bool = True,
    damage_vocab: Optional[List[str]] = None,
    component_vocab: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Convert one extraction dict into numeric features.

    Produces:
      - counts: n_components, n_damage_types
      - severity: max/mean severity (minor=1, medium=2, severe=3)
      - location flags: has_front/rear/left/right/any
      - vocab flags for common damages and components
      - optional has_make_in_text/has_model_in_text
    """
    damage_vocab = damage_vocab or ["dent", "scratch", "scratches", "broken", "crack", "bent", "shattered", "misaligned"]
    component_vocab = component_vocab or ["door", "bumper", "hood", "bonnet", "fender", "panel", "wheel", "rim", "mirror", "windshield", "windscreen", "roof", "trunk", "boot"]

    sev_map = {"minor": 1, "medium": 2, "severe": 3}
    sev_scores = [sev_map[s] for s in ex.get("severity", []) if s in sev_map]

    components = set(ex.get("components", []))
    damages = set(ex.get("damage_types", []))
    loc_text = " ".join(ex.get("location", []))

    feats: Dict[str, float] = {
        f"{prefix}n_components": float(len(components)),
        f"{prefix}n_damage_types": float(len(damages)),
        f"{prefix}max_severity": float(max(sev_scores) if sev_scores else 0),
        f"{prefix}mean_severity": float(float(np.mean(sev_scores)) if sev_scores else 0),

        f"{prefix}has_front": float("front" in loc_text),
        f"{prefix}has_rear_back": float(("rear" in loc_text) or ("back" in loc_text)),
        f"{prefix}has_left": float("left" in loc_text),
        f"{prefix}has_right": float("right" in loc_text),
        f"{prefix}has_location": float(1.0 if loc_text.strip() else 0.0),
    }

    # Damage flags
    for dmg in damage_vocab:
        feats[f"{prefix}dmg_{dmg}"] = float(dmg in damages)

    # Component flags
    for part in component_vocab:
        feats[f"{prefix}part_{part}"] = float(part in components)

    # Make/model presence only (avoid high-cardinality raw categories unless you explicitly want them)
    if include_make_model:
        feats[f"{prefix}has_make_in_text"] = float(1.0 if ex.get("make") else 0.0)
        feats[f"{prefix}has_model_in_text"] = float(1.0 if ex.get("model") else 0.0)

    return feats


# -----------------------------
# Convenience: apply to dataframe
# -----------------------------
def add_llm_text_features(
    df: pd.DataFrame,
    *,
    extractor: OpenAIJSONSchemaExtractor,
    text_col: str,
    prefix: str,
    include_make_model: bool = True,
    damage_vocab: Optional[List[str]] = None,
    component_vocab: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Adds LLM-derived numeric features for a given text column.

    This function:
      1) extracts structured JSON per row using extractor.extract(text)
      2) converts it into numeric features via extraction_to_features
      3) joins those features back to the dataframe
    """
    out = df.copy()

    features_list: List[Dict[str, float]] = []
    for txt in out[text_col].fillna("").astype(str).tolist():
        ex = extractor.extract(txt)
        feats = extraction_to_features(
            ex,
            prefix=prefix,
            include_make_model=include_make_model,
            damage_vocab=damage_vocab,
            component_vocab=component_vocab,
        )
        features_list.append(feats)

    feat_df = pd.DataFrame(features_list, index=out.index)
    return pd.concat([out, feat_df], axis=1)