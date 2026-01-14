"""
Train / test split utilities.

Provides controlled (80% train and 20% test) and reproducible data splitting logic used
during model training and evaluation.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42


def split_train_test(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=cfg.test_size, random_state=cfg.random_state)
