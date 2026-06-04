"""
Reusable preprocessing pipeline for cryptocurrency market data.

Handles missing values, duplicates, dtypes, outliers, and scaling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

NUMERIC_PRICE_COLS = ["open", "high", "low", "close", "volume", "market_cap"]
ScalerType = Literal["standard", "robust", "none"]


class PreprocessingPipeline:
    """End-to-end preprocessing with fit/transform for time-series data."""

    def __init__(
        self,
        scaler_type: ScalerType = "robust",
        outlier_iqr_multiplier: float = 3.0,
        fill_method: str = "ffill",
    ):
        self.scaler_type = scaler_type
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.fill_method = fill_method
        self.scaler: Optional[StandardScaler | RobustScaler] = None
        self.feature_columns_: list[str] = []
        self.is_fitted_ = False

    def _create_scaler(self) -> StandardScaler | RobustScaler:
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown scaler: {self.scaler_type}")

    @staticmethod
    def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct column types."""
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["symbol"] = out["symbol"].astype(str)
        for col in NUMERIC_PRICE_COLS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate symbol-date rows, keeping last observation."""
        before = len(df)
        out = df.drop_duplicates(subset=["symbol", "date"], keep="last")
        removed = before - len(out)
        if removed:
            logger.info("Removed %d duplicate rows", removed)
        return out

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill within symbol groups, then backward-fill, then median."""
        out = df.copy()
        out = out.sort_values(["symbol", "date"])

        for col in NUMERIC_PRICE_COLS:
            if col not in out.columns:
                continue
            out[col] = out.groupby("symbol")[col].transform(
                lambda s: s.ffill() if self.fill_method == "ffill" else s.bfill()
            )
            out[col] = out.groupby("symbol")[col].transform(lambda s: s.bfill())
            out[col] = out[col].fillna(out[col].median())

        return out

    def clip_outliers(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Clip extreme values using IQR per symbol."""
        out = df.copy()
        columns = columns or [c for c in NUMERIC_PRICE_COLS if c in out.columns]

        for symbol, group_idx in out.groupby("symbol").groups.items():
            mask = out.index.isin(group_idx)
            subset = out.loc[mask, columns]
            q1 = subset.quantile(0.25)
            q3 = subset.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.outlier_iqr_multiplier * iqr
            upper = q3 + self.outlier_iqr_multiplier * iqr
            out.loc[mask, columns] = subset.clip(lower=lower, upper=upper, axis=1)

        return out

    def fit(self, df: pd.DataFrame, feature_columns: list[str]) -> "PreprocessingPipeline":
        """Fit scaler on training feature columns."""
        self.feature_columns_ = feature_columns.copy()
        if self.scaler_type == "none":
            self.is_fitted_ = True
            return self

        self.scaler = self._create_scaler()
        X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        self.scaler.fit(X)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to feature columns."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before transform.")

        out = df.copy()
        if self.scaler_type == "none" or self.scaler is None:
            return out

        X = out[self.feature_columns_].replace([np.inf, -np.inf], np.nan).fillna(0)
        out[self.feature_columns_] = self.scaler.transform(X)
        return out

    def fit_transform(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, feature_columns)
        return self.transform(df)

    def run_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run non-scaling cleaning steps."""
        out = self.fix_dtypes(df)
        out = self.remove_duplicates(out)
        out = self.handle_missing_values(out)
        out = self.clip_outliers(out)
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def preprocess_dataset(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    scaler_type: ScalerType = "robust",
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, PreprocessingPipeline]:
    """
    Full preprocessing: clean + optional scale.

    Returns cleaned dataframe and fitted pipeline.
    """
    pipeline = PreprocessingPipeline(scaler_type=scaler_type)
    cleaned = pipeline.run_cleaning(df)

    if feature_columns and scaler_type != "none":
        if fit_scaler:
            cleaned = pipeline.fit_transform(cleaned, feature_columns)
        else:
            cleaned = pipeline.transform(cleaned)

    return cleaned, pipeline


def save_processed(df: pd.DataFrame, filename: str = "processed_combined.csv") -> Path:
    """Persist processed dataset."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved processed data: %s", path)
    return path


def load_processed(filename: str = "processed_combined.csv") -> pd.DataFrame:
    """Load processed dataset."""
    path = DATA_PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df
