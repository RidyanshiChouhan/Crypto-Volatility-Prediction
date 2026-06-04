"""
Inference utilities for volatility prediction and risk classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, RISK_HIGH_PERCENTILE, RISK_LOW_PERCENTILE
from src.feature_engineering import TARGET_COL, add_technical_indicators, get_feature_columns
from src.training import load_model

logger = logging.getLogger(__name__)

RiskLevel = Literal["Low Risk", "Medium Risk", "High Risk"]


class VolatilityPredictor:
    """Load trained model and produce volatility forecasts with risk labels."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or BEST_MODEL_PATH
        self.model = None
        self.metadata: dict = {}
        self.feature_columns: list[str] = []
        self._vol_thresholds: Tuple[float, float] = (0.0, 0.0)

    def load(self) -> "VolatilityPredictor":
        """Load model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run: python scripts/train_pipeline.py"
            )
        self.model, self.metadata = load_model(self.model_path)
        self.feature_columns = self.metadata.get("feature_columns", [])
        self._set_risk_thresholds()
        return self

    def _set_risk_thresholds(self) -> None:
        """Set risk thresholds from training metrics distribution if available."""
        metrics = self.metadata.get("metrics", {})
        vols = []
        for m in metrics.values():
            if "rmse" in m:
                vols.append(m["rmse"])
        if vols:
            low = float(np.percentile(vols, RISK_LOW_PERCENTILE))
            high = float(np.percentile(vols, RISK_HIGH_PERCENTILE))
        else:
            low, high = 0.02, 0.05
        self._vol_thresholds = (low, high)

    def prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Engineer features for a single symbol subset."""
        subset = df[df["symbol"] == symbol].copy()
        if subset.empty:
            raise ValueError(f"No data for symbol {symbol}")
        enriched = add_technical_indicators(subset)
        return enriched.dropna(subset=self.feature_columns or get_feature_columns(enriched)).reset_index(drop=True)

    def predict(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Predict volatility for latest rows of a symbol."""
        if self.model is None:
            self.load()

        enriched = self.prepare_features(df, symbol)
        if enriched.empty:
            raise ValueError("Insufficient data for prediction after feature engineering")

        cols = self.feature_columns or get_feature_columns(enriched)
        X = enriched[cols].iloc[[-1]].values
        pred = float(self.model.predict(X)[0])

        enriched = enriched.copy()
        enriched["predicted_volatility"] = np.nan
        enriched.loc[enriched.index[-1], "predicted_volatility"] = pred
        return enriched

    def classify_risk(self, predicted_volatility: float) -> RiskLevel:
        """Map predicted volatility to risk level."""
        low, high = self._vol_thresholds
        if predicted_volatility <= low:
            return "Low Risk"
        if predicted_volatility >= high:
            return "High Risk"
        return "Medium Risk"

    def predict_with_risk(self, df: pd.DataFrame, symbol: str) -> dict:
        """Return prediction dict with volatility and risk."""
        enriched = self.predict(df, symbol)
        pred_vol = float(enriched["predicted_volatility"].dropna().iloc[-1])
        risk = self.classify_risk(pred_vol)
        latest = enriched.iloc[-1]
        return {
            "symbol": symbol,
            "date": str(latest["date"]),
            "close": float(latest["close"]),
            "volume": float(latest["volume"]),
            "predicted_volatility": pred_vol,
            "risk_level": risk,
            "actual_volatility": float(latest.get("rolling_volatility_7", np.nan)),
        }


def portfolio_risk_analysis(
    holdings: dict[str, float],
    predictions: dict[str, dict],
) -> dict:
    """
    Analyze portfolio exposure and estimated volatility.

    holdings: {symbol: weight or amount}
    predictions: {symbol: predict_with_risk output}
    """
    total = sum(holdings.values())
    if total <= 0:
        return {"error": "Invalid holdings"}

    weights = {s: v / total for s, v in holdings.items()}
    weighted_vol = 0.0
    exposure = []
    risk_counts = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}

    for symbol, weight in weights.items():
        pred = predictions.get(symbol, {})
        vol = pred.get("predicted_volatility", 0.0)
        weighted_vol += weight * vol
        risk = pred.get("risk_level", "Medium Risk")
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        exposure.append(
            {
                "symbol": symbol,
                "weight_pct": round(weight * 100, 2),
                "predicted_volatility": vol,
                "risk_level": risk,
            }
        )

    dominant_risk = max(risk_counts, key=risk_counts.get)
    return {
        "total_holdings": total,
        "estimated_portfolio_volatility": weighted_vol,
        "portfolio_risk": dominant_risk,
        "exposure": exposure,
    }
