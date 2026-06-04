"""
Advanced feature engineering with 15+ technical indicators.

Target: next-day volatility Volatility(t+1) without data leakage.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None  # type: ignore

from src.config import TARGET_HORIZON, VOLATILITY_WINDOW

logger = logging.getLogger(__name__)

TARGET_COL = "target_volatility_next"
FEATURE_EXCLUDE = {"date", "symbol", TARGET_COL}


def _daily_return(close: pd.Series) -> pd.Series:
    return close.pct_change()


def _log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window, min_periods=max(2, window // 2)).std()


def add_technical_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """Add 15+ technical indicators for a single symbol time series."""
    df = group.copy().sort_values("date")
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Returns
    df["daily_return"] = _daily_return(close)
    df["log_return"] = _log_return(close)

    # Rolling volatility (historical, not shifted for features)
    df["rolling_volatility_7"] = _rolling_volatility(df["daily_return"], 7)
    df["rolling_volatility_14"] = _rolling_volatility(df["daily_return"], 14)
    df["rolling_volatility_30"] = _rolling_volatility(df["daily_return"], 30)

    # SMA / EMA
    for w in (7, 14, 30):
        df[f"sma_{w}"] = close.rolling(window=w, min_periods=1).mean()
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    if ta is not None:
        df["rsi"] = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["macd"] = macd.iloc[:, 0]
            if macd.shape[1] > 1:
                df["macd_signal"] = macd.iloc[:, 1]
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            df["bb_upper"] = bb.iloc[:, 0]
            df["bb_mid"] = bb.iloc[:, 1]
            df["bb_lower"] = bb.iloc[:, 2]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
        df["atr"] = ta.atr(high, low, close, length=14)
    else:
        # Manual fallbacks without pandas-ta
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20.replace(0, np.nan)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

    # Momentum & ROC
    df["momentum"] = close - close.shift(10)
    df["rate_of_change"] = close.pct_change(periods=10) * 100

    # Liquidity & price range
    df["liquidity_ratio"] = volume / close.replace(0, np.nan)
    df["price_range"] = (high - low) / close.replace(0, np.nan)

    # Volatility index (normalized rolling vol)
    rv7 = df["rolling_volatility_7"]
    df["volatility_index"] = (rv7 - rv7.rolling(30, min_periods=5).mean()) / (
        rv7.rolling(30, min_periods=5).std().replace(0, np.nan)
    )

    # Price relative to moving averages
    df["close_to_sma30"] = close / df["sma_30"].replace(0, np.nan) - 1
    df["volume_sma_ratio"] = volume / volume.rolling(14, min_periods=1).mean().replace(0, np.nan)

    return df


def create_target(df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    """
    Create next-day volatility target without leakage.

    Target at time t = std(daily_return) over window ending at t+horizon,
    i.e. realized volatility known only after t+1.
    """
    out = df.copy()
    # Realized vol using returns from t+1 (shift returns backward so vol at t uses future day)
    out["current_volatility"] = _rolling_volatility(out["daily_return"], VOLATILITY_WINDOW)
    # Next day volatility: shift current_volatility by -horizon (future value)
    out[TARGET_COL] = out.groupby("symbol")["current_volatility"].shift(-horizon)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering across all symbols."""
    frames = []
    for symbol, group in df.groupby("symbol"):
        enriched = add_technical_indicators(group)
        frames.append(enriched)

    combined = pd.concat(frames, ignore_index=True)
    combined = create_target(combined)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature column names."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        TARGET_COL,
        "current_volatility",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "market_cap",
    }
    return [c for c in numeric if c not in exclude]


def drop_na_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN in features or target."""
    feature_cols = get_feature_columns(df)
    subset = feature_cols + [TARGET_COL]
    return df.dropna(subset=subset).reset_index(drop=True)


def compute_feature_importance_correlation(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Pearson correlation of features with target as importance proxy."""
    feature_cols = get_feature_columns(df)
    corrs = []
    for col in feature_cols:
        valid = df[[col, target_col]].dropna()
        if len(valid) < 10:
            continue
        corrs.append(
            {
                "feature": col,
                "importance": abs(valid[col].corr(valid[target_col])),
                "correlation": valid[col].corr(valid[target_col]),
            }
        )
    importance_df = pd.DataFrame(corrs).sort_values("importance", ascending=False)
    return importance_df


def generate_feature_report(df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Generate markdown feature engineering report."""
    feature_cols = get_feature_columns(df)
    importance = compute_feature_importance_correlation(df)

    lines = [
        "# Feature Engineering Report",
        "",
        f"**Total engineered features:** {len(feature_cols)}",
        f"**Training samples:** {len(drop_na_for_training(df))}",
        "",
        "## Feature List",
        "",
    ]
    for i, col in enumerate(feature_cols, 1):
        lines.append(f"{i}. `{col}`")

    lines.extend(["", "## Top 10 Features by |Correlation| with Target", ""])
    if not importance.empty:
        for _, row in importance.head(10).iterrows():
            lines.append(f"- **{row['feature']}**: {row['importance']:.4f}")

    report = "\n".join(lines)
    if output_path:
        from pathlib import Path

        Path(output_path).write_text(report, encoding="utf-8")
    return report
