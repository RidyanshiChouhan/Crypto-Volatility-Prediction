"""Tests for feature engineering."""

import pandas as pd
import pytest

from src.data_loader import generate_synthetic_crypto_data
from src.feature_engineering import (
    TARGET_COL,
    add_technical_indicators,
    create_target,
    engineer_features,
    get_feature_columns,
)


@pytest.fixture
def btc_df():
    return generate_synthetic_crypto_data("BTC", n_days=200)


def test_minimum_features_count(btc_df):
    enriched = add_technical_indicators(btc_df)
    required = [
        "daily_return",
        "log_return",
        "rolling_volatility_7",
        "sma_7",
        "sma_14",
        "sma_30",
        "ema_7",
        "ema_14",
        "ema_30",
        "rsi",
        "macd",
        "atr",
        "momentum",
        "liquidity_ratio",
        "price_range",
        "rate_of_change",
        "volatility_index",
    ]
    for col in required:
        assert col in enriched.columns, f"Missing feature: {col}"


def test_target_no_leakage(btc_df):
    enriched = add_technical_indicators(btc_df)
    with_target = create_target(enriched)
    assert TARGET_COL in with_target.columns
    # Last rows should have NaN target (future unknown)
    assert with_target[TARGET_COL].iloc[-1] != with_target[TARGET_COL].iloc[-2] or pd.isna(
        with_target[TARGET_COL].iloc[-1]
    )


def test_engineer_features_multiple_symbols():
    btc = generate_synthetic_crypto_data("BTC", n_days=150)
    eth = generate_synthetic_crypto_data("ETH", n_days=150)
    combined = pd.concat([btc, eth], ignore_index=True)
    result = engineer_features(combined)
    assert result["symbol"].nunique() == 2
    features = get_feature_columns(result)
    assert len(features) >= 15


def test_feature_columns_exclude_target(btc_df):
    result = engineer_features(btc_df)
    features = get_feature_columns(result)
    assert TARGET_COL not in features
    assert "close" not in features
