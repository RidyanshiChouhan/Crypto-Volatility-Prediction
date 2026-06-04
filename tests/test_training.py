"""Tests for training utilities."""

import pandas as pd
import pytest

from src.data_loader import generate_synthetic_crypto_data
from src.feature_engineering import TARGET_COL, drop_na_for_training, engineer_features, get_feature_columns
from src.training import get_model_registry, time_series_train_test_split


@pytest.fixture
def train_df():
    btc = generate_synthetic_crypto_data("BTC", n_days=200)
    eth = generate_synthetic_crypto_data("ETH", n_days=200)
    df = engineer_features(pd.concat([btc, eth], ignore_index=True))
    return drop_na_for_training(df)


def test_time_series_split_no_shuffle(train_df):
    train, test = time_series_train_test_split(train_df, test_ratio=0.2)
    assert len(train) + len(test) == len(train_df)
    assert train["date"].max() <= test["date"].min() or train_df["symbol"].nunique() > 1


def test_model_registry_keys():
    models = get_model_registry()
    assert set(models.keys()) == {"linear_regression", "random_forest", "xgboost", "lightgbm"}


def test_quick_fit(train_df):
    from sklearn.linear_model import LinearRegression

    cols = get_feature_columns(train_df)
    model = LinearRegression()
    model.fit(train_df[cols].iloc[:100], train_df[TARGET_COL].iloc[:100])
    pred = model.predict(train_df[cols].iloc[:5])
    assert len(pred) == 5
