"""Tests for prediction and evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import generate_synthetic_crypto_data
from src.evaluation import compute_metrics, compare_models, select_best_model
from src.feature_engineering import TARGET_COL, drop_na_for_training, engineer_features, get_feature_columns
from src.predict import VolatilityPredictor, portfolio_risk_analysis
from src.training import save_model, load_model, time_series_train_test_split
from src.config import BEST_MODEL_PATH
import joblib
from pathlib import Path
import tempfile


@pytest.fixture
def trained_model_setup(tmp_path):
    btc = generate_synthetic_crypto_data("BTC", n_days=300)
    eth = generate_synthetic_crypto_data("ETH", n_days=300)
    df = engineer_features(pd.concat([btc, eth], ignore_index=True))
    df = drop_na_for_training(df)
    feature_cols = get_feature_columns(df)
    train_df, test_df = time_series_train_test_split(df, test_ratio=0.2)

    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
    model.fit(train_df[feature_cols], train_df[TARGET_COL])

    model_path = tmp_path / "test_model.pkl"
    save_model(model, model_path, {"feature_columns": feature_cols})
    return model_path, df, feature_cols


def test_compute_metrics():
    y_true = np.array([0.1, 0.2, 0.15])
    y_pred = np.array([0.11, 0.19, 0.16])
    metrics = compute_metrics(y_true, y_pred)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0


def test_compare_models():
    results = {
        "a": {"rmse": 0.1, "mae": 0.08, "r2": 0.9},
        "b": {"rmse": 0.2, "mae": 0.15, "r2": 0.7},
    }
    comp = compare_models(results)
    assert select_best_model(comp) == "a"


def test_predictor(trained_model_setup):
    model_path, df, _ = trained_model_setup
    predictor = VolatilityPredictor(model_path)
    predictor.load()
    result = predictor.predict_with_risk(df, "BTC")
    assert "predicted_volatility" in result
    assert result["risk_level"] in ("Low Risk", "Medium Risk", "High Risk")


def test_portfolio_risk():
    holdings = {"BTC": 5000, "ETH": 3000}
    preds = {
        "BTC": {"predicted_volatility": 0.02, "risk_level": "Low Risk"},
        "ETH": {"predicted_volatility": 0.06, "risk_level": "High Risk"},
    }
    analysis = portfolio_risk_analysis(holdings, preds)
    assert analysis["estimated_portfolio_volatility"] > 0
    assert len(analysis["exposure"]) == 2
