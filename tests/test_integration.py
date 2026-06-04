"""Light integration tests for training pipeline."""

import pandas as pd

from src.data_loader import generate_synthetic_crypto_data
from src.feature_engineering import engineer_features, drop_na_for_training, generate_feature_report
from src.preprocessing import preprocess_dataset, save_processed, load_processed
from src.training import run_full_training_pipeline, train_all_models, time_series_train_test_split
from src.feature_engineering import get_feature_columns, TARGET_COL
from src.evaluation import compute_metrics
import numpy as np


def test_full_pipeline_small(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir()
    reports_dir.mkdir()
    monkeypatch.setattr("src.preprocessing.DATA_PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr("src.config.MODELS_DIR", models_dir)
    monkeypatch.setattr("src.config.REPORTS_DIR", reports_dir)
    monkeypatch.setattr("src.config.BEST_MODEL_PATH", models_dir / "best_model.pkl")
    monkeypatch.setattr("src.config.METRICS_PATH", reports_dir / "model_metrics.json")
    monkeypatch.setattr("src.config.COMPARISON_PATH", reports_dir / "model_comparison.csv")
    monkeypatch.setattr("src.config.FEATURE_IMPORTANCE_PATH", reports_dir / "feature_importance.csv")
    monkeypatch.setattr("src.config.TUNED_MODEL_PATH", models_dir / "tuned_model.pkl")
    monkeypatch.setattr("src.training.MODELS_DIR", models_dir)
    monkeypatch.setattr("src.training.REPORTS_DIR", reports_dir)
    monkeypatch.setattr("src.training.BEST_MODEL_PATH", models_dir / "best_model.pkl")
    monkeypatch.setattr("src.training.METRICS_PATH", reports_dir / "model_metrics.json")
    monkeypatch.setattr("src.evaluation.COMPARISON_PATH", reports_dir / "model_comparison.csv")
    monkeypatch.setattr("src.evaluation.METRICS_PATH", reports_dir / "model_metrics.json")
    monkeypatch.setattr("src.training.FEATURE_IMPORTANCE_PATH", reports_dir / "feature_importance.csv")
    monkeypatch.setattr("src.training.TUNED_MODEL_PATH", models_dir / "tuned_model.pkl")

    btc = generate_synthetic_crypto_data("BTC", n_days=120)
    eth = generate_synthetic_crypto_data("ETH", n_days=120)
    raw = pd.concat([btc, eth], ignore_index=True)
    cleaned, _ = preprocess_dataset(raw, scaler_type="none")
    featured = engineer_features(cleaned)
    save_processed(featured, "processed_combined.csv")

    meta = run_full_training_pipeline(featured, tune=False)
    assert meta["best_model_name"] in {"linear_regression", "random_forest", "xgboost", "lightgbm"}
    assert (models_dir / "best_model.pkl").exists()


def test_train_all_models_direct():
    btc = generate_synthetic_crypto_data("BTC", n_days=100)
    df = drop_na_for_training(engineer_features(btc))
    train, test = time_series_train_test_split(df, test_ratio=0.2)
    cols = get_feature_columns(df)
    models, comp, metrics = train_all_models(train, test, cols)
    assert len(models) == 4
    assert len(comp) == 4


def test_compute_metrics_empty():
    m = compute_metrics(np.array([np.nan]), np.array([1.0]))
    assert np.isnan(m["rmse"])
