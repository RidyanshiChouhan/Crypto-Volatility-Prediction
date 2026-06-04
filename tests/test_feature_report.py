"""Tests for feature report and preprocessing load."""

import pandas as pd
import pytest

from src.data_loader import generate_synthetic_crypto_data
from src.feature_engineering import engineer_features, generate_feature_report, compute_feature_importance_correlation
from src.preprocessing import load_processed, save_processed, preprocess_dataset
from src.config import REPORTS_DIR


def test_generate_feature_report(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.REPORTS_DIR", tmp_path)
    df = engineer_features(generate_synthetic_crypto_data("BTC", n_days=100))
    report = generate_feature_report(df, str(tmp_path / "report.md"))
    assert "Feature Engineering Report" in report
    assert (tmp_path / "report.md").exists()


def test_load_processed_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("src.preprocessing.DATA_PROCESSED_DIR", tmp_path)
    raw = generate_synthetic_crypto_data("BTC", n_days=50)
    cleaned, _ = preprocess_dataset(raw, scaler_type="none")
    featured = engineer_features(cleaned)
    save_processed(featured, "test.csv")
    loaded = load_processed("test.csv")
    assert len(loaded) == len(featured)


def test_importance_short_series():
    df = engineer_features(generate_synthetic_crypto_data("BTC", n_days=30))
    imp = compute_feature_importance_correlation(df)
    assert "feature" in imp.columns or imp.empty
