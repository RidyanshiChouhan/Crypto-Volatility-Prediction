"""Tests for training save/load and registry."""

from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LinearRegression

from src.training import save_model, load_model, extract_feature_importance, get_model_registry


def test_save_load_model(tmp_path):
    model = LinearRegression().fit([[1], [2], [3]], [1, 2, 3])
    path = tmp_path / "m.pkl"
    save_model(model, path, {"feature_columns": ["x"]})
    loaded, meta = load_model(path)
    assert hasattr(loaded, "predict")
    assert meta["feature_columns"] == ["x"]


def test_load_legacy_payload(tmp_path):
    path = tmp_path / "legacy.pkl"
    joblib.dump(LinearRegression(), path)
    model, meta = load_model(path)
    assert meta == {}


def test_extract_feature_importance():
    model = LinearRegression().fit([[1, 2], [2, 3], [3, 4]], [1, 2, 3])
    df = extract_feature_importance(model, ["a", "b"])
    assert len(df) == 2


def test_all_models_in_registry():
    assert len(get_model_registry()) == 4
