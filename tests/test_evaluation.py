"""Tests for evaluation module."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compare_models,
    compute_metrics,
    format_metrics_table,
    save_comparison,
    save_metrics,
    select_best_model,
)


def test_compute_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_compare_and_select():
    results = {
        "m1": {"rmse": 0.05, "mae": 0.04, "r2": 0.8},
        "m2": {"rmse": 0.10, "mae": 0.08, "r2": 0.5},
    }
    df = compare_models(results)
    assert select_best_model(df) == "m1"
    assert "rmse" in df.columns


def test_save_metrics_and_comparison(tmp_path):
    metrics = {"best": "xgboost", "rmse": 0.03}
    comp = compare_models({"a": {"rmse": 0.1, "mae": 0.08, "r2": 0.7}})
    save_metrics(metrics, tmp_path / "m.json")
    save_comparison(comp, tmp_path / "c.csv")
    assert (tmp_path / "m.json").exists()
    assert (tmp_path / "c.csv").exists()


def test_format_metrics_table():
    df = compare_models({"x": {"rmse": 0.1, "mae": 0.05, "r2": 0.9}})
    table = format_metrics_table(df)
    assert "rmse" in table.lower() or "x" in table
