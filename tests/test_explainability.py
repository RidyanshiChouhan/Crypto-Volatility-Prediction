"""Tests for SHAP explainability (mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
from src.explainability import run_shap_analysis


@pytest.fixture
def sample_X():
    rng = np.random.default_rng(0)
    X = rng.random((50, 5))
    cols = [f"f{i}" for i in range(5)]
    return pd.DataFrame(X, columns=cols), cols


def test_run_shap_analysis_tree(sample_X, tmp_path):
    X, cols = sample_X
    model = MagicMock()
    model.predict = lambda x: np.mean(x, axis=1)

    with patch("src.explainability.shap.TreeExplainer") as mock_te:
        inst = MagicMock()
        inst.shap_values.return_value = np.random.randn(len(X), len(cols))
        inst.expected_value = 0.0
        mock_te.return_value = inst

        result = run_shap_analysis(model, X, cols, output_dir=tmp_path, max_samples=30)

    assert "summary_plot" in result
    assert (tmp_path / "shap_summary.png").exists()
