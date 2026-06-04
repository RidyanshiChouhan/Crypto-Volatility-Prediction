"""
Model evaluation metrics and comparison utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import COMPARISON_PATH, METRICS_PATH, REPORTS_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and R²."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Build comparison table from model name -> metrics dict."""
    rows = []
    for name, metrics in results.items():
        rows.append({"model": name, **metrics})
    df = pd.DataFrame(rows)
    return df.sort_values("rmse").reset_index(drop=True)


def select_best_model(comparison_df: pd.DataFrame) -> str:
    """Select model with lowest RMSE."""
    if comparison_df.empty:
        raise ValueError("Empty comparison dataframe")
    return str(comparison_df.iloc[0]["model"])


def save_metrics(metrics: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Save metrics JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or METRICS_PATH
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_comparison(comparison_df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save comparison CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or COMPARISON_PATH
    comparison_df.to_csv(path, index=False)
    return path


def format_metrics_table(comparison_df: pd.DataFrame) -> str:
    """Markdown table for README/reports."""
    try:
        return comparison_df.to_markdown(index=False, floatfmt=".6f")
    except ImportError:
        lines = ["| " + " | ".join(comparison_df.columns) + " |"]
        lines.append("| " + " | ".join(["---"] * len(comparison_df.columns)) + " |")
        for _, row in comparison_df.iterrows():
            lines.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(lines)
