"""
Model training, hyperparameter tuning, and MLflow experiment tracking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODELS_DIR,
    N_SPLITS_TS,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE_RATIO,
    TUNED_MODEL_PATH,
)
from src.evaluation import compare_models, compute_metrics, save_comparison, save_metrics, select_best_model
from src.config import MLFLOW_TRACKING_URI
from src.feature_engineering import TARGET_COL, drop_na_for_training, get_feature_columns
from src.model_io import load_model, save_model

logger = logging.getLogger(__name__)


def _mlflow_run():
    """Lazy MLflow init (not required for Streamlit inference)."""
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("crypto_volatility_prediction")
    return mlflow


def time_series_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = TEST_SIZE_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split per symbol then combine (no shuffle)."""
    train_parts, test_parts = [], []
    for _, group in df.groupby("symbol"):
        n = len(group)
        split_idx = int(n * (1 - test_ratio))
        if split_idx < 50:
            split_idx = max(int(n * 0.8), 30)
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])
    return pd.concat(train_parts), pd.concat(test_parts)


def get_model_registry() -> Dict[str, Any]:
    """Return untrained model instances."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def train_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Train all models and return fitted models, comparison, metrics."""
    feature_columns = feature_columns or get_feature_columns(train_df)
    X_train = train_df[feature_columns].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[feature_columns].values
    y_test = test_df[TARGET_COL].values

    models: Dict[str, Any] = {}
    metrics_map: Dict[str, Dict[str, float]] = {}

    mlflow = _mlflow_run()

    for name, model in get_model_registry().items():
        logger.info("Training %s...", name)
        with mlflow.start_run(run_name=name, nested=True):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            metrics_map[name] = metrics
            models[name] = model

            mlflow.log_params({"model": name, "n_features": len(feature_columns)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("%s - RMSE: %.6f, MAE: %.6f, R²: %.4f", name, metrics["rmse"], metrics["mae"], metrics["r2"])

    comparison = compare_models(metrics_map)
    return models, comparison, metrics_map


def hyperparameter_tune(
    train_df: pd.DataFrame,
    model_name: str = "xgboost",
    n_iter: int = 20,
    feature_columns: Optional[list[str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """RandomizedSearchCV with TimeSeriesSplit."""
    feature_columns = feature_columns or get_feature_columns(train_df)
    X = train_df[feature_columns].values
    y = train_df[TARGET_COL].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_TS)

    if model_name == "xgboost":
        estimator = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
    elif model_name == "lightgbm":
        estimator = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
    else:
        estimator = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV RMSE: %.6f", -search.best_score_)
    return search.best_estimator_, search.best_params_


def extract_feature_importance(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).ravel()
    else:
        return pd.DataFrame({"feature": feature_columns, "importance": 0.0})

    return (
        pd.DataFrame({"feature": feature_columns, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_full_training_pipeline(
    df: pd.DataFrame,
    tune: bool = True,
    tune_model: str = "xgboost",
) -> Dict[str, Any]:
    """
    End-to-end training: split, train, compare, tune, save best model.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = drop_na_for_training(df)
    feature_columns = get_feature_columns(df)
    train_df, test_df = time_series_train_test_split(df)

    mlflow = _mlflow_run()
    with mlflow.start_run(run_name="full_pipeline"):
        models, comparison, metrics_map = train_all_models(train_df, test_df, feature_columns)
        save_comparison(comparison)

        best_name = select_best_model(comparison)
        best_model = models[best_name]
        baseline_rmse = metrics_map["linear_regression"]["rmse"]
        best_rmse = metrics_map[best_name]["rmse"]
        improvement = (baseline_rmse - best_rmse) / baseline_rmse * 100 if baseline_rmse else 0

        metadata = {
            "best_model_name": best_name,
            "feature_columns": feature_columns,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "metrics": metrics_map,
            "improvement_over_baseline_pct": improvement,
        }

        if tune:
            tuned_model, best_params = hyperparameter_tune(train_df, model_name=tune_model, feature_columns=feature_columns)
            X_test = test_df[feature_columns].values
            y_test = test_df[TARGET_COL].values
            tuned_metrics = compute_metrics(y_test, tuned_model.predict(X_test))
            metadata["tuned_params"] = best_params
            metadata["tuned_metrics"] = tuned_metrics
            save_model(tuned_model, TUNED_MODEL_PATH, metadata)
            if tuned_metrics["rmse"] < best_rmse:
                best_model = tuned_model
                best_name = f"{tune_model}_tuned"
                metadata["best_model_name"] = best_name

        importance_df = extract_feature_importance(best_model, feature_columns)
        importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

        save_model(best_model, BEST_MODEL_PATH, metadata)
        save_metrics(metadata, METRICS_PATH)

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("improvement_pct", improvement)

    logger.info("Best model: %s | Improvement over linear: %.2f%%", best_name, improvement)
    return metadata
