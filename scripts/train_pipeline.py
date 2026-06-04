#!/usr/bin/env python
"""End-to-end pipeline: load data -> preprocess -> features -> train -> SHAP."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR
from src.data_loader import fetch_and_save_all, load_all_raw
from src.explainability import run_shap_analysis
from src.feature_engineering import (
    compute_feature_importance_correlation,
    engineer_features,
    generate_feature_report,
)
from src.preprocessing import preprocess_dataset, save_processed
from src.config import BEST_MODEL_PATH
from src.training import run_full_training_pipeline, time_series_train_test_split, load_model
from src.feature_engineering import drop_na_for_training, get_feature_columns, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Step 1: Fetch / load data")
    try:
        raw = load_all_raw()
    except FileNotFoundError:
        logger.info("No raw CSV found. Fetching from API / synthetic fallback...")
        raw = fetch_and_save_all(days=365)

    logger.info("Step 2: Preprocess")
    cleaned, _ = preprocess_dataset(raw, scaler_type="none", fit_scaler=False)

    logger.info("Step 3: Feature engineering")
    featured = engineer_features(cleaned)
    save_processed(featured, "processed_combined.csv")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_feature_report(featured, str(REPORTS_DIR / "feature_engineering_report.md"))
    importance = compute_feature_importance_correlation(featured)
    importance.to_csv(REPORTS_DIR / "feature_correlation_importance.csv", index=False)

    logger.info("Step 4: Train models")
    metadata = run_full_training_pipeline(featured, tune=True, tune_model="xgboost")

    logger.info("Step 5: SHAP explainability")
    df = drop_na_for_training(featured)
    feature_cols = get_feature_columns(df)
    train_df, _ = time_series_train_test_split(df)
    model, _ = load_model(BEST_MODEL_PATH)
    run_shap_analysis(model, train_df[feature_cols], feature_cols)

    logger.info("Pipeline complete. Records: %d | Best model: %s", len(df), metadata.get("best_model_name"))


if __name__ == "__main__":
    main()
