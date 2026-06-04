#!/usr/bin/env python
"""Fast offline bootstrap: synthetic data + train + reports (no API)."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR
from src.data_loader import fetch_and_save_all, generate_synthetic_crypto_data, save_raw
from src.feature_engineering import compute_feature_importance_correlation, engineer_features, generate_feature_report
from src.preprocessing import preprocess_dataset, save_processed
from src.training import run_full_training_pipeline

import pandas as pd


def main() -> None:
    print("Generating synthetic data (7 symbols x 1500 days)...")
    frames = []
    for sym in ["BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE"]:
        df = generate_synthetic_crypto_data(sym, n_days=1500)
        save_raw(df, sym)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(f"Total rows: {len(raw)}")

    cleaned, _ = preprocess_dataset(raw, scaler_type="none", fit_scaler=False)
    featured = engineer_features(cleaned)
    save_processed(featured, "processed_combined.csv")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_feature_report(featured, str(REPORTS_DIR / "feature_engineering_report.md"))
    compute_feature_importance_correlation(featured).to_csv(
        REPORTS_DIR / "feature_correlation_importance.csv", index=False
    )

    print("Training models (fast mode: tune=False)...")
    meta = run_full_training_pipeline(featured, tune=False)
    print(json.dumps({"best": meta["best_model_name"], "n_train": meta["n_train"], "rows": len(raw)}, indent=2))
    print("Done. Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
