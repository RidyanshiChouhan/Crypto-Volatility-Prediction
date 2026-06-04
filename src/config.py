"""Central configuration for paths, symbols, and model defaults."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local MLflow SQLite backend (MLflow 3.x; avoids file-store deprecation on Windows)
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
_mlflow_uri = f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"
os.environ.setdefault("MLFLOW_TRACKING_URI", _mlflow_uri)
MLFLOW_TRACKING_URI = _mlflow_uri

# Directories
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_ASSETS_DIR = REPORTS_DIR / "assets"
DOCS_DIR = PROJECT_ROOT / "docs"

# Supported cryptocurrencies (CoinGecko IDs)
CRYPTO_SYMBOLS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "ADA": "cardano",
    "XRP": "ripple",
    "DOGE": "dogecoin",
}

SYMBOL_DISPLAY = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "BNB": "Binance Coin",
    "SOL": "Solana",
    "ADA": "Cardano",
    "XRP": "XRP",
    "DOGE": "Dogecoin",
}

# Feature engineering
VOLATILITY_WINDOW = 7
TARGET_HORIZON = 1
ROLLING_VOL_WINDOWS = [7, 14, 30]

# Training
RANDOM_STATE = 42
TEST_SIZE_RATIO = 0.2
N_SPLITS_TS = 5

# Model files
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
TUNED_MODEL_PATH = MODELS_DIR / "tuned_model.pkl"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# CoinGecko
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
API_RATE_LIMIT_SLEEP = 1.2

# Risk thresholds (predicted volatility percentiles on training distribution)
RISK_LOW_PERCENTILE = 33
RISK_HIGH_PERCENTILE = 67
