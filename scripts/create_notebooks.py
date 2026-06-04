#!/usr/bin/env python
"""Generate Jupyter notebooks for EDA, Feature Engineering, and Modeling."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def make_notebook(cells: list[dict], filename: str) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }
    path = NOTEBOOKS_DIR / filename
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Created {path}")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.split("\n"),
        "outputs": [],
        "execution_count": None,
    }


if __name__ == "__main__":
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    make_notebook(
        [
            md("# 01 - Exploratory Data Analysis\n\nProfessional EDA for cryptocurrency volatility prediction."),
            code(
                """import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import fetch_and_save_all, load_all_raw
from src.preprocessing import preprocess_dataset
from src.feature_engineering import engineer_features
from src.config import REPORTS_ASSETS_DIR

REPORTS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

try:
    df = load_all_raw()
except FileNotFoundError:
    df = fetch_and_save_all(days=365)

df, _ = preprocess_dataset(df, scaler_type='none', fit_scaler=False)
df = engineer_features(df)
df.head()"""
            ),
            md("## Dataset Overview"),
            code(
                """print('Shape:', df.shape)
print('\\nDtypes:\\n', df.dtypes)
print('\\nMissing values:\\n', df.isnull().sum().sort_values(ascending=False).head(15))"""
            ),
            md("## Statistical Summary"),
            code("df.describe().T"),
            md("## Price Trends"),
            code(
                """fig = px.line(df, x='date', y='close', color='symbol', title='Price Trends')
fig.write_html(str(REPORTS_ASSETS_DIR / 'price_trends.html'))
fig.write_image(str(REPORTS_ASSETS_DIR / 'price_trends.png'))
fig.show()"""
            ),
            md("## Volume & Market Cap"),
            code(
                """fig = px.area(df, x='date', y='volume', color='symbol', title='Volume Trends')
fig.write_image(str(REPORTS_ASSETS_DIR / 'volume_trends.png'))
fig.show()

fig = px.line(df, x='date', y='market_cap', color='symbol', title='Market Cap')
fig.write_image(str(REPORTS_ASSETS_DIR / 'market_cap_trends.png'))
fig.show()"""
            ),
            md("## Correlation Heatmap"),
            code(
                """numeric = df.select_dtypes(include='number')
corr = numeric.corr()
fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu'))
fig.update_layout(title='Correlation Heatmap', height=800)
fig.write_image(str(REPORTS_ASSETS_DIR / 'correlation_heatmap.png'))
fig.show()"""
            ),
            md("## Volatility Distribution"),
            code(
                """fig = px.histogram(df, x='rolling_volatility_7', color='symbol', nbins=50, title='Volatility Distribution')
fig.write_image(str(REPORTS_ASSETS_DIR / 'volatility_distribution.png'))
fig.show()"""
            ),
        ],
        "01_EDA.ipynb",
    )

    make_notebook(
        [
            md("# 02 - Feature Engineering\n\n15+ technical indicators and target creation."),
            code(
                """import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_processed
from src.feature_engineering import (
    engineer_features, get_feature_columns, drop_na_for_training,
    compute_feature_importance_correlation, generate_feature_report, TARGET_COL
)
from src.config import REPORTS_DIR

try:
    df = load_processed('processed_combined.csv')
except FileNotFoundError:
    from src.data_loader import fetch_and_save_all
    from src.preprocessing import preprocess_dataset
    raw = fetch_and_save_all(days=365)
    df, _ = preprocess_dataset(raw, scaler_type='none')
    df = engineer_features(df)

feature_cols = get_feature_columns(df)
print(f'Features: {len(feature_cols)}')
print(feature_cols)"""
            ),
            md("## Feature Importance (Correlation)"),
            code(
                """importance = compute_feature_importance_correlation(df)
importance.head(15)

report = generate_feature_report(df, str(REPORTS_DIR / 'feature_engineering_report.md'))
print(report[:500])"""
            ),
            md("## Target Variable"),
            code(
                """train_df = drop_na_for_training(df)
print('Samples:', len(train_df))
train_df[['date', 'symbol', 'rolling_volatility_7', TARGET_COL]].tail(10)"""
            ),
        ],
        "02_Feature_Engineering.ipynb",
    )

    make_notebook(
        [
            md("# 03 - Modeling\n\nTrain, compare, tune, and evaluate ML models."),
            code(
                """import sys
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_processed
from src.feature_engineering import engineer_features
from src.training import run_full_training_pipeline
from src.config import COMPARISON_PATH, METRICS_PATH, BEST_MODEL_PATH

try:
    df = load_processed('processed_combined.csv')
except FileNotFoundError:
    from src.data_loader import fetch_and_save_all
    from src.preprocessing import preprocess_dataset
    raw = fetch_and_save_all(days=365)
    df, _ = preprocess_dataset(raw, scaler_type='none')
    df = engineer_features(df)

metadata = run_full_training_pipeline(df, tune=True)
print('Best model:', metadata['best_model_name'])
print('Improvement over baseline:', round(metadata['improvement_over_baseline_pct'], 2), '%')"""
            ),
            md("## Model Comparison"),
            code(
                """comparison = pd.read_csv(COMPARISON_PATH)
comparison

import plotly.express as px
fig = px.bar(comparison, x='model', y=['rmse', 'mae'], barmode='group', title='Model Metrics')
fig.show()"""
            ),
            md("## Metrics"),
            code("json.loads(METRICS_PATH.read_text())"),
        ],
        "03_Modeling.ipynb",
    )
