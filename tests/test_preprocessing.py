"""Tests for preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import PreprocessingPipeline, preprocess_dataset


@pytest.fixture
def sample_raw_df():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "symbol": ["BTC"] * 100 + ["ETH"] * 100,
            "open": np.random.uniform(100, 200, 200),
            "high": np.random.uniform(200, 250, 200),
            "low": np.random.uniform(80, 100, 200),
            "close": np.random.uniform(100, 200, 200),
            "volume": np.random.uniform(1e6, 1e7, 200),
            "market_cap": np.random.uniform(1e9, 1e10, 200),
        }
    )


def test_fix_dtypes(sample_raw_df):
    pipeline = PreprocessingPipeline()
    result = pipeline.fix_dtypes(sample_raw_df)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["symbol"].dtype == object or str(result["symbol"].dtype) == "string"


def test_remove_duplicates():
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01"],
            "symbol": ["BTC", "BTC"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [1, 2],
            "market_cap": [1, 2],
        }
    )
    pipeline = PreprocessingPipeline()
    result = pipeline.remove_duplicates(df)
    assert len(result) == 1


def test_handle_missing_values(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "close"] = np.nan
    pipeline = PreprocessingPipeline()
    result = pipeline.handle_missing_values(df)
    assert result["close"].isna().sum() == 0


def test_clip_outliers(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "close"] = 1e12
    pipeline = PreprocessingPipeline(outlier_iqr_multiplier=1.5)
    result = pipeline.clip_outliers(df)
    assert result["close"].max() < 1e12


def test_scaler_fit_transform(sample_raw_df):
    cleaned, pipeline = preprocess_dataset(
        sample_raw_df,
        feature_columns=["close", "volume"],
        scaler_type="standard",
        fit_scaler=True,
    )
    assert pipeline.is_fitted_
    assert cleaned["close"].std() < 2
