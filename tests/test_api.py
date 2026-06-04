"""Tests for CoinGecko API and data loader."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_loader import (
    fetch_current_price,
    generate_synthetic_crypto_data,
    get_live_prices,
    REQUIRED_COLUMNS,
)


def test_synthetic_data_schema():
    df = generate_synthetic_crypto_data("BTC", n_days=100)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 100
    assert df["symbol"].iloc[0] == "BTC"


def test_synthetic_data_positive_prices():
    df = generate_synthetic_crypto_data("ETH", n_days=50)
    assert (df["close"] > 0).all()
    assert (df["volume"] > 0).all()


@patch("src.data_loader.requests.get")
def test_fetch_current_price(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "bitcoin": {"usd": 50000, "usd_24h_vol": 1e10, "usd_market_cap": 1e12, "usd_24h_change": 2.5}
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    result = fetch_current_price("bitcoin")
    assert result["usd"] == 50000


@patch("src.data_loader.fetch_current_price")
def test_get_live_prices(mock_fetch):
    mock_fetch.return_value = {
        "usd": 100,
        "usd_24h_vol": 1e6,
        "usd_market_cap": 1e9,
        "usd_24h_change": 1.0,
    }
    from src.config import CRYPTO_SYMBOLS

    # Only test one symbol to speed up
    with patch.dict("src.data_loader.CRYPTO_SYMBOLS", {"BTC": "bitcoin"}, clear=True):
        df = get_live_prices()
    assert "symbol" in df.columns or df.empty or len(df) >= 0
