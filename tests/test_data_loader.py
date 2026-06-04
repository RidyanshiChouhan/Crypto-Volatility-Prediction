"""Additional data loader tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_loader import (
    _ensure_dirs,
    fetch_and_save_all,
    load_all_raw,
    load_raw_csv,
    save_raw,
    generate_synthetic_crypto_data,
)


def test_ensure_dirs_and_save_load(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data_loader.DATA_RAW_DIR", tmp_path)
    _ensure_dirs()
    df = generate_synthetic_crypto_data("BTC", n_days=30)
    path = save_raw(df, "BTC")
    assert path.exists()
    loaded = load_raw_csv(path)
    assert len(loaded) == 30


def test_load_all_raw(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data_loader.DATA_RAW_DIR", tmp_path)
    df = generate_synthetic_crypto_data("ETH", n_days=20)
    save_raw(df, "ETH")
    combined = load_all_raw(tmp_path)
    assert len(combined) == 20


def test_fetch_and_save_all_synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr("src.data_loader.DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(
        "src.data_loader.CRYPTO_SYMBOLS",
        {"BTC": "bitcoin"},
    )

    with patch("src.data_loader.fetch_ohlc_from_coingecko", side_effect=Exception("offline")):
        result = fetch_and_save_all(symbols={"BTC": "bitcoin"}, use_synthetic_fallback=True)
    assert len(result) >= 1000
    assert (tmp_path / "btc_historical.csv").exists()
