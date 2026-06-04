"""
Data loading and collection for cryptocurrency historical OHLCV data.

Supports CoinGecko API fetch, CSV persistence, and synthetic fallback for offline use.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import (
    API_RATE_LIMIT_SLEEP,
    COINGECKO_BASE_URL,
    CRYPTO_SYMBOLS,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "market_cap",
]


def _ensure_dirs() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlc_from_coingecko(
    coin_id: str,
    symbol: str,
    days: int = 365,
) -> pd.DataFrame:
    """
    Fetch OHLC data from CoinGecko market_chart endpoint.

    CoinGecko OHLC endpoint returns [timestamp, open, high, low, close].
  Volume and market cap are merged from market_chart.
    """
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": min(days, 365)}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    ohlc = response.json()

    if not ohlc:
        raise ValueError(f"No OHLC data returned for {symbol}")

    df = pd.DataFrame(ohlc, columns=["timestamp", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop(columns=["timestamp"])

    time.sleep(API_RATE_LIMIT_SLEEP)

    chart_url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
    chart_params = {"vs_currency": "usd", "days": min(days, 365)}
    chart_resp = requests.get(chart_url, params=chart_params, timeout=30)
    chart_resp.raise_for_status()
    chart = chart_resp.json()

    prices = pd.DataFrame(chart.get("prices", []), columns=["timestamp", "price"])
    volumes = pd.DataFrame(chart.get("total_volumes", []), columns=["timestamp", "volume"])
    caps = pd.DataFrame(chart.get("market_caps", []), columns=["timestamp", "market_cap"])

    if not prices.empty:
        prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        volumes["date"] = pd.to_datetime(volumes["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        caps["date"] = pd.to_datetime(caps["timestamp"], unit="ms", utc=True).dt.tz_localize(None)

        vol_daily = volumes.groupby(volumes["date"].dt.date)["volume"].mean().reset_index()
        vol_daily.columns = ["date", "volume"]
        vol_daily["date"] = pd.to_datetime(vol_daily["date"])

        cap_daily = caps.groupby(caps["date"].dt.date)["market_cap"].mean().reset_index()
        cap_daily.columns = ["date", "market_cap"]
        cap_daily["date"] = pd.to_datetime(cap_daily["date"])

        df["date_only"] = df["date"].dt.date
        vol_daily["date_only"] = vol_daily["date"].dt.date
        cap_daily["date_only"] = cap_daily["date"].dt.date

        df = df.merge(vol_daily[["date_only", "volume"]], on="date_only", how="left")
        df = df.merge(cap_daily[["date_only", "market_cap"]], on="date_only", how="left")
        df = df.drop(columns=["date_only"])
    else:
        df["volume"] = np.nan
        df["market_cap"] = np.nan

    df["symbol"] = symbol
    df = df[REQUIRED_COLUMNS]
    return df.sort_values("date").reset_index(drop=True)


def fetch_current_price(coin_id: str) -> dict:
    """Fetch real-time price snapshot from CoinGecko."""
    url = f"{COINGECKO_BASE_URL}/simple/price"
    params = {
        "ids": coin_id,
        "vs_currencies": "usd",
        "include_24hr_vol": "true",
        "include_market_cap": "true",
        "include_24hr_change": "true",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json().get(coin_id, {})


def generate_synthetic_crypto_data(
    symbol: str,
    n_days: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for offline development and tests."""
    rng = np.random.default_rng(seed + hash(symbol) % 1000)
    dates = pd.date_range(end=datetime.utcnow().date(), periods=n_days, freq="D")

    base_prices = {
        "BTC": 45000,
        "ETH": 2800,
        "BNB": 320,
        "SOL": 95,
        "ADA": 0.45,
        "XRP": 0.55,
        "DOGE": 0.08,
    }
    price = float(base_prices.get(symbol, 100))
    rows = []

    for d in dates:
        daily_return = rng.normal(0.001, 0.03)
        open_p = price
        close_p = price * (1 + daily_return)
        high_p = max(open_p, close_p) * (1 + abs(rng.normal(0, 0.01)))
        low_p = min(open_p, close_p) * (1 - abs(rng.normal(0, 0.01)))
        volume = abs(rng.lognormal(15, 1))
        market_cap = close_p * volume * rng.uniform(0.8, 1.2)

        rows.append(
            {
                "date": d,
                "symbol": symbol,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume,
                "market_cap": market_cap,
            }
        )
        price = close_p

    return pd.DataFrame(rows)


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load a single raw CSV file."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_all_raw(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate all raw CSV files."""
    data_dir = data_dir or DATA_RAW_DIR
    files = list(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = [load_raw_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def save_raw(df: pd.DataFrame, symbol: str) -> Path:
    """Save dataframe to raw data directory."""
    _ensure_dirs()
    path = DATA_RAW_DIR / f"{symbol.lower()}_historical.csv"
    df.to_csv(path, index=False)
    logger.info("Saved raw data: %s (%d rows)", path, len(df))
    return path


def fetch_and_save_all(
    symbols: Optional[dict[str, str]] = None,
    days: int = 365,
    use_synthetic_fallback: bool = True,
) -> pd.DataFrame:
    """
    Fetch data for all configured symbols and persist to data/raw/.

    Falls back to synthetic data if API fails.
    """
    _ensure_dirs()
    symbols = symbols or CRYPTO_SYMBOLS
    all_frames: list[pd.DataFrame] = []

    for symbol, coin_id in symbols.items():
        try:
            logger.info("Fetching %s (%s)...", symbol, coin_id)
            df = fetch_ohlc_from_coingecko(coin_id, symbol, days=days)
        except Exception as exc:
            logger.warning("API fetch failed for %s: %s", symbol, exc)
            if not use_synthetic_fallback:
                raise
            logger.info("Using synthetic data for %s", symbol)
            df = generate_synthetic_crypto_data(symbol, n_days=max(days * 4, 1000))

        save_raw(df, symbol)
        all_frames.append(df)
        time.sleep(API_RATE_LIMIT_SLEEP)

    return pd.concat(all_frames, ignore_index=True)


def get_live_prices(symbols: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """Return current prices for all symbols."""
    symbols = symbols or CRYPTO_SYMBOLS
    rows = []
    for symbol, coin_id in symbols.items():
        try:
            data = fetch_current_price(coin_id)
            rows.append(
                {
                    "symbol": symbol,
                    "price_usd": data.get("usd"),
                    "volume_24h": data.get("usd_24h_vol"),
                    "market_cap": data.get("usd_market_cap"),
                    "change_24h_pct": data.get("usd_24h_change"),
                }
            )
            time.sleep(0.3)
        except Exception as exc:
            logger.warning("Live price failed for %s: %s", symbol, exc)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    combined = fetch_and_save_all(days=365)
    print(f"Collected {len(combined)} total rows across {combined['symbol'].nunique()} symbols")
