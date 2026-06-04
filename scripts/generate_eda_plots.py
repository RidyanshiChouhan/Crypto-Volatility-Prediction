#!/usr/bin/env python
"""Generate EDA plots for reports/assets/."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_ASSETS_DIR
from src.preprocessing import load_processed

REPORTS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load_processed("processed_combined.csv")

    # Price trends (Plotly)
    fig = px.line(df, x="date", y="close", color="symbol", title="Cryptocurrency Price Trends")
    fig.write_html(str(REPORTS_ASSETS_DIR / "price_trends.html"))
    fig.write_image(str(REPORTS_ASSETS_DIR / "price_trends.png"))

    # Volume
    fig = px.area(df, x="date", y="volume", color="symbol", title="Trading Volume Trends")
    fig.write_html(str(REPORTS_ASSETS_DIR / "volume_trends.html"))
    fig.write_image(str(REPORTS_ASSETS_DIR / "volume_trends.png"))

    # Market cap
    fig = px.line(df, x="date", y="market_cap", color="symbol", title="Market Capitalization")
    fig.write_html(str(REPORTS_ASSETS_DIR / "market_cap_trends.html"))
    fig.write_image(str(REPORTS_ASSETS_DIR / "market_cap_trends.png"))

    # Correlation heatmap (per symbol aggregate features)
    numeric = df.select_dtypes(include="number").drop(columns=["market_cap"], errors="ignore")
    corr = numeric.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu"))
    fig.update_layout(title="Feature Correlation Heatmap", height=800)
    fig.write_html(str(REPORTS_ASSETS_DIR / "correlation_heatmap.html"))
    fig.write_image(str(REPORTS_ASSETS_DIR / "correlation_heatmap.png"))

    # Volatility distribution
    if "rolling_volatility_7" in df.columns:
        fig = px.histogram(df, x="rolling_volatility_7", color="symbol", nbins=50, title="Volatility Distribution")
        fig.write_html(str(REPORTS_ASSETS_DIR / "volatility_distribution.html"))
        fig.write_image(str(REPORTS_ASSETS_DIR / "volatility_distribution.png"))

    # Seaborn comparison
    plt.figure(figsize=(12, 6))
    for symbol in df["symbol"].unique()[:7]:
        sub = df[df["symbol"] == symbol].tail(200)
        plt.plot(sub["date"], sub["close"], label=symbol)
    plt.legend()
    plt.title("Crypto Price Comparison (Recent)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_ASSETS_DIR / "crypto_comparison.png", dpi=150)
    plt.close()

    print(f"EDA plots saved to {REPORTS_ASSETS_DIR}")


if __name__ == "__main__":
    main()
