"""
Cryptocurrency Volatility Prediction System - Streamlit Dashboard

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.styles import CUSTOM_CSS
from app.utils import export_csv, export_pdf_report, filter_by_date_range, risk_color
from src.config import (
    BEST_MODEL_PATH,
    CRYPTO_SYMBOLS,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    SYMBOL_DISPLAY,
)
from src.data_loader import generate_synthetic_crypto_data, get_live_prices, load_all_raw
from src.feature_engineering import engineer_features
from src.predict import VolatilityPredictor, portfolio_risk_analysis
from src.preprocessing import load_processed

st.set_page_config(
    page_title="Crypto Volatility Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_market_data() -> pd.DataFrame:
    """Load processed or raw market data."""
    processed_path = PROJECT_ROOT / "data" / "processed" / "processed_combined.csv"
    if processed_path.exists():
        df = load_processed("processed_combined.csv")
        if "daily_return" not in df.columns:
            df = engineer_features(df)
        return df
    try:
        raw = load_all_raw()
        return engineer_features(raw)
    except FileNotFoundError:
        # Streamlit Cloud: no raw CSV/API — use bundled synthetic data
        frames = [
            generate_synthetic_crypto_data(sym, n_days=800, seed=42)
            for sym in CRYPTO_SYMBOLS
        ]
        combined = pd.concat(frames, ignore_index=True)
        return engineer_features(combined)


@st.cache_resource
def get_predictor() -> VolatilityPredictor:
    predictor = VolatilityPredictor(BEST_MODEL_PATH)
    if BEST_MODEL_PATH.exists():
        predictor.load()
    return predictor


@st.cache_data(ttl=60)
def load_live_prices_cached() -> pd.DataFrame:
    return get_live_prices()


def render_kpi_card(label: str, value: str, col) -> None:
    with col:
        st.markdown(
            f"""
            <div class="glass-card">
                <p class="kpi-label">{label}</p>
                <p class="kpi-value">{value}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.markdown('<p class="hero-title">Cryptocurrency Volatility Prediction</p>', unsafe_allow_html=True)
    st.caption("Production ML system for next-day volatility forecasting | CoinGecko real-time data")

    # Auto-refresh
    refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", 0, 120, 60)
    if refresh_sec > 0:
        st.markdown(f'<meta http-equiv="refresh" content="{refresh_sec}">', unsafe_allow_html=True)

    df = load_market_data()
    symbols = sorted(df["symbol"].unique().tolist())

    st.sidebar.header("Controls")
    selected_symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        symbols,
        format_func=lambda s: f"{s} - {SYMBOL_DISPLAY.get(s, s)}",
    )

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_start = max(min_date, max_date - timedelta(days=180))

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, max_date

    symbol_df = df[df["symbol"] == selected_symbol].copy()
    symbol_df = filter_by_date_range(symbol_df, start_date, end_date)

    # Live prices
    with st.expander("Real-Time Market Data (CoinGecko)", expanded=False):
        try:
            live = load_live_prices_cached()
            st.dataframe(live, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Live data unavailable: {exc}")

    # KPIs
    latest = symbol_df.iloc[-1] if not symbol_df.empty else None
    predictor = get_predictor()
    prediction = None

    if latest is not None and BEST_MODEL_PATH.exists():
        try:
            prediction = predictor.predict_with_risk(df, selected_symbol)
        except Exception as exc:
            st.sidebar.error(f"Prediction error: {exc}")

    col1, col2, col3, col4 = st.columns(4)
    price_str = f"${latest['close']:,.2f}" if latest is not None else "N/A"
    vol_pred = f"{prediction['predicted_volatility']:.4f}" if prediction else "N/A"
    risk_str = prediction["risk_level"] if prediction else "N/A"
    volume_str = f"{latest['volume']:,.0f}" if latest is not None else "N/A"

    render_kpi_card("Current Price", price_str, col1)
    render_kpi_card("Predicted Volatility", vol_pred, col2)
    render_kpi_card("Risk Score", risk_str, col3)
    render_kpi_card("Volume", volume_str, col4)

    # Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Historical Price")
        if not symbol_df.empty:
            fig = px.line(symbol_df, x="date", y="close", title=f"{selected_symbol} Price")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Volatility Forecast")
        if not symbol_df.empty and "rolling_volatility_7" in symbol_df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=symbol_df["date"],
                    y=symbol_df["rolling_volatility_7"],
                    name="Historical Vol (7d)",
                    line=dict(color="#3b82f6"),
                )
            )
            if prediction:
                fig.add_hline(
                    y=prediction["predicted_volatility"],
                    line_dash="dash",
                    line_color="#00d4aa",
                    annotation_text="Predicted Next-Day Vol",
                )
            fig.update_layout(
                template="plotly_dark",
                title="Volatility Trend & Forecast",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Charts row 2
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Volume Trend")
        if not symbol_df.empty:
            fig = px.bar(symbol_df.tail(60), x="date", y="volume", title="Recent Volume")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Feature Importance")
        if FEATURE_IMPORTANCE_PATH.exists():
            imp = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(12)
            fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Top Features")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train models to view feature importance.")

    # Prediction section
    st.markdown("---")
    st.subheader("Volatility Prediction")
    if st.button("Predict Volatility", type="primary"):
        if not BEST_MODEL_PATH.exists():
            st.error("Model not found. Run: `python scripts/train_pipeline.py`")
        elif prediction:
            risk = prediction["risk_level"]
            css_class = {
                "Low Risk": "risk-badge-low",
                "Medium Risk": "risk-badge-medium",
                "High Risk": "risk-badge-high",
            }[risk]
            st.markdown(
                f'<div class="{css_class}" style="display:inline-block;font-size:1.2rem;">{risk}</div>',
                unsafe_allow_html=True,
            )
            st.json(prediction)

    # Portfolio Risk Analyzer
    st.markdown("---")
    st.subheader("Portfolio Risk Analyzer")
    st.caption("Enter holding amounts (USD or units) per asset")

    holdings = {}
    cols = st.columns(min(len(symbols), 4))
    for i, sym in enumerate(symbols[:7]):
        with cols[i % len(cols)]:
            val = st.number_input(sym, min_value=0.0, value=0.0, step=100.0, key=f"hold_{sym}")
            if val > 0:
                holdings[sym] = val

    if st.button("Analyze Portfolio Risk") and holdings:
        preds = {}
        for sym in holdings:
            try:
                preds[sym] = predictor.predict_with_risk(df, sym)
            except Exception:
                preds[sym] = {"predicted_volatility": 0.03, "risk_level": "Medium Risk"}
        analysis = portfolio_risk_analysis(holdings, preds)
        st.metric("Estimated Portfolio Volatility", f"{analysis['estimated_portfolio_volatility']:.4f}")
        st.metric("Portfolio Risk", analysis["portfolio_risk"])
        st.dataframe(pd.DataFrame(analysis["exposure"]), use_container_width=True)

    # Model metrics
    if METRICS_PATH.exists():
        with st.expander("Model Performance Metrics"):
            metrics = json.loads(METRICS_PATH.read_text())
            if "metrics" in metrics:
                st.dataframe(pd.DataFrame(metrics["metrics"]).T, use_container_width=True)

    # Downloads
    st.markdown("---")
    st.subheader("Download Reports")
    dl1, dl2 = st.columns(2)

    with dl1:
        csv_bytes = export_csv(symbol_df)
        st.download_button("Download CSV", csv_bytes, f"{selected_symbol}_data.csv", "text/csv")

    with dl2:
        rows = [["Metric", "Value"]]
        if prediction:
            rows += [
                ["Symbol", prediction["symbol"]],
                ["Predicted Volatility", f"{prediction['predicted_volatility']:.6f}"],
                ["Risk Level", prediction["risk_level"]],
                ["Close Price", f"${prediction['close']:,.2f}"],
            ]
        pdf_bytes = export_pdf_report(
            "Crypto Volatility Report",
            rows,
            f"Asset: {selected_symbol} | Date range: {start_date} to {end_date}",
        )
        st.download_button("Download PDF", pdf_bytes, f"{selected_symbol}_report.pdf", "application/pdf")

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Records:** {len(df):,}\n\n"
        f"**Symbols:** {len(symbols)}\n\n"
        f"**Model:** {'Loaded' if BEST_MODEL_PATH.exists() else 'Not trained'}"
    )


if __name__ == "__main__":
    main()
