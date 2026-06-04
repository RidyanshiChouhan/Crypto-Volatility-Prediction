"""
Cryptocurrency Volatility Prediction System - Streamlit Dashboard

Run locally: streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import traceback
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from app.styles import CUSTOM_CSS
from app.utils import export_csv, export_pdf_report, filter_by_date_range, risk_color
from src.config import (
    BEST_MODEL_PATH,
    CRYPTO_SYMBOLS,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    SYMBOL_DISPLAY,
)
from src.data_loader import generate_synthetic_crypto_data, get_live_prices
from src.feature_engineering import engineer_features
from src.predict import VolatilityPredictor, portfolio_risk_analysis

st.set_page_config(
    page_title="Crypto Volatility Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_market_data() -> pd.DataFrame:
    """Load market data (processed CSV, or synthetic fallback for Cloud)."""
    processed_path = PROJECT_ROOT / "data" / "processed" / "processed_combined.csv"
    if processed_path.exists():
        from src.preprocessing import load_processed

        df = load_processed("processed_combined.csv")
        if "daily_return" not in df.columns:
            df = engineer_features(df)
        return df

    frames = [
        generate_synthetic_crypto_data(sym, n_days=800, seed=42)
        for sym in CRYPTO_SYMBOLS.keys()
    ]
    return engineer_features(pd.concat(frames, ignore_index=True))


@st.cache_resource(show_spinner=False)
def get_predictor() -> VolatilityPredictor | None:
    """Load ML model once; return None if missing."""
    if not BEST_MODEL_PATH.exists():
        return None
    try:
        predictor = VolatilityPredictor(BEST_MODEL_PATH)
        predictor.load()
        return predictor
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def load_live_prices_cached() -> pd.DataFrame:
    try:
        return get_live_prices()
    except Exception:
        return pd.DataFrame()


def _parse_date_range(
    date_range,
    min_date: date,
    max_date: date,
) -> tuple[date, date]:
    """Normalize Streamlit date_input (tuple or single date)."""
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    elif isinstance(date_range, tuple) and len(date_range) == 1:
        start = end = date_range[0]
    elif hasattr(date_range, "year"):
        start = end = date_range
    else:
        start = max(min_date, max_date - timedelta(days=180))
        end = max_date
    if start > end:
        start, end = end, start
    return start, end


def _safe_predict(predictor: VolatilityPredictor | None, df: pd.DataFrame, symbol: str) -> dict | None:
    if predictor is None:
        return None
    try:
        return predictor.predict_with_risk(df, symbol)
    except Exception as exc:
        st.warning(f"Prediction unavailable for {symbol}: {exc}")
        return None


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
    st.markdown(
        '<p class="hero-title">Cryptocurrency Volatility Prediction</p>',
        unsafe_allow_html=True,
    )
    st.caption("Production ML system for next-day volatility forecasting")

    try:
        df = load_market_data()
        if df.empty:
            st.error("No market data available.")
            return

        symbols = sorted(df["symbol"].dropna().unique().tolist())
        if not symbols:
            st.error("No symbols found in dataset.")
            return

        # Sidebar
        st.sidebar.header("Controls")
        selected_symbol = st.sidebar.selectbox(
            "Select Cryptocurrency",
            symbols,
            format_func=lambda s: f"{s} - {SYMBOL_DISPLAY.get(s, s)}",
        )

        min_date = pd.Timestamp(df["date"].min()).date()
        max_date = pd.Timestamp(df["date"].max()).date()
        default_start = max(min_date, max_date - timedelta(days=180))

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        start_date, end_date = _parse_date_range(date_range, min_date, max_date)

        symbol_df = df[df["symbol"] == selected_symbol].copy()
        symbol_df = filter_by_date_range(symbol_df, start_date, end_date)

        if symbol_df.empty:
            st.warning("No data in selected date range. Widen the range in the sidebar.")
            symbol_df = df[df["symbol"] == selected_symbol].tail(60)

        predictor = get_predictor()
        prediction = _safe_predict(predictor, df, selected_symbol)
        latest = symbol_df.iloc[-1] if not symbol_df.empty else None

        # Live prices (optional)
        with st.expander("Real-Time Market Data (CoinGecko)", expanded=False):
            live = load_live_prices_cached()
            if live.empty:
                st.info("Live prices unavailable (API limit or offline). Dashboard uses historical/synthetic data.")
            else:
                st.dataframe(live, use_container_width=True, hide_index=True)

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        price_str = f"${float(latest['close']):,.2f}" if latest is not None else "N/A"
        vol_pred = f"{prediction['predicted_volatility']:.4f}" if prediction else "N/A"
        risk_str = prediction["risk_level"] if prediction else "N/A"
        volume_str = f"{float(latest['volume']):,.0f}" if latest is not None else "N/A"

        render_kpi_card("Current Price", price_str, col1)
        render_kpi_card("Predicted Volatility", vol_pred, col2)
        render_kpi_card("Risk Score", risk_str, col3)
        render_kpi_card("Volume", volume_str, col4)

        if prediction:
            risk = prediction["risk_level"]
            css = {
                "Low Risk": "risk-badge-low",
                "Medium Risk": "risk-badge-medium",
                "High Risk": "risk-badge-high",
            }.get(risk, "risk-badge-medium")
            st.markdown(
                f'<div class="{css}" style="display:inline-block;margin:0.5rem 0;">'
                f"Forecast risk: <strong>{risk}</strong></div>",
                unsafe_allow_html=True,
            )

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Historical Price")
            if not symbol_df.empty:
                fig = px.line(symbol_df, x="date", y="close", title=f"{selected_symbol} Price")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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
                st.info("Feature importance file not bundled with this deployment.")

        # Portfolio — use form so inputs don't break reruns
        st.markdown("---")
        st.subheader("Portfolio Risk Analyzer")
        st.caption("Enter holding amounts per asset, then submit.")

        with st.form("portfolio_form", clear_on_submit=False):
            holdings: dict[str, float] = {}
            form_cols = st.columns(min(len(symbols), 4))
            for i, sym in enumerate(symbols[:7]):
                with form_cols[i % len(form_cols)]:
                    val = st.number_input(sym, min_value=0.0, value=0.0, step=100.0, key=f"hold_{sym}")
                    if val > 0:
                        holdings[sym] = val
            submitted = st.form_submit_button("Analyze Portfolio Risk", type="primary")

        if submitted:
            if not holdings:
                st.warning("Enter at least one holding amount greater than 0.")
            else:
                preds = {}
                for sym in holdings:
                    preds[sym] = _safe_predict(predictor, df, sym) or {
                        "predicted_volatility": 0.03,
                        "risk_level": "Medium Risk",
                    }
                analysis = portfolio_risk_analysis(holdings, preds)
                if "error" in analysis:
                    st.error(analysis["error"])
                else:
                    st.metric("Estimated Portfolio Volatility", f"{analysis['estimated_portfolio_volatility']:.4f}")
                    st.metric("Portfolio Risk", analysis["portfolio_risk"])
                    st.dataframe(pd.DataFrame(analysis["exposure"]), use_container_width=True, hide_index=True)

        # Metrics
        if METRICS_PATH.exists():
            with st.expander("Model Performance Metrics"):
                try:
                    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                    if "metrics" in metrics:
                        st.dataframe(pd.DataFrame(metrics["metrics"]).T, use_container_width=True)
                except json.JSONDecodeError:
                    st.warning("Could not read metrics file.")

        # Downloads
        st.markdown("---")
        st.subheader("Download Reports")
        dl1, dl2 = st.columns(2)
        with dl1:
            if not symbol_df.empty:
                st.download_button(
                    "Download CSV",
                    export_csv(symbol_df),
                    f"{selected_symbol}_data.csv",
                    "text/csv",
                    key="dl_csv",
                )
        with dl2:
            rows = [["Metric", "Value"], ["Symbol", selected_symbol]]
            if prediction:
                rows += [
                    ["Predicted Volatility", f"{prediction['predicted_volatility']:.6f}"],
                    ["Risk Level", prediction["risk_level"]],
                    ["Close Price", f"${prediction['close']:,.2f}"],
                ]
            try:
                st.download_button(
                    "Download PDF",
                    export_pdf_report(
                        "Crypto Volatility Report",
                        rows,
                        f"Asset: {selected_symbol} | {start_date} to {end_date}",
                    ),
                    f"{selected_symbol}_report.pdf",
                    "application/pdf",
                    key="dl_pdf",
                )
            except Exception as exc:
                st.error(f"PDF export failed: {exc}")

        st.sidebar.markdown("---")
        st.sidebar.info(
            f"**Records:** {len(df):,}\n\n"
            f"**Symbols:** {len(symbols)}\n\n"
            f"**Model:** {'Loaded' if predictor else 'Not available'}"
        )

    except Exception:
        st.error("Something went wrong while loading the dashboard.")
        with st.expander("Error details"):
            st.code(traceback.format_exc())


# Only when running app/streamlit_app.py directly (local dev)
if __name__ == "__main__":
    main()
