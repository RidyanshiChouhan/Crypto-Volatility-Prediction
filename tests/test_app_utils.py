"""Tests for Streamlit app utilities."""

import pandas as pd

from app.utils import export_csv, export_pdf_report, filter_by_date_range, risk_color


def test_export_csv():
    df = pd.DataFrame({"a": [1, 2]})
    data = export_csv(df)
    assert b"a" in data


def test_filter_by_date_range():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10), "v": range(10)})
    out = filter_by_date_range(df, "2024-01-03", "2024-01-07")
    assert len(out) == 5


def test_risk_colors():
    assert risk_color("Low Risk") == "#00d4aa"
    assert risk_color("High Risk") == "#ff4757"


def test_export_pdf():
    pdf = export_pdf_report("Test", [["Metric", "Value"], ["RMSE", "0.03"]], "note")
    assert pdf[:4] == b"%PDF"
