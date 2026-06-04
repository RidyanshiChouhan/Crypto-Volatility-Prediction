"""
SHAP-based explainability for volatility models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPORTS_ASSETS_DIR, REPORTS_DIR
from src.feature_engineering import TARGET_COL, get_feature_columns

logger = logging.getLogger(__name__)


def run_shap_analysis(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    output_dir: Optional[Path] = None,
    max_samples: int = 500,
) -> dict:
    """
    Generate SHAP global and local explanations.

    Saves summary plot, bar plot, and explainability report.
    """
    output_dir = output_dir or REPORTS_ASSETS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X

    if len(X_arr) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_arr), max_samples, replace=False)
        X_sample = X_arr[idx]
    else:
        X_sample = X_arr

    import shap

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        logger.info("TreeExplainer failed, using KernelExplainer subset")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, min(100, len(X_sample))))
        shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    summary_path = output_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (global importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    bar_path = output_dir / "shap_importance_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Local waterfall for last sample
    local_path = output_dir / "shap_local_waterfall.png"
    try:
        if hasattr(shap, "Explanation"):
            exp = shap.Explanation(
                values=shap_values[-1] if isinstance(shap_values, np.ndarray) else shap_values[-1],
                base_values=explainer.expected_value if hasattr(explainer, "expected_value") else 0,
                data=X_sample[-1],
                feature_names=feature_names,
            )
            plt.figure()
            shap.waterfall_plot(exp, show=False)
            plt.savefig(local_path, dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as exc:
        logger.warning("Waterfall plot skipped: %s", exc)
        local_path = None

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": feature_names, "shap_importance": mean_abs})
    importance = importance.sort_values("shap_importance", ascending=False)
    importance.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    report_lines = [
        "# Explainability Report (SHAP)",
        "",
        "## Global Feature Impact",
        "",
        "Top features by mean |SHAP value|:",
        "",
    ]
    for _, row in importance.head(15).iterrows():
        report_lines.append(f"- **{row['feature']}**: {row['shap_importance']:.6f}")

    report_lines.extend(
        [
            "",
            "## Artifacts",
            f"- Summary plot: `{summary_path.name}`",
            f"- Bar plot: `{bar_path.name}`",
        ]
    )
    if local_path:
        report_lines.append(f"- Local waterfall: `{local_path.name}`")

    report_path = REPORTS_DIR / "explainability_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "summary_plot": str(summary_path),
        "bar_plot": str(bar_path),
        "local_plot": str(local_path) if local_path else None,
        "importance": importance,
        "report": str(report_path),
    }
