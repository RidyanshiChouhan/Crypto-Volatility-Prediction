"""
Streamlit Community Cloud entry point.
Loads app/streamlit_app.py with project root on sys.path.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_deps() -> None:
    """Fail fast with a clear message if Cloud did not install requirements."""
    missing = []
    for pkg in ("streamlit", "pandas", "numpy", "plotly", "sklearn", "xgboost", "lightgbm", "joblib"):
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            "Missing packages: "
            + ", ".join(missing)
            + ". Ensure requirements.txt is at repo root and Python version is 3.11 "
            "(set in Streamlit app settings or .python-version)."
        )


_check_deps()

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "crypto_volatility_dashboard",
    ROOT / "app" / "streamlit_app.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
