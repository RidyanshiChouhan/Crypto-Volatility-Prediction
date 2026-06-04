"""
Streamlit Community Cloud entry point (root).
Imports the dashboard module which calls main() on load.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.streamlit_app  # noqa: F401 — runs dashboard via main() at module end
