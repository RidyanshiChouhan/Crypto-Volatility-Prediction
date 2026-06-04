"""
Streamlit Community Cloud entry point (root).
Must call main() on every rerun — do not only import the module.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.streamlit_app import main

main()
