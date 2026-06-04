"""
Entry point for Streamlit Community Cloud.
Main app: app/streamlit_app.py
"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / "app" / "streamlit_app.py"), run_name="__main__")
