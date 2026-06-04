@echo off
cd /d "%~dp0"
echo Starting Crypto Volatility Prediction dashboard...
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
pause
