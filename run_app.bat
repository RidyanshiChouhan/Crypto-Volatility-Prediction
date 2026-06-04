@echo off
cd /d "%~dp0"
echo Starting Crypto Volatility Prediction dashboard...
".venv\Scripts\python.exe" -m streamlit run app/streamlit_app.py
pause
