# Quick Setup (Fast Path)

## 1. Install (minimal — faster than full requirements)

```powershell
cd Crypto-Volatility-Prediction
python -m venv .venv
pip install -r requirements-minimal.txt
```

**Windows — PowerShell blocks `Activate.ps1`?** Use either:

```powershell
# Option A: no activation needed (recommended)
.\.venv\Scripts\python.exe -m pip install -r requirements-minimal.txt
.\.venv\Scripts\python.exe scripts\bootstrap.py
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

```cmd
# Option B: double-click or run from CMD
run_app.bat
```

```powershell
# Option C: allow local scripts once (Current User only)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

> `pandas-ta` is optional; indicators use built-in fallbacks if missing.

## 2. Bootstrap data + model (offline, ~2–5 min)

```powershell
python scripts/bootstrap.py
```

## 3. Run dashboard

```powershell
streamlit run app/streamlit_app.py
```

## 4. Tests

```powershell
pytest tests/ -q --cov=src --cov=app --cov-fail-under=80
```

## Full pipeline (API + tuning + SHAP)

```powershell
pip install -r requirements.txt
python scripts/train_pipeline.py
```
