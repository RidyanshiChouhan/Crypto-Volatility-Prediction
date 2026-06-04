# Fix: "Error installing requirements" on Streamlit Cloud

## What was wrong

The old `requirements.txt` included packages that **fail or timeout** on Streamlit Cloud:

- `pandas-ta` (beta version)
- `mlflow`, `shap` (heavy; not needed to run the dashboard)
- `kaleido`, `pytest`, notebook tools

## What we fixed

1. **`requirements.txt`** — only packages needed to run the app (pinned versions)
2. **`packages.txt`** — `libgomp1` for LightGBM on Linux
3. **`src/model_io.py`** — model load without importing MLflow at startup
4. **App fallback** — synthetic data if CSV files are missing on Cloud

## What you must do now

### 1. Push the fix to GitHub

```powershell
cd "c:\Users\Computer 02\OneDrive - Medi-Caps Group of Institutions\Documents\New folder\Crypto-Volatility-Prediction"

git add requirements.txt requirements-dev.txt packages.txt src/ app/ docs/ tests/
git commit -m "Fix Streamlit Cloud requirements and cloud deployment"
git push origin main
```

### 2. Reboot the app on Streamlit Cloud

1. Open https://share.streamlit.io
2. Open your app → **Manage app** (bottom right)
3. **⋮** menu → **Reboot app** (or delete app and redeploy)

### 3. Check settings

| Setting | Value |
|---------|--------|
| Repository | `RidyanshiChouhan/Crypto-Volatility-Prediction` |
| Branch | `main` |
| Main file | `streamlit_app.py` |
| Python version | **3.11** (recommended) |

### 4. Read logs if it still fails

**Manage app** → **Logs** → look for the first red `pip` error line.

---

## Local development (full stack)

```powershell
pip install -r requirements-dev.txt
```
