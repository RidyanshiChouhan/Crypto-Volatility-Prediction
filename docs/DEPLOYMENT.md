# Deployment Guide

## Prerequisites

- Python 3.12+
- GitHub account
- Trained model (`python scripts/train_pipeline.py`)

---

## 1. Streamlit Community Cloud

1. Push repository to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. **New app** → Select repo → Main file: `app/streamlit_app.py`
4. Python version: **3.12**
5. Add `requirements.txt` at repo root
6. Deploy

**Secrets (optional):** None required for CoinGecko public API.

**Note:** First deploy runs without model until you commit `models/best_model.pkl` or run training in CI and artifact upload.

---

## 2. Render

1. Create **Web Service** from GitHub repo
2. Environment: **Docker** or Native Python
3. **Docker:** Use included `Dockerfile`
4. **Native:**
   - Build: `pip install -r requirements.txt && python scripts/train_pipeline.py`
   - Start: `streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
5. Set health check path: `/_stcore/health`

---

## 3. Railway

1. [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Use `Dockerfile` or Nixpacks with start command:
   ```
   streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```
3. Expose port **8501** (or `$PORT`)

---

## 4. Docker (Local / VPS)

```bash
docker compose up --build
```

Open: http://localhost:8501

---

## 5. Model Artifacts for Cloud

Option A: Commit `models/best_model.pkl` (if < 100MB)

Option B: Train on startup (Dockerfile includes training step)

Option C: GitHub Actions artifact → download in deploy hook

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PYTHONPATH` | Set to project root (`.`) |

No API key required for CoinGecko free tier (rate limits apply).
