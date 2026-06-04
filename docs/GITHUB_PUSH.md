# Push to GitHub (RidyanshiChouhan)

## Security first

- **Never share your GitHub password in chat, email, or code.**
- GitHub no longer accepts account passwords for `git push`. Use a **Personal Access Token (PAT)** or **SSH key**.
- If you shared a password anywhere, **change it now**: GitHub → Settings → Password.

---

## Step 1 — Create the repository on GitHub

1. Open [github.com/new](https://github.com/new)
2. Repository name: `Crypto-Volatility-Prediction`
3. Description: `End-to-end ML system for cryptocurrency volatility prediction with Streamlit dashboard`
4. Choose **Public**
5. Do **not** add README, .gitignore, or license (this project already has them)
6. Click **Create repository**

---

## Step 2 — Push from your PC

Open **PowerShell** in the project folder:

```powershell
cd "c:\Users\Computer 02\OneDrive - Medi-Caps Group of Institutions\Documents\New folder\Crypto-Volatility-Prediction"

git init
git branch -M main
git add .
git commit -m "Initial commit: Crypto Volatility Prediction System"

git remote add origin https://github.com/RidyanshiChouhan/Crypto-Volatility-Prediction.git
git push -u origin main
```

When prompted for credentials:

| Field | Value |
|-------|--------|
| Username | `RidyanshiChouhan` |
| Password | Paste a **Personal Access Token** (not your GitHub password) |

### Create a PAT (one-time)

1. GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. **Generate new token (classic)**
3. Scopes: check **`repo`**
4. Copy the token and use it as the password when `git push` asks

---

## Step 3 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with **GitHub**
3. **New app** → Repository: `RidyanshiChouhan/Crypto-Volatility-Prediction`
4. Branch: `main`
5. Main file path: **`streamlit_app.py`** (root) or `app/streamlit_app.py`
6. Deploy

After the first deploy, open the app URL and run locally once if the model is missing:

```powershell
.\.venv\Scripts\python.exe scripts\bootstrap.py
```

Then commit `models/best_model.pkl` only if under 100 MB (see `.gitignore` exception below).

---

## Optional — Include trained model in repo

If `models/best_model.pkl` exists and is small enough for GitHub:

1. In `.gitignore`, the line `models/*.pkl` blocks models. To ship the best model, add:

   ```
   !models/best_model.pkl
   ```

2. Then:

   ```powershell
   git add models/best_model.pkl
   git commit -m "Add trained model for Streamlit Cloud"
   git push
   ```

---

## Verify remote

```powershell
git remote -v
git status
```

Repository URL: **https://github.com/RidyanshiChouhan/Crypto-Volatility-Prediction**
