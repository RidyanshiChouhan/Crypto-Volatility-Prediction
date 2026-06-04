# Push to GitHub & Deploy on Streamlit Cloud

Your profile: [github.com/RidyanshiChouhan](https://github.com/RidyanshiChouhan)

## Step 1 — Create the repository on GitHub

1. Open [Create a new repository](https://github.com/new)
2. **Repository name:** `Crypto-Volatility-Prediction`
3. **Public**
4. Do **not** add README, .gitignore, or license (this project already has them)
5. Click **Create repository**

## Step 2 — Push from your PC

In PowerShell (no need to `activate` the venv):

```powershell
cd "c:\Users\Computer 02\OneDrive - Medi-Caps Group of Institutions\Documents\New folder\Crypto-Volatility-Prediction"

git remote add origin https://github.com/RidyanshiChouhan/Crypto-Volatility-Prediction.git
git branch -M main
git push -u origin main
```

When prompted for credentials:

- **Username:** `RidyanshiChouhan`
- **Password:** use a **Personal Access Token (PAT)**, not your GitHub account password  
  Create one: GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → Generate → scope: `repo`

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. **New app**
3. Repository: `RidyanshiChouhan/Crypto-Volatility-Prediction`
4. Branch: `main`
5. Main file path: `streamlit_app.py`
6. Deploy

After the repo is connected, the **Deploy** button in the local app will work too.

## Security

Never share your GitHub password in chat or commit it to the repo. If you exposed a password, **change it immediately** on GitHub and use a PAT instead.
