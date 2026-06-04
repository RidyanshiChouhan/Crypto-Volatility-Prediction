# GitHub Portfolio Setup Guide

## 1. Create Repository

```bash
cd Crypto-Volatility-Prediction
git init
git add .
git commit -m "Initial commit: Crypto Volatility Prediction System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Crypto-Volatility-Prediction.git
git push -u origin main
```

## 2. Repository Settings

### Description
```
End-to-end ML system predicting cryptocurrency volatility with XGBoost, SHAP, and Streamlit dashboard.
```

### Topics (Tags)
```
machine-learning, cryptocurrency, volatility-prediction, streamlit, xgboost, lightgbm, mlops, data-science, python, shap, plotly, time-series, fintech, portfolio-project
```

### Website URL
Your Streamlit Cloud URL: `https://YOUR_APP.streamlit.app`

## 3. README Badges

Replace `YOUR_USERNAME` in README with your GitHub username.

## 4. Social Preview

1. Settings → General → Social preview
2. Upload banner from `docs/assets/banner.png` (create 1280×640 image)

## 5. Pin Repository

On your GitHub profile → Customize pins → Select this repo.

## 6. GitHub Actions

CI runs automatically on push. Enable Actions in repo Settings.

## 7. Demo GIF

Record Streamlit session:
```bash
streamlit run app/streamlit_app.py
```
Use ScreenToGif or LICEcap → save as `docs/assets/demo.gif`

## 8. Enable GitHub Pages (Optional)

Host documentation from `/docs` folder if desired.

## 9. Release

Create v1.0.0 release with trained model artifact attached.

## 10. Portfolio Description Files

See `docs/PORTFOLIO_COPY.md` for resume, LinkedIn, and profile text.
