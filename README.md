# CME Challenge Trading Dashboard

Unified Streamlit dashboard for Brent + Gold futures trading during geopolitical risk regimes.

## Modules

- **`market_dashboard.py`** — broad universe monitoring (indices + commodities + macro)
- **`geo_dashboard.py`** — Brent/Gold-focused dashboard with geopolitical indicators
- **`markov_regime.py`** — Markov regime model with conditional return analysis
- **`app.py`** — Streamlit web interface combining all three

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard will open at `http://localhost:8501`.

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (see commands below)
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click "New app" → select your repo → set main file to `app.py`
5. Deploy

Auto-deploys on every push to main.

## Git setup (first time)

```bash
git init
git add .
git commit -m "Initial dashboard commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Updates

```bash
git add .
git commit -m "Your message"
git push
```

## Data refresh

Streamlit caches data for 1 hour. Click **Refresh Data** in the sidebar
to force refresh (useful when geopolitical news breaks).

## Structure

```
cme_models/
├── app.py                  # Streamlit app (entry point)
├── market_dashboard.py     # Full universe monitoring
├── geo_dashboard.py        # Brent/Gold focused
├── markov_regime.py        # Regime classification + forecast
├── requirements.txt        # Python deps
├── README.md              # This file
└── .gitignore             # Git exclusions
```

## Tabs

1. **Summary** — one-glance view with VIX/DXY/yields, current regime state,
   5-day forecast, Brent/Gold snapshots, active alerts
2. **Market Overview** — full universe (indices, commodities, cross-asset)
3. **Brent + Gold** — price charts, spreads, correlations, context
4. **Regime Model** — transition matrix heatmap, forecast, conditional returns