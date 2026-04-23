# CME Challenge Trading Dashboard

Unified Streamlit dashboard for Brent + Gold futures trading during geopolitical risk regimes.

## Modules

- **`market_dashboard.py`** — broad universe monitoring (indices + commodities + macro)
- **`geo_dashboard.py`** — Brent/Gold-focused dashboard with geopolitical indicators
- **`markov_regime.py`** — Markov regime model with conditional return analysis
- **`app.py`** — Streamlit web interface combining all three

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
