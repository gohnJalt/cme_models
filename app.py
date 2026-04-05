"""
CME Challenge — Unified Trading Dashboard
==========================================
Streamlit web app that combines market monitoring, geopolitical risk tracking,
and Markov regime analysis into a single interface.

Modules integrated:
    - market_dashboard.MarketDashboard   (broad universe scan)
    - geo_dashboard.GeoTradeDashboard    (Brent + Gold deep dive)
    - markov_regime.RegimeModel          (regime forecasting)

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub, connect repo at https://share.streamlit.io
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Your existing modules (no changes needed)
from market_dashboard import MarketDashboard, DashboardConfig
from geo_dashboard import GeoTradeDashboard, GeoTradeConfig
from markov_regime import RegimeModel, RegimeConfig, fit_from_dashboard_data
from strategy_sweep import (
    StrategySweep, SweepConfig, sweep_from_regime_model, format_sweep_table
)


# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="CME Challenge Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Cached data loaders
# ============================================================================
# Cache for 1 hour — data updates once per hour max, avoids yfinance rate limits
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_market_dashboard():
    d = MarketDashboard()
    d.refresh()
    return d

@st.cache_data(ttl=3600, show_spinner="Fetching Brent/Gold data...")
def load_geo_dashboard():
    d = GeoTradeDashboard()
    d.refresh()
    return d

@st.cache_data(ttl=3600, show_spinner="Fitting regime model...")
def load_regime_model():
    """Fetches its own 3y history — the regime model needs long lookback
    for the rolling percentile classification to warm up."""
    import yfinance as yf
    raw = yf.download(
        ['^VIX', 'BZ=F', 'GC=F'],
        period='3y', interval='1d',
        progress=False, auto_adjust=True,
    )
    vix = raw['Close']['^VIX'].dropna()
    brent = raw['Close']['BZ=F'].dropna()
    gold = raw['Close']['GC=F'].dropna()
    return RegimeModel().fit(
        driver_series=vix,
        target_series={'BZ=F': brent, 'GC=F': gold},
    )


@st.cache_data(ttl=3600, show_spinner="Fetching E-Micro contracts...")
def load_micro_contracts():
    """
    E-Micro futures: smaller notional versions of standard E-mini contracts.
    Note: yfinance has patchy coverage on micros — we fetch what's available.
    """
    import yfinance as yf
    micro_tickers = {
        'MES=F': 'Micro E-mini S&P 500',
        'MNQ=F': 'Micro E-mini Nasdaq-100',
        'M2K=F': 'Micro E-mini Russell 2000',
        'MYM=F': 'Micro E-mini Dow',
        'MGC=F': 'Micro Gold',
    }
    raw = yf.download(
        list(micro_tickers.keys()),
        period='1y', interval='1d',
        progress=False, auto_adjust=True,
    )
    data = {}
    for t, name in micro_tickers.items():
        try:
            df = pd.DataFrame({
                'open':  raw['Open'][t],
                'high':  raw['High'][t],
                'low':   raw['Low'][t],
                'close': raw['Close'][t],
                'volume': raw['Volume'][t],
            }).dropna(subset=['close'])
            if len(df) > 30:
                data[t] = {'df': df, 'name': name}
        except KeyError:
            pass
    return data


@st.cache_data(ttl=3600, show_spinner="Running strategy sweep...")
def run_strategy_sweep():
    """Run strategy sweep on Brent and Gold using regime model."""
    import yfinance as yf
    raw = yf.download(
        ['^VIX', 'BZ=F', 'GC=F'],
        period='3y', interval='1d',
        progress=False, auto_adjust=True,
    )
    vix = raw['Close']['^VIX'].dropna()
    brent = raw['Close']['BZ=F'].dropna()
    gold = raw['Close']['GC=F'].dropna()

    model = RegimeModel().fit(
        driver_series=vix,
        target_series={'BZ=F': brent, 'GC=F': gold},
    )
    results = sweep_from_regime_model(model, {'BZ=F': brent, 'GC=F': gold})
    return model.current_state, results


# ============================================================================
# Plotting helpers
# ============================================================================
def plot_price_with_ma(data, ticker, title=None, sma_windows=(20, 50), lookback_days=180):
    """Price chart with customizable SMAs.
    
    Parameters
    ----------
    data : dict of ticker -> OHLCV DataFrame
    ticker : which ticker to plot
    title : chart title (defaults to ticker)
    sma_windows : tuple of SMA windows to overlay (e.g. (20, 50, 100))
    lookback_days : how many recent days to show
    """
    if ticker not in data:
        return None
    df = data[ticker].tail(lookback_days).copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name=ticker,
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    ))

    # Color palette for SMAs
    sma_colors = ['#ffa726', '#ab47bc', '#29b6f6', '#66bb6a', '#ef5350']
    for i, window in enumerate(sma_windows):
        # Compute SMA on full series then slice (for edge accuracy)
        sma = data[ticker]['close'].rolling(window).mean().tail(lookback_days)
        fig.add_trace(go.Scatter(
            x=sma.index, y=sma.values, name=f'SMA{window}',
            line=dict(color=sma_colors[i % len(sma_colors)], width=1.5),
        ))

    fig.update_layout(
        title=title or ticker,
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
    )
    return fig


def plot_vix_regime(vix_series, regime_series, config_labels):
    """VIX time series colored by regime."""
    fig = go.Figure()

    # Color mapping
    colors = {'calm': '#66bb6a', 'normal': '#ffa726', 'stressed': '#ef5350'}

    # Plot VIX
    fig.add_trace(go.Scatter(
        x=vix_series.index, y=vix_series.values,
        mode='lines', name='VIX',
        line=dict(color='white', width=1),
    ))

    # Add colored bands per regime
    aligned = pd.DataFrame({'vix': vix_series, 'regime': regime_series}).dropna()
    for label in config_labels:
        subset = aligned[aligned['regime'] == label]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset.index, y=subset['vix'],
                mode='markers', name=label,
                marker=dict(color=colors.get(label, 'gray'), size=4),
            ))

    fig.update_layout(
        title="VIX with Regime Classification",
        yaxis_title="VIX",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
    )
    return fig


def plot_transition_heatmap(transition_matrix):
    """Heatmap of transition matrix."""
    tm = transition_matrix
    fig = go.Figure(data=go.Heatmap(
        z=tm.values, x=tm.columns, y=tm.index,
        text=[[f"{v:.1%}" for v in row] for row in tm.values],
        texttemplate="%{text}", textfont={"size": 14},
        colorscale='Blues', zmin=0, zmax=1,
        showscale=False,
    ))
    fig.update_layout(
        title="Transition Matrix (P(next state | current state))",
        xaxis_title="To State", yaxis_title="From State",
        height=300, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def plot_forecast(forecast_df, current_state):
    """Stacked area chart of regime probabilities over forecast horizon."""
    colors = {'calm': '#66bb6a', 'normal': '#ffa726', 'stressed': '#ef5350'}
    fig = go.Figure()
    for col in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df[col] * 100,
            name=col, stackgroup='one', mode='lines',
            line=dict(width=0.5, color=colors.get(col, 'gray')),
            fillcolor=colors.get(col, 'gray'),
        ))
    fig.update_layout(
        title=f"Regime Probability Forecast (from {current_state.upper()})",
        xaxis_title="Days Ahead", yaxis_title="Probability (%)",
        height=350, margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified', yaxis=dict(range=[0, 100]),
    )
    return fig


def plot_conditional_returns(conditional_stats, asset):
    """Bar chart of 5-day forward returns by regime for an asset."""
    if asset not in conditional_stats:
        return None
    stats = conditional_stats[asset]
    regimes = list(stats.keys())
    means = [stats[r]['mean_ret_5d_fwd'] * 100 for r in regimes]
    win_rates = [stats[r]['win_rate_5d_fwd'] * 100 for r in regimes]

    colors = {'calm': '#66bb6a', 'normal': '#ffa726', 'stressed': '#ef5350'}
    bar_colors = [colors.get(r, 'gray') for r in regimes]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=regimes, y=means, name='Mean 5d fwd return (%)',
        marker_color=bar_colors,
        text=[f"{m:+.2f}%" for m in means], textposition='auto',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=regimes, y=win_rates, name='Win rate (%)',
        mode='markers+lines', marker=dict(size=12, color='cyan'),
        line=dict(color='cyan', width=2, dash='dot'),
    ), secondary_y=True)

    fig.update_layout(
        title=f"{asset}: 5-Day Forward Return by Regime",
        height=350, margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x',
    )
    fig.update_yaxes(title_text="Mean Return (%)", secondary_y=False)
    fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True, range=[0, 100])
    return fig


def plot_spread_history(data, t1='BZ=F', t2='CL=F', title="Brent-WTI Spread"):
    """Time series of the spread between two contracts."""
    if t1 not in data or t2 not in data:
        return None
    df = pd.DataFrame({
        t1: data[t1]['close'],
        t2: data[t2]['close'],
    }).dropna()
    spread = df[t1] - df[t2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread.values, name=f"{t1} - {t2}",
        line=dict(color='cyan', width=2), fill='tozeroy',
        fillcolor='rgba(0, 188, 212, 0.1)',
    ))
    fig.add_hline(y=spread.mean(), line_dash="dash",
                  line_color="gray", annotation_text="1y mean")
    fig.update_layout(
        title=title, yaxis_title="Spread ($)",
        height=300, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def plot_correlation_history(data, t1='BZ=F', t2='GC=F', window=20):
    """Rolling correlation between two assets."""
    if t1 not in data or t2 not in data:
        return None
    r1 = data[t1]['close'].pct_change()
    r2 = data[t2]['close'].pct_change()
    corr = r1.rolling(window).corr(r2).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corr.index, y=corr.values, name=f"{window}d corr",
        line=dict(color='orange', width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.5, line_dash="dot", line_color="red",
                  annotation_text="high corr threshold")
    fig.update_layout(
        title=f"{t1}-{t2} {window}d Rolling Correlation",
        yaxis_title="Correlation", yaxis=dict(range=[-1, 1]),
        height=300, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.title("CME Challenge")
    st.caption("Brent + Gold Trading Dashboard")

    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Data refreshes hourly. Click above to force refresh.")

    # Status panel
    try:
        market_d = load_market_dashboard()
        st.success(f"Data: {market_d.loader.as_of.date()}")
        st.caption(f"Loaded {len(market_d.loader.data)} tickers")
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

    st.divider()
    st.caption("[ Modules ]")
    st.caption("- Market dashboard (full universe)")
    st.caption("- Geo dashboard (Brent/Gold)")
    st.caption("- Markov regime model")


# ============================================================================
# Main content
# ============================================================================
st.title("CME Challenge Trading Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Tabs
tab_summary, tab_market, tab_geo, tab_regime, tab_sweep, tab_micros = st.tabs([
    "Summary", "Market Overview", "Brent + Gold", "Regime Model",
    "Strategy Sweep", "E-Micros"
])


# ============================================================================
# Tab 1: Summary — the one-glance view
# ============================================================================
with tab_summary:
    st.header("At-a-Glance Summary")

    # Load everything we need
    geo_d = load_geo_dashboard()
    regime_model = load_regime_model()

    # Top row: key regime metrics
    col1, col2, col3, col4 = st.columns(4)
    r = geo_d.regime_stats
    with col1:
        st.metric(
            "VIX",
            f"{r.get('vix', 0):.2f}",
            f"{r.get('vix_change_1d', 0)*100:+.1f}% today",
            delta_color="inverse",  # VIX up = bad
        )
    with col2:
        st.metric(
            "VIX Percentile (1y)",
            f"{r.get('vix_pct_1y', 0)*100:.0f}%",
        )
    with col3:
        st.metric(
            "DXY",
            f"{r.get('dxy', 0):.2f}",
            f"{r.get('dxy_change_5d', 0)*100:+.2f}% (5d)",
            delta_color="inverse",  # DXY up = commodity headwind
        )
    with col4:
        st.metric(
            "10Y Yield",
            f"{r.get('yield_10y', 0):.2f}%",
            f"{r.get('yield_chg_5d_bps', 0):+.0f} bps (5d)",
        )

    st.divider()

    # Regime state + forecast
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Current Regime")
        state_color = {'calm': 'green', 'normal': 'orange', 'stressed': 'red'}.get(
            regime_model.current_state, 'gray'
        )
        st.markdown(
            f"### <span style='color:{state_color}'>{regime_model.current_state.upper()}</span>",
            unsafe_allow_html=True
        )
        st.caption(
            f"VIX={regime_model.current_driver_value:.2f}, "
            f"{regime_model.current_driver_percentile*100:.0f}th percentile"
        )

        # Stationary distribution
        stat = regime_model.stationary_distribution()
        st.caption("Long-run frequencies:")
        for s, p in stat.items():
            st.caption(f"  {s}: {p*100:.1f}%")

    with col_right:
        st.subheader("5-Day Forecast")
        forecast = regime_model.forecast(horizon=5)
        st.plotly_chart(
            plot_forecast(forecast, regime_model.current_state),
            width='stretch',
            key='forecast_summary',
        )

    st.divider()

    # Brent + Gold positioning
    st.subheader("Trading Contract Snapshot")
    col_brent, col_gold = st.columns(2)

    with col_brent:
        st.markdown("### Brent (BZ=F)")
        b = geo_d.brent_stats
        if b:
            sub1, sub2, sub3 = st.columns(3)
            sub1.metric("Close", f"${b.get('close', 0):.2f}",
                        f"{b.get('ret_1d', 0)*100:+.2f}%")
            sub2.metric("5d", f"{b.get('ret_5d', 0)*100:+.2f}%",
                        f"20d: {b.get('ret_20d', 0)*100:+.2f}%")
            sub3.metric("Trend", b.get('trend', '?'),
                        f"vs 20MA: {b.get('vs_ma20_pct', 0)*100:+.1f}%")

            # Expected path
            try:
                brent_path = regime_model.expected_path_return('BZ=F', horizon=5)
                st.info(
                    f"**Regime-weighted 5d expected return:** "
                    f"{brent_path['cumulative_expected_return']*100:+.2f}% "
                    f"(path vol: {brent_path['path_vol']*100:.2f}%)"
                )
            except Exception as e:
                pass

    with col_gold:
        st.markdown("### Gold (GC=F)")
        g = geo_d.gold_stats
        if g:
            sub1, sub2, sub3 = st.columns(3)
            sub1.metric("Close", f"${g.get('close', 0):.2f}",
                        f"{g.get('ret_1d', 0)*100:+.2f}%")
            sub2.metric("5d", f"{g.get('ret_5d', 0)*100:+.2f}%",
                        f"20d: {g.get('ret_20d', 0)*100:+.2f}%")
            sub3.metric("Trend", g.get('trend', '?'),
                        f"vs 20MA: {g.get('vs_ma20_pct', 0)*100:+.1f}%")

            try:
                gold_path = regime_model.expected_path_return('GC=F', horizon=5)
                st.info(
                    f"**Regime-weighted 5d expected return:** "
                    f"{gold_path['cumulative_expected_return']*100:+.2f}% "
                    f"(path vol: {gold_path['path_vol']*100:.2f}%)"
                )
            except Exception:
                pass

    st.divider()

    # Alerts
    st.subheader("Active Alerts")
    if geo_d.alert_list:
        for priority, msg in geo_d.alert_list:
            icon = {1: "🚨", 2: "⚠️", 3: "📌", 4: "ℹ️"}.get(priority, "•")
            if priority == 1:
                st.error(f"{icon} {msg}")
            elif priority == 2:
                st.warning(f"{icon} {msg}")
            else:
                st.info(f"{icon} {msg}")
    else:
        st.success("No active alerts")


# ============================================================================
# Tab 2: Market Overview
# ============================================================================
with tab_market:
    st.header("Broad Market Overview")
    market_d = load_market_dashboard()

    # Macro regime
    st.subheader("Macro Regime")
    r = market_d.regime_dict
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VIX", f"{r.get('vix_level', 0):.2f}",
                f"{r.get('vix_change_1d', 0)*100:+.1f}%")
    col2.metric("VIX Term",
                f"{r.get('vix_term_structure', 0):.2f}",
                r.get('vix_term_regime', 'n/a').upper())
    col3.metric("DXY", f"{r.get('dxy_level', 0):.2f}",
                f"{r.get('dxy_change_5d', 0)*100:+.2f}% (5d)")
    col4.metric("10Y Yield", f"{r.get('yield_10y', 0):.2f}%",
                f"{r.get('yield_change_5d_bps', 0):+.0f} bps (5d)")

    # Alerts
    if market_d.anomaly_flags:
        st.subheader("Anomalies")
        for flag in market_d.anomaly_flags:
            st.warning(flag)

    # Indices
    st.subheader("Equity Index Futures")
    if market_d.indices_momentum is not None and not market_d.indices_momentum.empty:
        df = market_d.indices_momentum.copy()
        # Format for display
        display_cols = ['ticker', 'close', 'ret_1d', 'ret_5d', 'ret_20d',
                        'vs_ma20_pct', 'vs_ma50_pct', 'trend_state', 'vol_percentile']
        display = df[display_cols].copy()
        for col in ['ret_1d', 'ret_5d', 'ret_20d', 'vs_ma20_pct', 'vs_ma50_pct', 'vol_percentile']:
            display[col] = (display[col] * 100).round(2)
        st.dataframe(display, width='stretch', hide_index=True)

    # Commodities
    st.subheader("Commodity Futures")
    if market_d.commodities_momentum is not None and not market_d.commodities_momentum.empty:
        df = market_d.commodities_momentum.copy()
        display_cols = ['ticker', 'close', 'ret_1d', 'ret_5d', 'ret_20d',
                        'vs_ma20_pct', 'vs_ma50_pct', 'trend_state', 'vol_percentile']
        display = df[display_cols].copy()
        for col in ['ret_1d', 'ret_5d', 'ret_20d', 'vs_ma20_pct', 'vs_ma50_pct', 'vol_percentile']:
            display[col] = (display[col] * 100).round(2)
        st.dataframe(display, width='stretch', hide_index=True)

    # Cross-asset
    st.subheader("Cross-Asset Signals")
    ca = market_d.cross_asset_dict
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Risk appetite:** {ca.get('risk_appetite', 'n/a')}")
        st.markdown(f"**NQ-ES 5d:** {ca.get('NQ_vs_ES_5d', 0)*100:+.2f}% "
                    f"| **20d:** {ca.get('NQ_vs_ES_20d', 0)*100:+.2f}%")
        st.markdown(f"**RTY-ES 5d:** {ca.get('RTY_vs_ES_5d', 0)*100:+.2f}% "
                    f"| **20d:** {ca.get('RTY_vs_ES_20d', 0)*100:+.2f}%")
    with col2:
        st.markdown(f"**Brent-WTI spread:** ${ca.get('brent_wti_spread', 0):.2f} "
                    f"({ca.get('brent_wti_spread_pct_1y', 0)*100:.0f}% 1y pct)")
        st.markdown(f"**Gold/Copper ratio:** {ca.get('gold_copper_ratio', 0):.1f} "
                    f"({ca.get('gold_copper_pct_1y', 0)*100:.0f}% 1y pct)")
        st.markdown(f"**NQ/YM ratio:** {ca.get('growth_value_ratio', 0):.4f} "
                    f"({ca.get('growth_value_pct_1y', 0)*100:.0f}% 1y pct)")


# ============================================================================
# Tab 3: Brent + Gold
# ============================================================================
with tab_geo:
    st.header("Brent + Gold Deep Dive")
    geo_d = load_geo_dashboard()

    # Alerts at top
    if geo_d.alert_list:
        st.subheader("Alerts")
        for priority, msg in geo_d.alert_list:
            if priority == 1:
                st.error(msg)
            elif priority == 2:
                st.warning(msg)
            else:
                st.info(msg)

    st.divider()

    # Price charts side by side
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_price_with_ma(
            geo_d.loader.data, 'BZ=F',
            title="Brent (BZ=F) - 6 months",
            sma_windows=(10, 20, 50, 100),
        )
        if fig:
            st.plotly_chart(fig, width='stretch', key='geo_price_brent')
    with col2:
        fig = plot_price_with_ma(
            geo_d.loader.data, 'GC=F',
            title="Gold (GC=F) - 6 months",
            sma_windows=(10, 20, 50, 100),
        )
        if fig:
            st.plotly_chart(fig, width='stretch', key='geo_price_gold')

    # Spread and correlation
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_spread_history(geo_d.loader.data, 'BZ=F', 'CL=F',
                                   title="Brent-WTI Spread (geopolitical premium)")
        if fig:
            st.plotly_chart(fig, width='stretch', key='geo_spread_brent_wti')
    with col2:
        fig = plot_correlation_history(geo_d.loader.data, 'BZ=F', 'GC=F', 20)
        if fig:
            st.plotly_chart(fig, width='stretch', key='geo_corr_brent_gold')

    # Brent context detail
    st.subheader("Brent Context")
    ctx = geo_d.brent_context
    col1, col2, col3 = st.columns(3)
    col1.metric("Brent-WTI Spread", f"${ctx.get('brent_wti_spread', 0):+.2f}",
                f"{ctx.get('brent_wti_spread_5d_chg', 0):+.2f} (5d)")
    col2.metric("1y Percentile", f"{ctx.get('brent_wti_spread_pct_1y', 0)*100:.0f}%")
    col3.metric("Brent-Gold Corr (20d)",
                f"{ctx.get('brent_gold_correlation_20d', 0):+.2f}")

    if 'geopolitical_premium' in ctx:
        st.info(f"**Geopolitical premium status:** {ctx['geopolitical_premium']}")

    st.divider()

    # Gold context detail
    st.subheader("Gold Context")
    gctx = geo_d.gold_context
    col1, col2, col3 = st.columns(3)
    col1.metric("Gold/Silver Ratio", f"{gctx.get('gold_silver_ratio', 0):.1f}",
                f"{gctx.get('gold_silver_5d_chg', 0)*100:+.2f}% (5d)")
    col2.metric("1y Percentile", f"{gctx.get('gold_silver_pct_1y', 0)*100:.0f}%")
    col3.metric("GDX vs Gold (5d)",
                f"{gctx.get('gdx_gold_5d_diff', 0)*100:+.2f}%")

    if 'gold_silver_signal' in gctx:
        st.info(f"**Gold/Silver signal:** {gctx['gold_silver_signal']}")
    if 'miners_signal' in gctx:
        st.info(f"**Miners signal:** {gctx['miners_signal']}")


# ============================================================================
# Tab 4: Regime Model
# ============================================================================
with tab_regime:
    st.header("Markov Regime Model")
    geo_d = load_geo_dashboard()
    regime_model = load_regime_model()

    # Current state
    st.subheader("Current State")
    col1, col2, col3 = st.columns(3)
    col1.metric("State", regime_model.current_state.upper())
    col2.metric("VIX Level", f"{regime_model.current_driver_value:.2f}")
    col3.metric("1y Percentile", f"{regime_model.current_driver_percentile*100:.1f}%")

    st.divider()

    # Transition matrix and forecast
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_transition_heatmap(regime_model.transition_matrix),
            width='stretch',
            key='regime_transition_matrix',
        )
        # Persistence metrics
        st.caption("**Regime persistence (diagonal):**")
        tm = regime_model.transition_matrix
        for state in tm.columns:
            st.caption(f"  {state}: {tm.loc[state, state]*100:.1f}% chance of staying next day")

    with col2:
        forecast = regime_model.forecast(horizon=5)
        st.plotly_chart(
            plot_forecast(forecast, regime_model.current_state),
            width='stretch',
            key='forecast_regime_tab',
        )
        st.caption("**Forecast table:**")
        st.dataframe(
            (forecast * 100).round(1).astype(str) + '%',
            width='stretch',
        )

    st.divider()

    # Conditional returns
    st.subheader("Historical Returns by Regime")
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_conditional_returns(regime_model.conditional_stats, 'BZ=F')
        if fig:
            st.plotly_chart(fig, width='stretch', key='cond_returns_brent')
    with col2:
        fig = plot_conditional_returns(regime_model.conditional_stats, 'GC=F')
        if fig:
            st.plotly_chart(fig, width='stretch', key='cond_returns_gold')

    # Full conditional stats tables
    st.subheader("Conditional Stats Detail")
    for asset, stats in regime_model.conditional_stats.items():
        st.markdown(f"**{asset}**")
        rows = []
        for state, s in stats.items():
            rows.append({
                'regime': state,
                'n_obs': s['n_obs'],
                'mean_1d': f"{s['mean_ret_1d']*100:+.2f}%",
                'vol_1d': f"{s['vol_1d']*100:.2f}%",
                'mean_5d_fwd': f"{s['mean_ret_5d_fwd']*100:+.2f}%",
                'win_rate_5d': f"{s['win_rate_5d_fwd']*100:.1f}%",
                'worst_5d': f"{s['worst_5d_fwd']*100:+.1f}%",
                'best_5d': f"{s['best_5d_fwd']*100:+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # Expected path returns
    st.divider()
    st.subheader("Expected Path Returns (from current state)")
    for asset in regime_model.conditional_stats:
        path = regime_model.expected_path_return(asset, horizon=5)
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{asset} 5d Expected Return",
                    f"{path['cumulative_expected_return']*100:+.2f}%")
        col2.metric("Path Vol", f"{path['path_vol']*100:.2f}%")
        col3.metric("Expected Sharpe (annz)",
                    f"{path['expected_sharpe_annualized']:.2f}")


# ============================================================================
# Tab 5: Strategy Sweep
# ============================================================================
with tab_sweep:
    st.header("Regime-Conditional Strategy Sweep")
    st.caption(
        "Tests trend-following and mean-reversion strategies across parameter grids, "
        "filtered to historical dates where the VIX regime matched the **current** regime. "
        "Top/bottom 5 results shown side-by-side for honest signal assessment."
    )

    try:
        current_regime_sweep, sweep_results = run_strategy_sweep()
    except Exception as e:
        st.error(f"Sweep failed: {e}")
        st.stop()

    st.info(
        f"**Current regime: {current_regime_sweep.upper()}**  — "
        f"showing results conditional on this regime only. "
        f"Each (n, m) pair requires at least 15 historical observations to be reported."
    )

    st.caption(
        "**Strategy definitions:** "
        "**Trend:** long when close > SMA(n) AND SMA(n) > SMA(m). "
        "**Mean-rev:** long when close is at least `threshold%` below SMA(lookback). "
        "Holding period: 5 days."
    )

    sweep_cols_trend = ['n', 'm', 'n_obs', 'mean', 'median', 'std',
                        'win_rate', 'worst', 'best', 'sharpe_approx']
    sweep_cols_mr = ['lookback', 'threshold_pct', 'n_obs', 'mean', 'median', 'std',
                     'win_rate', 'worst', 'best', 'sharpe_approx']

    for asset in sweep_results:
        st.divider()
        st.subheader(f"{asset}")
        res = sweep_results[asset]

        # Trend-following
        st.markdown("**Trend-Following:**")
        if res['trend_top'] is not None and len(res['trend_top']) > 0:
            col_top, col_bot = st.columns(2)
            with col_top:
                st.caption("TOP 5 (best mean return)")
                st.dataframe(
                    format_sweep_table(res['trend_top'], sweep_cols_trend),
                    width='stretch', hide_index=True,
                )
            with col_bot:
                st.caption("BOTTOM 5 (worst mean return)")
                st.dataframe(
                    format_sweep_table(res['trend_bottom'], sweep_cols_trend),
                    width='stretch', hide_index=True,
                )
        else:
            st.caption("No trend-following combos met minimum observation threshold")

        # Mean-reversion
        st.markdown("**Mean-Reversion:**")
        if res['mr_top'] is not None and len(res['mr_top']) > 0:
            col_top, col_bot = st.columns(2)
            with col_top:
                st.caption("TOP 5 (best mean return)")
                st.dataframe(
                    format_sweep_table(res['mr_top'], sweep_cols_mr),
                    width='stretch', hide_index=True,
                )
            with col_bot:
                st.caption("BOTTOM 5 (worst mean return)")
                st.dataframe(
                    format_sweep_table(res['mr_bottom'], sweep_cols_mr),
                    width='stretch', hide_index=True,
                )
        else:
            st.caption("No mean-reversion combos met minimum observation threshold")

    with st.expander("How to read these results"):
        st.markdown("""
        **Columns (all returns are 5-day forward, shown as %):**
        - **n_obs**: number of historical signal triggers in the current regime
        - **mean / median**: central tendency of 5-day returns after signals
        - **std**: dispersion — higher means noisier signal
        - **win_rate**: % of signals that produced positive returns
        - **worst / best**: tail outcomes
        - **sharpe_approx**: mean / std (not annualized, rough edge quality)

        **What to look for:**
        - High mean + high win_rate + low std = consistent edge
        - High mean + low win_rate = a few lucky outliers, probably noise
        - Top 5 and bottom 5 similar in magnitude = strategy has no regime edge
        - Bottom 5 strongly negative = possible contrarian signal

        **Caveats:**
        - Results are biased toward parameters that *happened* to work in-sample
        - Current regime may have thin history — check n_obs carefully
        - Doesn't account for transaction costs or slippage
        - Use as input to your judgment, not as an auto-trade rule
        """)


# ============================================================================
# Tab 6: E-Micros
# ============================================================================
with tab_micros:
    st.header("E-Micro Contracts")
    st.caption(
        "Smaller notional equivalents of standard E-mini contracts. "
        "Useful for contest sizing when full E-mini margin is too large."
    )

    try:
        micro_data = load_micro_contracts()
    except Exception as e:
        st.error(f"Failed to load micros: {e}")
        micro_data = {}

    if not micro_data:
        st.warning(
            "No E-Micro data loaded. yfinance has patchy coverage for micros — "
            "they may be unavailable or temporarily rate-limited. "
            "Try clicking Refresh Data in the sidebar."
        )
    else:
        st.success(f"Loaded {len(micro_data)} micro contracts")

        # Build a momentum table
        rows = []
        for ticker, info in micro_data.items():
            df = info['df']
            close = df['close']
            if len(close) < 50:
                continue
            latest = close.iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ret_1d = close.pct_change().iloc[-1]
            ret_5d = close.pct_change(5).iloc[-1]
            ret_20d = close.pct_change(20).iloc[-1]
            rvol = close.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)
            rows.append({
                'ticker': ticker,
                'name': info['name'],
                'close': round(latest, 2),
                'ret_1d': round(ret_1d * 100, 2),
                'ret_5d': round(ret_5d * 100, 2),
                'ret_20d': round(ret_20d * 100, 2),
                'vs_ma20_pct': round((latest - ma20) / ma20 * 100, 2),
                'vs_ma50_pct': round((latest - ma50) / ma50 * 100, 2),
                'ann_vol_pct': round(rvol * 100, 1),
            })
        if rows:
            st.dataframe(
                pd.DataFrame(rows), width='stretch', hide_index=True,
            )

        # Price charts - 2 per row
        st.divider()
        st.subheader("Charts (candlestick + SMAs)")
        tickers = list(micro_data.keys())
        # Transform to format compatible with plot_price_with_ma
        plot_data = {t: info['df'] for t, info in micro_data.items()}

        for i in range(0, len(tickers), 2):
            col_a, col_b = st.columns(2)
            if i < len(tickers):
                t = tickers[i]
                with col_a:
                    fig = plot_price_with_ma(
                        plot_data, t,
                        title=f"{t} — {micro_data[t]['name']}",
                        sma_windows=(10, 20, 50),
                    )
                    if fig:
                        st.plotly_chart(fig, width='stretch', key=f'micro_chart_{t}')
            if i + 1 < len(tickers):
                t = tickers[i + 1]
                with col_b:
                    fig = plot_price_with_ma(
                        plot_data, t,
                        title=f"{t} — {micro_data[t]['name']}",
                        sma_windows=(10, 20, 50),
                    )
                    if fig:
                        st.plotly_chart(fig, width='stretch', key=f'micro_chart_{t}')


# Footer
st.divider()
st.caption(
    "Built with Streamlit. Data from yfinance. "
    "For educational/contest use only — not investment advice."
)