"""
Market Indicator Dashboard
==========================
Daily data pipeline that surfaces regime, momentum, cross-asset, and anomaly
signals across equity index futures and commodity futures. Pure monitoring,
no trading decisions - you make the call.

Universe (default):
    Indices:     ES=F, NQ=F, RTY=F, YM=F
    Commodities: CL=F (WTI), BZ=F (Brent), GC=F (Gold), HG=F (Copper), NG=F (NatGas)
    Macro:       ^VIX, ^VIX3M, DX-Y.NYB (DXY), ^TNX (10Y yield)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DashboardConfig:
    # Universe definition
    indices: list = field(default_factory=lambda: ['ES=F', 'NQ=F', 'RTY=F', 'YM=F'])
    commodities: list = field(default_factory=lambda: [
        'CL=F',   # WTI Crude
        'BZ=F',   # Brent Crude
        'GC=F',   # Gold
        'HG=F',   # Copper
        'NG=F',   # Natural Gas
    ])
    macro: dict = field(default_factory=lambda: {
        'vix':     '^VIX',
        'vix3m':   '^VIX3M',
        'dxy':     'DX-Y.NYB',
        'tnx':     '^TNX',         # 10Y yield
    })

    # Lookback parameters
    history_period: str = '2y'
    percentile_window: int = 252       # 1 year for percentile calcs
    short_ma: int = 20
    long_ma: int = 50
    vol_window: int = 20
    mom_windows: tuple = (1, 5, 20)    # return lookbacks

    # Anomaly thresholds
    vol_spike_z: float = 2.0           # flag if daily move > this many std devs
    dispersion_threshold: float = 0.02 # flag if cross-index spread > 2%
    vix_change_flag: float = 0.15      # flag if VIX moves > 15% in a day


# ============================================================================
# Data Layer
# ============================================================================

class DataLoader:
    """Fetches all required series from yfinance."""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.data: dict = {}  # ticker -> DataFrame of OHLCV
        self.as_of: Optional[pd.Timestamp] = None

    def fetch(self) -> 'DataLoader':
        all_tickers = (
            self.config.indices
            + self.config.commodities
            + list(self.config.macro.values())
        )
        print(f"Fetching {len(all_tickers)} tickers ({self.config.history_period})...")

        raw = yf.download(
            all_tickers,
            period=self.config.history_period,
            interval='1d',
            progress=False,
            auto_adjust=True,
        )

        # Unpack by ticker
        for t in all_tickers:
            try:
                df = pd.DataFrame({
                    'open':  raw['Open'][t],
                    'high':  raw['High'][t],
                    'low':   raw['Low'][t],
                    'close': raw['Close'][t],
                    'volume': raw['Volume'][t],
                }).dropna(subset=['close'])
                if len(df) > 0:
                    self.data[t] = df
            except KeyError:
                print(f"  [warning] no data for {t}")

        if self.data:
            self.as_of = max(df.index[-1] for df in self.data.values())
            print(f"  Loaded {len(self.data)} tickers, latest: {self.as_of.date()}")
        return self


# ============================================================================
# Indicator Modules
# ============================================================================

class MomentumIndicators:
    """Compute momentum / trend state for each contract."""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def compute(self, data: dict, tickers: list) -> pd.DataFrame:
        cfg = self.config
        rows = []
        for t in tickers:
            if t not in data:
                continue
            df = data[t]
            close = df['close']
            if len(close) < cfg.long_ma:
                continue

            latest = close.iloc[-1]
            ma_s = close.rolling(cfg.short_ma).mean().iloc[-1]
            ma_l = close.rolling(cfg.long_ma).mean().iloc[-1]

            # Returns
            rets = {f'ret_{w}d': close.pct_change(w).iloc[-1] for w in cfg.mom_windows}

            # Realized vol and its percentile
            daily_ret = close.pct_change()
            realized_vol = daily_ret.rolling(cfg.vol_window).std() * np.sqrt(252)
            vol_now = realized_vol.iloc[-1]
            vol_pct = realized_vol.rolling(cfg.percentile_window).rank(pct=True).iloc[-1]

            # Drawdown from 20-day high
            high_20 = close.rolling(20).max().iloc[-1]
            dd_from_high = (latest - high_20) / high_20

            # Bounce from 20-day low
            low_20 = close.rolling(20).min().iloc[-1]
            dist_from_low = (latest - low_20) / low_20

            # 52-week position
            high_252 = close.rolling(252).max().iloc[-1] if len(close) >= 252 else close.max()
            low_252 = close.rolling(252).min().iloc[-1] if len(close) >= 252 else close.min()
            pct_of_range = (latest - low_252) / (high_252 - low_252) if high_252 > low_252 else 0.5

            rows.append({
                'ticker': t,
                'close': latest,
                **rets,
                'vs_ma20_pct': (latest - ma_s) / ma_s,
                'vs_ma50_pct': (latest - ma_l) / ma_l,
                'above_ma20': latest > ma_s,
                'above_ma50': latest > ma_l,
                'trend_state': self._trend_label(latest, ma_s, ma_l),
                'realized_vol': vol_now,
                'vol_percentile': vol_pct,
                'dd_from_20d_high': dd_from_high,
                'dist_from_20d_low': dist_from_low,
                'pct_52w_range': pct_of_range,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _trend_label(price, ma_s, ma_l):
        if price > ma_s and price > ma_l and ma_s > ma_l:
            return 'strong_up'
        if price > ma_s and price > ma_l:
            return 'up'
        if price < ma_s and price < ma_l and ma_s < ma_l:
            return 'strong_down'
        if price < ma_s and price < ma_l:
            return 'down'
        return 'mixed'


class RegimeIndicators:
    """Macro regime: VIX, DXY, rates, term structure."""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def compute(self, data: dict) -> dict:
        cfg = self.config
        out = {}

        # VIX level, percentile, 1d change
        vix = data.get(cfg.macro['vix'])
        if vix is not None:
            v = vix['close']
            out['vix_level'] = v.iloc[-1]
            out['vix_change_1d'] = v.pct_change().iloc[-1]
            out['vix_percentile_1y'] = v.rolling(cfg.percentile_window).rank(pct=True).iloc[-1]
            out['vix_ma20'] = v.rolling(20).mean().iloc[-1]
            out['vix_vs_ma20'] = (v.iloc[-1] - out['vix_ma20']) / out['vix_ma20']

        # VIX term structure (contango vs backwardation)
        vix3m = data.get(cfg.macro['vix3m'])
        if vix is not None and vix3m is not None:
            v_now = vix['close'].iloc[-1]
            v3m_now = vix3m['close'].iloc[-1]
            out['vix_term_structure'] = v3m_now - v_now
            out['vix_term_regime'] = 'contango' if v3m_now > v_now else 'backwardation'
            # Backwardation = market stress; contango = normal

        # DXY
        dxy = data.get(cfg.macro['dxy'])
        if dxy is not None:
            d = dxy['close']
            out['dxy_level'] = d.iloc[-1]
            out['dxy_change_5d'] = d.pct_change(5).iloc[-1]
            out['dxy_change_20d'] = d.pct_change(20).iloc[-1]
            out['dxy_vs_ma50'] = (d.iloc[-1] - d.rolling(50).mean().iloc[-1]) / d.rolling(50).mean().iloc[-1]

        # 10Y yield
        tnx = data.get(cfg.macro['tnx'])
        if tnx is not None:
            y = tnx['close']  # ^TNX is yield * 10, e.g. 42.5 = 4.25%
            out['yield_10y'] = y.iloc[-1] / 10
            out['yield_change_5d_bps'] = (y.iloc[-1] - y.iloc[-6]) * 10 if len(y) > 5 else np.nan
            out['yield_change_20d_bps'] = (y.iloc[-1] - y.iloc[-21]) * 10 if len(y) > 20 else np.nan

        return out


class CrossAssetIndicators:
    """Cross-contract spreads and ratios."""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def compute(self, data: dict) -> dict:
        out = {}
        cfg = self.config

        # Index dispersion
        def spread(t1, t2, window=5):
            if t1 in data and t2 in data:
                r1 = data[t1]['close'].pct_change(window).iloc[-1]
                r2 = data[t2]['close'].pct_change(window).iloc[-1]
                return r1 - r2
            return np.nan

        out['NQ_vs_ES_5d'] = spread('NQ=F', 'ES=F')
        out['RTY_vs_ES_5d'] = spread('RTY=F', 'ES=F')
        out['YM_vs_ES_5d'] = spread('YM=F', 'ES=F')

        out['NQ_vs_ES_20d'] = spread('NQ=F', 'ES=F', 20)
        out['RTY_vs_ES_20d'] = spread('RTY=F', 'ES=F', 20)

        # Risk-on/off interpretation
        if not np.isnan(out['RTY_vs_ES_20d']):
            if out['RTY_vs_ES_20d'] > 0.01:
                out['risk_appetite'] = 'risk_on (small caps leading)'
            elif out['RTY_vs_ES_20d'] < -0.01:
                out['risk_appetite'] = 'risk_off (small caps lagging)'
            else:
                out['risk_appetite'] = 'neutral'

        # Growth vs value proxy (NQ vs YM)
        if 'NQ=F' in data and 'YM=F' in data:
            ratio = data['NQ=F']['close'] / data['YM=F']['close']
            out['growth_value_ratio'] = ratio.iloc[-1]
            out['growth_value_pct_1y'] = ratio.rolling(cfg.percentile_window).rank(pct=True).iloc[-1]

        # Crude spread (Brent - WTI)
        if 'BZ=F' in data and 'CL=F' in data:
            out['brent_wti_spread'] = data['BZ=F']['close'].iloc[-1] - data['CL=F']['close'].iloc[-1]
            spread_series = data['BZ=F']['close'] - data['CL=F']['close']
            out['brent_wti_spread_pct_1y'] = spread_series.rolling(cfg.percentile_window).rank(pct=True).iloc[-1]

        # Gold/Copper (fear vs growth)
        if 'GC=F' in data and 'HG=F' in data:
            ratio = data['GC=F']['close'] / data['HG=F']['close']
            out['gold_copper_ratio'] = ratio.iloc[-1]
            out['gold_copper_pct_1y'] = ratio.rolling(cfg.percentile_window).rank(pct=True).iloc[-1]

        return out


class AnomalyDetector:
    """Flag unusual events worth attention."""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def detect(self, data: dict, momentum_df: pd.DataFrame, regime: dict) -> list:
        cfg = self.config
        flags = []

        # VIX spike
        if 'vix_change_1d' in regime:
            if abs(regime['vix_change_1d']) >= cfg.vix_change_flag:
                direction = 'up' if regime['vix_change_1d'] > 0 else 'down'
                flags.append(
                    f"VIX {direction} {abs(regime['vix_change_1d'])*100:.1f}% today "
                    f"(now {regime['vix_level']:.1f})"
                )

        # VIX backwardation (stress signal)
        if regime.get('vix_term_regime') == 'backwardation':
            flags.append(
                f"VIX term structure BACKWARDATED "
                f"(spread: {regime.get('vix_term_structure', 0):.2f}) — market stress signal"
            )

        # Large 1d moves (z-score based)
        for t, df in data.items():
            if len(df) < 60:
                continue
            daily_ret = df['close'].pct_change()
            z = (daily_ret.iloc[-1] - daily_ret.rolling(60).mean().iloc[-1]) / \
                (daily_ret.rolling(60).std().iloc[-1] + 1e-9)
            if abs(z) >= cfg.vol_spike_z:
                pct = daily_ret.iloc[-1] * 100
                flags.append(f"{t}: {pct:+.2f}% today ({z:+.1f} std dev)")

        # High realized vol percentile
        if not momentum_df.empty:
            high_vol = momentum_df[momentum_df['vol_percentile'] >= 0.90]
            for _, row in high_vol.iterrows():
                flags.append(
                    f"{row['ticker']}: realized vol at {row['vol_percentile']*100:.0f}th pctile "
                    f"({row['realized_vol']*100:.0f}% annualized)"
                )

        # Deep drawdowns from recent highs
        if not momentum_df.empty:
            deep_dd = momentum_df[momentum_df['dd_from_20d_high'] <= -0.05]
            for _, row in deep_dd.iterrows():
                flags.append(
                    f"{row['ticker']}: {row['dd_from_20d_high']*100:.1f}% off 20d high"
                )

        # Near 52w extremes
        if not momentum_df.empty:
            at_highs = momentum_df[momentum_df['pct_52w_range'] >= 0.95]
            at_lows = momentum_df[momentum_df['pct_52w_range'] <= 0.05]
            for _, row in at_highs.iterrows():
                flags.append(f"{row['ticker']}: near 52-week HIGH ({row['pct_52w_range']*100:.0f}% of range)")
            for _, row in at_lows.iterrows():
                flags.append(f"{row['ticker']}: near 52-week LOW ({row['pct_52w_range']*100:.0f}% of range)")

        return flags


# ============================================================================
# Main Dashboard
# ============================================================================

class MarketDashboard:
    """Orchestrates all indicators and produces the snapshot report."""

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.loader = DataLoader(self.config)
        self.momentum = MomentumIndicators(self.config)
        self.regime = RegimeIndicators(self.config)
        self.cross_asset = CrossAssetIndicators(self.config)
        self.anomaly = AnomalyDetector(self.config)

        # Output state
        self.indices_momentum: Optional[pd.DataFrame] = None
        self.commodities_momentum: Optional[pd.DataFrame] = None
        self.regime_dict: dict = {}
        self.cross_asset_dict: dict = {}
        self.anomaly_flags: list = []

    def refresh(self) -> 'MarketDashboard':
        self.loader.fetch()
        self.indices_momentum = self.momentum.compute(self.loader.data, self.config.indices)
        self.commodities_momentum = self.momentum.compute(self.loader.data, self.config.commodities)
        self.regime_dict = self.regime.compute(self.loader.data)
        self.cross_asset_dict = self.cross_asset.compute(self.loader.data)
        self.anomaly_flags = self.anomaly.detect(
            self.loader.data,
            pd.concat([self.indices_momentum, self.commodities_momentum], ignore_index=True),
            self.regime_dict,
        )
        return self

    def report(self) -> str:
        """Generate formatted terminal report."""
        lines = []
        lines.append("=" * 78)
        lines.append(f"  MARKET DASHBOARD   |   as of {self.loader.as_of.date()}   |   "
                     f"{datetime.now().strftime('%H:%M')}")
        lines.append("=" * 78)

        # --- Anomaly flags (top priority) ---
        lines.append("\n [ ALERTS ]")
        if self.anomaly_flags:
            for flag in self.anomaly_flags:
                lines.append(f"  > {flag}")
        else:
            lines.append("  (no alerts)")

        # --- Regime ---
        lines.append("\n [ MACRO REGIME ]")
        r = self.regime_dict
        if 'vix_level' in r:
            arrow = 'UP' if r.get('vix_change_1d', 0) > 0 else 'DOWN'
            lines.append(
                f"  VIX:       {r['vix_level']:>6.2f}   "
                f"({arrow} {abs(r['vix_change_1d'])*100:>4.1f}% 1d)   "
                f"pct: {r['vix_percentile_1y']*100:>5.1f}%   "
                f"vs 20MA: {r['vix_vs_ma20']*100:+.1f}%"
            )
        if 'vix_term_structure' in r:
            lines.append(
                f"  VIX term:  {r['vix_term_structure']:>+6.2f}   "
                f"[{r['vix_term_regime'].upper()}]"
            )
        if 'dxy_level' in r:
            lines.append(
                f"  DXY:       {r['dxy_level']:>6.2f}   "
                f"5d: {r['dxy_change_5d']*100:>+5.2f}%   "
                f"20d: {r['dxy_change_20d']*100:>+5.2f}%"
            )
        if 'yield_10y' in r:
            lines.append(
                f"  10Y yield: {r['yield_10y']:>6.2f}%  "
                f"5d: {r['yield_change_5d_bps']:>+5.1f} bps  "
                f"20d: {r['yield_change_20d_bps']:>+5.1f} bps"
            )

        # --- Indices momentum ---
        lines.append("\n [ EQUITY INDEX FUTURES ]")
        lines.append(self._format_momentum_table(self.indices_momentum))

        # --- Commodity momentum ---
        lines.append("\n [ COMMODITY FUTURES ]")
        lines.append(self._format_momentum_table(self.commodities_momentum))

        # --- Cross-asset ---
        lines.append("\n [ CROSS-ASSET SIGNALS ]")
        ca = self.cross_asset_dict
        if 'risk_appetite' in ca:
            lines.append(f"  Risk appetite:        {ca['risk_appetite']}")
        if 'NQ_vs_ES_5d' in ca:
            lines.append(
                f"  NQ-ES (5d):           {ca['NQ_vs_ES_5d']*100:>+5.2f}%     "
                f"NQ-ES (20d): {ca['NQ_vs_ES_20d']*100:>+5.2f}%"
            )
        if 'RTY_vs_ES_5d' in ca:
            lines.append(
                f"  RTY-ES (5d):          {ca['RTY_vs_ES_5d']*100:>+5.2f}%     "
                f"RTY-ES (20d): {ca['RTY_vs_ES_20d']*100:>+5.2f}%"
            )
        if 'brent_wti_spread' in ca:
            lines.append(
                f"  Brent-WTI spread:    ${ca['brent_wti_spread']:>5.2f}   "
                f"(1y pct: {ca['brent_wti_spread_pct_1y']*100:.0f}%)"
            )
        if 'gold_copper_ratio' in ca:
            lines.append(
                f"  Gold/Copper ratio:    {ca['gold_copper_ratio']:>6.1f}   "
                f"(1y pct: {ca['gold_copper_pct_1y']*100:.0f}%)"
            )
        if 'growth_value_ratio' in ca:
            lines.append(
                f"  NQ/YM (growth/value): {ca['growth_value_ratio']:>6.4f}   "
                f"(1y pct: {ca['growth_value_pct_1y']*100:.0f}%)"
            )

        lines.append("\n" + "=" * 78)
        return "\n".join(lines)

    @staticmethod
    def _format_momentum_table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "  (no data)"
        cols_fmt = {
            'ticker': '{:<6}',
            'close': '{:>10.2f}',
            'ret_1d': '{:>+6.2f}%',
            'ret_5d': '{:>+6.2f}%',
            'ret_20d': '{:>+7.2f}%',
            'vs_ma20_pct': '{:>+6.2f}%',
            'vs_ma50_pct': '{:>+6.2f}%',
            'trend_state': '{:<12}',
            'vol_percentile': '{:>5.0f}%',
            'dd_from_20d_high': '{:>+6.2f}%',
        }
        header = (
            f"  {'tkr':<6}{'close':>10}  {'1d':>7} {'5d':>7} {'20d':>8}  "
            f"{'vs20MA':>7} {'vs50MA':>7}  {'trend':<12}{'vol%':>6} {'dd20':>7}"
        )
        lines = [header, "  " + "-" * 82]
        for _, row in df.iterrows():
            line = (
                f"  {row['ticker']:<6}{row['close']:>10.2f}  "
                f"{row['ret_1d']*100:>+6.2f}% {row['ret_5d']*100:>+6.2f}% {row['ret_20d']*100:>+7.2f}%  "
                f"{row['vs_ma20_pct']*100:>+6.2f}% {row['vs_ma50_pct']*100:>+6.2f}%  "
                f"{row['trend_state']:<12}"
                f"{row['vol_percentile']*100:>5.0f}% {row['dd_from_20d_high']*100:>+6.2f}%"
            )
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return all indicators as a dict for programmatic use."""
        return {
            'as_of': self.loader.as_of.date().isoformat() if self.loader.as_of else None,
            'indices': self.indices_momentum.to_dict('records') if self.indices_momentum is not None else [],
            'commodities': self.commodities_momentum.to_dict('records') if self.commodities_momentum is not None else [],
            'regime': self.regime_dict,
            'cross_asset': self.cross_asset_dict,
            'anomaly_flags': self.anomaly_flags,
        }

    def save_snapshot(self, path: str = 'dashboard_snapshot.csv'):
        """Append today's snapshot to historical log for tracking."""
        snap = {'timestamp': pd.Timestamp.now(), 'as_of': self.loader.as_of}
        snap.update({f'regime_{k}': v for k, v in self.regime_dict.items()})
        snap.update({f'xasset_{k}': v for k, v in self.cross_asset_dict.items()})
        for _, row in self.indices_momentum.iterrows():
            for col in ['ret_1d', 'ret_5d', 'ret_20d', 'vs_ma20_pct', 'trend_state', 'vol_percentile']:
                snap[f"{row['ticker']}_{col}"] = row[col]
        for _, row in self.commodities_momentum.iterrows():
            for col in ['ret_1d', 'ret_5d', 'ret_20d', 'vs_ma20_pct', 'trend_state', 'vol_percentile']:
                snap[f"{row['ticker']}_{col}"] = row[col]

        df = pd.DataFrame([snap])
        try:
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_csv(path, index=False)
        print(f"  [snapshot saved to {path}]")


# ============================================================================
# Entry point
# ============================================================================

