"""
Brent + Gold Geopolitical Trading Dashboard
============================================
Focused monitoring pipeline for trading Brent crude and gold futures during
active geopolitical risk regimes (US/Israel-Iran conflict).

Key insight: Brent and gold share safe-haven flows during escalation but
respond to different fundamental drivers. This dashboard tracks both the
composite risk regime AND the contract-specific drivers.

Universe:
    Trading:        BZ=F (Brent), GC=F (Gold)
    Brent context:  CL=F (WTI), RB=F (gasoline), HO=F (heating oil)
    Gold context:   SI=F (silver), GDX (miners ETF), ^TNX (10Y yield)
    Risk regime:    ^VIX, DX-Y.NYB (DXY), ^MOVE (bond vol, if available)
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
class GeoTradeConfig:
    # Core trading contracts
    brent: str = 'BZ=F'
    gold: str = 'GC=F'

    # Brent context
    wti: str = 'CL=F'
    gasoline: str = 'RB=F'
    heating_oil: str = 'HO=F'

    # Gold context
    silver: str = 'SI=F'
    gold_miners: str = 'GDX'

    # Macro / risk regime
    vix: str = '^VIX'
    dxy: str = 'DX-Y.NYB'
    tnx: str = '^TNX'
    # TIPS yield proxy (real rates) - we'll compute from nominal - breakevens if available
    tips_etf: str = 'TIP'   # iShares TIPS ETF as real-rate proxy

    # Parameters
    history_period: str = '1y'
    percentile_window: int = 252
    short_window: int = 20
    vol_window: int = 20

    # Alert thresholds (tuned for geopolitical regime - more sensitive than normal)
    gap_flag_pct: float = 0.010         # flag overnight gap > 1%
    intraday_flag_pct: float = 0.015    # flag daily move > 1.5%
    vix_spike_pct: float = 0.10         # flag VIX up 10%+ in a day
    dxy_move_flag_pct: float = 0.005    # flag DXY move > 0.5% (big for FX)
    correlation_window: int = 20        # for rolling Brent-Gold correlation


# ============================================================================
# Data Layer
# ============================================================================

class DataLoader:
    def __init__(self, config: GeoTradeConfig):
        self.config = config
        self.data: dict = {}
        self.as_of: Optional[pd.Timestamp] = None

    def fetch(self) -> 'DataLoader':
        cfg = self.config
        all_tickers = [
            cfg.brent, cfg.gold,
            cfg.wti, cfg.gasoline, cfg.heating_oil,
            cfg.silver, cfg.gold_miners,
            cfg.vix, cfg.dxy, cfg.tnx, cfg.tips_etf,
        ]
        print(f"Fetching {len(all_tickers)} tickers ({cfg.history_period})...")

        raw = yf.download(
            all_tickers,
            period=cfg.history_period,
            interval='1d',
            progress=False,
            auto_adjust=True,
        )

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
            print(f"  Loaded {len(self.data)}/{len(all_tickers)} tickers, latest: {self.as_of.date()}")
        return self


# ============================================================================
# Contract-specific indicators
# ============================================================================

class ContractMonitor:
    """Deep dive on a single contract: price action, vol state, key levels."""

    def __init__(self, config: GeoTradeConfig):
        self.config = config

    def compute(self, ticker: str, data: dict) -> dict:
        if ticker not in data:
            return {}
        df = data[ticker]
        close = df['close']
        cfg = self.config

        out = {'ticker': ticker, 'close': close.iloc[-1]}

        # Returns at multiple horizons
        for w, label in [(1, '1d'), (5, '5d'), (20, '20d'), (60, '60d')]:
            if len(close) > w:
                out[f'ret_{label}'] = close.pct_change(w).iloc[-1]

        # Overnight gap (open vs prior close)
        if len(df) >= 2:
            prev_close = close.iloc[-2]
            today_open = df['open'].iloc[-1]
            out['overnight_gap'] = (today_open - prev_close) / prev_close
            # Intraday move
            out['intraday_move'] = (close.iloc[-1] - today_open) / today_open

        # Trend state
        if len(close) >= 50:
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            out['ma20'] = ma20
            out['ma50'] = ma50
            out['vs_ma20_pct'] = (close.iloc[-1] - ma20) / ma20
            out['vs_ma50_pct'] = (close.iloc[-1] - ma50) / ma50
            out['trend'] = self._trend(close.iloc[-1], ma20, ma50)

        # Volatility state
        daily_ret = close.pct_change()
        if len(daily_ret) >= cfg.vol_window:
            rvol = daily_ret.rolling(cfg.vol_window).std() * np.sqrt(252)
            out['realized_vol'] = rvol.iloc[-1]
            if len(rvol) >= cfg.percentile_window:
                out['vol_percentile'] = rvol.rolling(cfg.percentile_window).rank(pct=True).iloc[-1]

        # Key levels: recent high/low, distance from them
        if len(close) >= 20:
            high_20 = close.rolling(20).max().iloc[-1]
            low_20 = close.rolling(20).min().iloc[-1]
            out['high_20d'] = high_20
            out['low_20d'] = low_20
            out['dd_from_20d_high'] = (close.iloc[-1] - high_20) / high_20
            out['above_20d_low_pct'] = (close.iloc[-1] - low_20) / low_20

        if len(close) >= 60:
            out['high_60d'] = close.rolling(60).max().iloc[-1]
            out['low_60d'] = close.rolling(60).min().iloc[-1]

        # ATR (for stop sizing)
        if len(df) >= 14:
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - close.shift()).abs(),
                (df['low'] - close.shift()).abs(),
            ], axis=1).max(axis=1)
            out['atr_14'] = tr.rolling(14).mean().iloc[-1]
            out['atr_pct'] = out['atr_14'] / close.iloc[-1]

        return out

    @staticmethod
    def _trend(p, ma_s, ma_l):
        if p > ma_s > ma_l: return 'strong_up'
        if p > ma_s and p > ma_l: return 'up'
        if p < ma_s < ma_l: return 'strong_down'
        if p < ma_s and p < ma_l: return 'down'
        return 'mixed'


# ============================================================================
# Geopolitical Risk Regime
# ============================================================================

class GeoRiskRegime:
    """Composite indicators of geopolitical stress and safe-haven flows."""

    def __init__(self, config: GeoTradeConfig):
        self.config = config

    def compute(self, data: dict) -> dict:
        cfg = self.config
        out = {}

        # VIX
        vix = data.get(cfg.vix)
        if vix is not None:
            v = vix['close']
            out['vix'] = v.iloc[-1]
            out['vix_change_1d'] = v.pct_change().iloc[-1]
            out['vix_change_5d'] = v.pct_change(5).iloc[-1]
            if len(v) >= cfg.percentile_window:
                out['vix_pct_1y'] = v.rolling(cfg.percentile_window).rank(pct=True).iloc[-1]

        # DXY (USD strength)
        dxy = data.get(cfg.dxy)
        if dxy is not None:
            d = dxy['close']
            out['dxy'] = d.iloc[-1]
            out['dxy_change_1d'] = d.pct_change().iloc[-1]
            out['dxy_change_5d'] = d.pct_change(5).iloc[-1]
            out['dxy_change_20d'] = d.pct_change(20).iloc[-1]

        # 10Y yield
        tnx = data.get(cfg.tnx)
        if tnx is not None:
            y = tnx['close']  # yield * 10
            out['yield_10y'] = y.iloc[-1] / 10
            if len(y) > 5:
                out['yield_chg_5d_bps'] = (y.iloc[-1] - y.iloc[-6]) * 10
            if len(y) > 20:
                out['yield_chg_20d_bps'] = (y.iloc[-1] - y.iloc[-21]) * 10

        # Real rate proxy via TIPS ETF (inverse relationship with gold)
        tips = data.get(cfg.tips_etf)
        if tips is not None:
            t = tips['close']
            out['tips_etf'] = t.iloc[-1]
            out['tips_change_5d'] = t.pct_change(5).iloc[-1]
            out['tips_change_20d'] = t.pct_change(20).iloc[-1]
            # TIPS up = real yields down = gold tailwind

        # Composite risk score (0-100, higher = more risk-off)
        risk_score = 0
        components = 0
        if 'vix_pct_1y' in out:
            risk_score += out['vix_pct_1y'] * 100
            components += 1
        if 'dxy_change_5d' in out:
            # DXY up in stress = risk-off flow into USD
            dxy_risk = 50 + (out['dxy_change_5d'] * 1000)  # rough scaling
            risk_score += np.clip(dxy_risk, 0, 100)
            components += 1
        if 'yield_chg_5d_bps' in out:
            # Falling yields = flight to quality = risk-off
            yield_risk = 50 - out['yield_chg_5d_bps']
            risk_score += np.clip(yield_risk, 0, 100)
            components += 1
        if components > 0:
            out['composite_risk_score'] = risk_score / components
            out['risk_regime'] = self._risk_label(out['composite_risk_score'])

        return out

    @staticmethod
    def _risk_label(score):
        if score >= 75: return 'EXTREME_RISK_OFF'
        if score >= 60: return 'elevated_risk_off'
        if score >= 40: return 'neutral'
        if score >= 25: return 'risk_on'
        return 'strong_risk_on'


# ============================================================================
# Brent-specific signals
# ============================================================================

class BrentContext:
    """Brent-specific context: spreads, crack, WTI relationship."""

    def __init__(self, config: GeoTradeConfig):
        self.config = config

    def compute(self, data: dict) -> dict:
        cfg = self.config
        out = {}

        brent = data.get(cfg.brent)
        wti = data.get(cfg.wti)
        gasoline = data.get(cfg.gasoline)
        heating_oil = data.get(cfg.heating_oil)

        # Brent-WTI spread: widening = non-US supply stress (geopolitical premium)
        if brent is not None and wti is not None:
            bz = brent['close']
            cl = wti['close']
            spread = bz - cl
            out['brent_wti_spread'] = spread.iloc[-1]
            out['brent_wti_spread_5d_chg'] = spread.iloc[-1] - spread.iloc[-6] if len(spread) > 5 else np.nan
            out['brent_wti_spread_20d_chg'] = spread.iloc[-1] - spread.iloc[-21] if len(spread) > 20 else np.nan
            if len(spread) >= cfg.percentile_window:
                out['brent_wti_spread_pct_1y'] = spread.rolling(cfg.percentile_window).rank(pct=True).iloc[-1]
            # Flag: is the geopolitical premium building?
            if not np.isnan(out.get('brent_wti_spread_5d_chg', np.nan)):
                if out['brent_wti_spread_5d_chg'] > 1.0:
                    out['geopolitical_premium'] = 'BUILDING (spread widening)'
                elif out['brent_wti_spread_5d_chg'] < -1.0:
                    out['geopolitical_premium'] = 'unwinding (spread narrowing)'
                else:
                    out['geopolitical_premium'] = 'stable'

        # Crack spread proxy (refiner margin): (gasoline + heating oil)/2 - crude
        # Positive and rising = demand signal
        if wti is not None and gasoline is not None and heating_oil is not None:
            # Gasoline/heating oil trade in $/gal, WTI in $/bbl (42 gal/bbl)
            cl_px = wti['close'].iloc[-1]
            gas_px = gasoline['close'].iloc[-1] * 42
            ho_px = heating_oil['close'].iloc[-1] * 42
            out['crack_spread_321'] = (2 * gas_px + ho_px) / 3 - cl_px
            # 3-2-1 crack: 3 bbl crude -> 2 bbl gasoline + 1 bbl heating oil

        # Brent-Gold correlation (geopolitical regimes push them together)
        gold = data.get(cfg.gold)
        if brent is not None and gold is not None:
            br_ret = brent['close'].pct_change()
            gd_ret = gold['close'].pct_change()
            if len(br_ret) >= cfg.correlation_window:
                corr = br_ret.rolling(cfg.correlation_window).corr(gd_ret).iloc[-1]
                out['brent_gold_correlation_20d'] = corr
                # High correlation = geopolitical regime dominant

        return out


# ============================================================================
# Gold-specific signals
# ============================================================================

class GoldContext:
    """Gold-specific context: silver ratio, miners, real rates."""

    def __init__(self, config: GeoTradeConfig):
        self.config = config

    def compute(self, data: dict) -> dict:
        cfg = self.config
        out = {}

        gold = data.get(cfg.gold)
        silver = data.get(cfg.silver)
        miners = data.get(cfg.gold_miners)

        # Gold-silver ratio: rising = fear dominant; falling = growth/inflation play
        if gold is not None and silver is not None:
            ratio = gold['close'] / silver['close']
            out['gold_silver_ratio'] = ratio.iloc[-1]
            if len(ratio) >= cfg.percentile_window:
                out['gold_silver_pct_1y'] = ratio.rolling(cfg.percentile_window).rank(pct=True).iloc[-1]
            out['gold_silver_5d_chg'] = ratio.pct_change(5).iloc[-1] if len(ratio) > 5 else np.nan
            # Interpretation
            if out.get('gold_silver_pct_1y', 0.5) > 0.8:
                out['gold_silver_signal'] = 'FEAR (ratio elevated)'
            elif out.get('gold_silver_pct_1y', 0.5) < 0.2:
                out['gold_silver_signal'] = 'growth/inflation play'
            else:
                out['gold_silver_signal'] = 'neutral'

        # Gold miners vs gold: miners lead gold at turning points
        if gold is not None and miners is not None:
            gd_ret_5d = gold['close'].pct_change(5).iloc[-1]
            gdx_ret_5d = miners['close'].pct_change(5).iloc[-1]
            out['gdx_5d_return'] = gdx_ret_5d
            out['gold_5d_return'] = gd_ret_5d
            out['gdx_gold_5d_diff'] = gdx_ret_5d - gd_ret_5d
            # Miners outperforming = bullish confirmation
            if out['gdx_gold_5d_diff'] > 0.02:
                out['miners_signal'] = 'BULLISH (miners leading)'
            elif out['gdx_gold_5d_diff'] < -0.02:
                out['miners_signal'] = 'BEARISH (miners lagging)'
            else:
                out['miners_signal'] = 'neutral'

        return out


# ============================================================================
# Alert engine
# ============================================================================

class AlertEngine:
    """Generate priority-ranked alerts for Brent + Gold decisions."""

    def __init__(self, config: GeoTradeConfig):
        self.config = config

    def generate(
        self,
        brent_stats: dict,
        gold_stats: dict,
        regime: dict,
        brent_ctx: dict,
        gold_ctx: dict,
        data: dict,
    ) -> list:
        cfg = self.config
        alerts = []  # (priority, message) - priority 1 = highest

        # CRITICAL: Gap moves on trading contracts
        for stats, name in [(brent_stats, 'BRENT'), (gold_stats, 'GOLD')]:
            if 'overnight_gap' in stats:
                gap = stats['overnight_gap']
                if abs(gap) >= cfg.gap_flag_pct:
                    direction = 'UP' if gap > 0 else 'DOWN'
                    alerts.append((1,
                        f"{name} GAPPED {direction} {abs(gap)*100:.2f}% overnight "
                        f"(current: {stats['close']:.2f})"
                    ))
            if 'ret_1d' in stats and abs(stats['ret_1d']) >= cfg.intraday_flag_pct:
                alerts.append((2,
                    f"{name} moved {stats['ret_1d']*100:+.2f}% today "
                    f"({stats.get('realized_vol', 0)*100:.0f}% vol regime)"
                ))

        # HIGH: Risk regime shifts
        if 'vix_change_1d' in regime and abs(regime['vix_change_1d']) >= cfg.vix_spike_pct:
            direction = 'SPIKE' if regime['vix_change_1d'] > 0 else 'CRUSH'
            alerts.append((2,
                f"VIX {direction}: {regime['vix_change_1d']*100:+.1f}% "
                f"(now {regime['vix']:.1f}, {regime.get('vix_pct_1y', 0)*100:.0f}th pctile)"
            ))

        # HIGH: DXY moves (affects both Brent and gold inversely)
        if 'dxy_change_1d' in regime and abs(regime['dxy_change_1d']) >= cfg.dxy_move_flag_pct:
            alerts.append((2,
                f"DXY moved {regime['dxy_change_1d']*100:+.2f}% "
                f"(5d: {regime.get('dxy_change_5d', 0)*100:+.2f}%) - "
                f"{'headwind' if regime['dxy_change_1d'] > 0 else 'tailwind'} for commodities"
            ))

        # HIGH: Brent-WTI spread regime
        if 'geopolitical_premium' in brent_ctx:
            if brent_ctx['geopolitical_premium'].startswith('BUILDING'):
                alerts.append((2,
                    f"Brent-WTI spread WIDENING: now ${brent_ctx['brent_wti_spread']:.2f} "
                    f"(+${brent_ctx['brent_wti_spread_5d_chg']:.2f} over 5d) - geopolitical premium building"
                ))

        # MEDIUM: Correlation regime
        if 'brent_gold_correlation_20d' in brent_ctx:
            corr = brent_ctx['brent_gold_correlation_20d']
            if corr > 0.5:
                alerts.append((3,
                    f"Brent-Gold correlation HIGH ({corr:.2f}) - geopolitical regime active, "
                    f"positions partially redundant"
                ))
            elif corr < -0.2:
                alerts.append((3,
                    f"Brent-Gold correlation NEGATIVE ({corr:.2f}) - diverging drivers, "
                    f"good diversification"
                ))

        # MEDIUM: Miners signal
        if gold_ctx.get('miners_signal', '').startswith(('BULLISH', 'BEARISH')):
            alerts.append((3, f"Gold miners: {gold_ctx['miners_signal']}"))

        # MEDIUM: Key level proximity
        for stats, name in [(brent_stats, 'BRENT'), (gold_stats, 'GOLD')]:
            if 'dd_from_20d_high' in stats:
                if stats['dd_from_20d_high'] > -0.005:
                    alerts.append((3,
                        f"{name} testing 20d high ({stats.get('high_20d', 0):.2f})"
                    ))
            if 'above_20d_low_pct' in stats and 'high_20d' in stats and 'low_20d' in stats:
                if stats['above_20d_low_pct'] < 0.005:
                    alerts.append((3,
                        f"{name} testing 20d low ({stats.get('low_20d', 0):.2f})"
                    ))

        # LOW: Vol regime context
        for stats, name in [(brent_stats, 'BRENT'), (gold_stats, 'GOLD')]:
            if stats.get('vol_percentile', 0) >= 0.85:
                alerts.append((4,
                    f"{name} vol at {stats['vol_percentile']*100:.0f}th pctile "
                    f"({stats['realized_vol']*100:.0f}% annualized) - size smaller"
                ))

        alerts.sort(key=lambda x: x[0])
        return alerts


# ============================================================================
# Main dashboard
# ============================================================================

class GeoTradeDashboard:
    def __init__(self, config: Optional[GeoTradeConfig] = None):
        self.config = config or GeoTradeConfig()
        self.loader = DataLoader(self.config)
        self.contract = ContractMonitor(self.config)
        self.regime = GeoRiskRegime(self.config)
        self.brent_ctx = BrentContext(self.config)
        self.gold_ctx = GoldContext(self.config)
        self.alerts = AlertEngine(self.config)

        # State
        self.brent_stats: dict = {}
        self.gold_stats: dict = {}
        self.regime_stats: dict = {}
        self.brent_context: dict = {}
        self.gold_context: dict = {}
        self.alert_list: list = []

    def refresh(self) -> 'GeoTradeDashboard':
        self.loader.fetch()
        self.brent_stats = self.contract.compute(self.config.brent, self.loader.data)
        self.gold_stats = self.contract.compute(self.config.gold, self.loader.data)
        self.regime_stats = self.regime.compute(self.loader.data)
        self.brent_context = self.brent_ctx.compute(self.loader.data)
        self.gold_context = self.gold_ctx.compute(self.loader.data)
        self.alert_list = self.alerts.generate(
            self.brent_stats, self.gold_stats, self.regime_stats,
            self.brent_context, self.gold_context, self.loader.data,
        )
        return self

    def report(self) -> str:
        lines = []
        lines.append("=" * 82)
        lines.append(f"  BRENT + GOLD DASHBOARD   |   as of {self.loader.as_of.date()}   |   "
                     f"{datetime.now().strftime('%H:%M')}")
        lines.append("=" * 82)

        # --- Alerts ---
        lines.append("\n [ ALERTS ]")
        if self.alert_list:
            prev_priority = None
            for priority, msg in self.alert_list:
                marker = {1: '[!!!]', 2: '[!! ]', 3: '[!  ]', 4: '[.  ]'}.get(priority, '[   ]')
                lines.append(f"  {marker} {msg}")
        else:
            lines.append("  (no alerts)")

        # --- Risk regime ---
        lines.append("\n [ RISK REGIME ]")
        r = self.regime_stats
        if 'composite_risk_score' in r:
            lines.append(
                f"  Composite risk: {r['composite_risk_score']:>5.1f}/100  "
                f"[{r['risk_regime'].upper().replace('_', ' ')}]"
            )
        if 'vix' in r:
            lines.append(
                f"  VIX:       {r['vix']:>6.2f}  "
                f"1d: {r['vix_change_1d']*100:>+5.1f}%  "
                f"5d: {r['vix_change_5d']*100:>+5.1f}%  "
                f"1y pct: {r.get('vix_pct_1y', 0)*100:>5.1f}%"
            )
        if 'dxy' in r:
            lines.append(
                f"  DXY:       {r['dxy']:>6.2f}  "
                f"1d: {r['dxy_change_1d']*100:>+5.2f}%  "
                f"5d: {r['dxy_change_5d']*100:>+5.2f}%  "
                f"20d: {r['dxy_change_20d']*100:>+5.2f}%"
            )
        if 'yield_10y' in r:
            lines.append(
                f"  10Y yield: {r['yield_10y']:>6.2f}%  "
                f"5d: {r.get('yield_chg_5d_bps', 0):>+5.1f} bps  "
                f"20d: {r.get('yield_chg_20d_bps', 0):>+5.1f} bps"
            )
        if 'tips_etf' in r:
            lines.append(
                f"  TIPS ETF:  {r['tips_etf']:>6.2f}  "
                f"5d: {r['tips_change_5d']*100:>+5.2f}%  "
                f"20d: {r['tips_change_20d']*100:>+5.2f}%  (proxy for real yields, inverse)"
            )

        # --- Brent ---
        lines.append("\n [ BRENT (BZ=F) ]")
        lines.append(self._format_contract(self.brent_stats))
        lines.append("\n   Context:")
        ctx = self.brent_context
        if 'brent_wti_spread' in ctx:
            lines.append(
                f"   Brent-WTI spread:    ${ctx['brent_wti_spread']:>+6.2f}  "
                f"(5d chg: ${ctx.get('brent_wti_spread_5d_chg', 0):>+5.2f})  "
                f"1y pct: {ctx.get('brent_wti_spread_pct_1y', 0)*100:>4.0f}%"
            )
        if 'geopolitical_premium' in ctx:
            lines.append(f"   Geopolitical premium: {ctx['geopolitical_premium']}")
        if 'crack_spread_321' in ctx:
            lines.append(f"   3-2-1 crack spread:  ${ctx['crack_spread_321']:>+6.2f}/bbl (refiner margin)")
        if 'brent_gold_correlation_20d' in ctx:
            lines.append(f"   Brent-Gold corr 20d: {ctx['brent_gold_correlation_20d']:>+5.2f}")

        # --- Gold ---
        lines.append("\n [ GOLD (GC=F) ]")
        lines.append(self._format_contract(self.gold_stats))
        lines.append("\n   Context:")
        gctx = self.gold_context
        if 'gold_silver_ratio' in gctx:
            lines.append(
                f"   Gold/Silver ratio:   {gctx['gold_silver_ratio']:>6.1f}  "
                f"1y pct: {gctx.get('gold_silver_pct_1y', 0)*100:>4.0f}%  "
                f"[{gctx.get('gold_silver_signal', 'n/a')}]"
            )
        if 'gdx_gold_5d_diff' in gctx:
            lines.append(
                f"   GDX vs Gold (5d):    "
                f"GDX: {gctx['gdx_5d_return']*100:>+5.2f}%  "
                f"Gold: {gctx['gold_5d_return']*100:>+5.2f}%  "
                f"diff: {gctx['gdx_gold_5d_diff']*100:>+5.2f}%  "
                f"[{gctx.get('miners_signal', 'n/a')}]"
            )

        lines.append("\n" + "=" * 82)
        return "\n".join(lines)

    @staticmethod
    def _format_contract(stats: dict) -> str:
        if not stats:
            return "   (no data)"
        lines = []
        lines.append(
            f"   close: {stats['close']:>8.2f}  |  "
            f"1d: {stats.get('ret_1d', 0)*100:>+5.2f}%  "
            f"5d: {stats.get('ret_5d', 0)*100:>+5.2f}%  "
            f"20d: {stats.get('ret_20d', 0)*100:>+6.2f}%  "
            f"60d: {stats.get('ret_60d', 0)*100:>+6.2f}%"
        )
        if 'overnight_gap' in stats:
            lines.append(
                f"   overnight gap: {stats['overnight_gap']*100:>+5.2f}%  |  "
                f"intraday: {stats['intraday_move']*100:>+5.2f}%"
            )
        if 'trend' in stats:
            lines.append(
                f"   trend: {stats['trend']:<12} |  "
                f"vs 20MA: {stats['vs_ma20_pct']*100:>+5.2f}%  "
                f"vs 50MA: {stats['vs_ma50_pct']*100:>+5.2f}%"
            )
        if 'realized_vol' in stats:
            lines.append(
                f"   vol: {stats['realized_vol']*100:>5.1f}% ann.  |  "
                f"pctile: {stats.get('vol_percentile', 0)*100:>4.0f}%  |  "
                f"ATR: ${stats.get('atr_14', 0):>6.2f} ({stats.get('atr_pct', 0)*100:.1f}%)"
            )
        if 'high_20d' in stats:
            lines.append(
                f"   20d range: [{stats['low_20d']:>7.2f} - {stats['high_20d']:>7.2f}]  |  "
                f"from high: {stats['dd_from_20d_high']*100:>+5.2f}%  "
                f"from low: {stats['above_20d_low_pct']*100:>+5.2f}%"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            'as_of': self.loader.as_of.date().isoformat() if self.loader.as_of else None,
            'brent': self.brent_stats,
            'gold': self.gold_stats,
            'regime': self.regime_stats,
            'brent_context': self.brent_context,
            'gold_context': self.gold_context,
            'alerts': self.alert_list,
        }

    def save_snapshot(self, path: str = 'geo_snapshot.csv'):
        """Append today's reading to historical log."""
        snap = {'timestamp': pd.Timestamp.now(), 'as_of': self.loader.as_of}
        for k, v in self.regime_stats.items():
            snap[f'regime_{k}'] = v
        for k, v in self.brent_stats.items():
            if k != 'ticker':
                snap[f'brent_{k}'] = v
        for k, v in self.gold_stats.items():
            if k != 'ticker':
                snap[f'gold_{k}'] = v
        for k, v in self.brent_context.items():
            snap[f'bctx_{k}'] = v
        for k, v in self.gold_context.items():
            snap[f'gctx_{k}'] = v
        snap['n_alerts'] = len(self.alert_list)

        df = pd.DataFrame([snap])
        try:
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_csv(path, index=False)
        print(f"  [snapshot saved to {path}]")