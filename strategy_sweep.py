"""
Strategy Sweep — Regime-Conditional Parameter Search
=====================================================
Tests trend-following and mean-reversion strategies across (n, m) parameter
grids, conditional on the current VIX regime.

The key insight: a global backtest averages across regimes and usually finds
nothing. By filtering historical signal dates to only those where the regime
matched the CURRENT regime, we answer the practical question:
"When we've been in this regime before, what (n, m) worked?"

Honest caveats baked into output:
- Minimum n_obs filter (default 15) — smaller samples are noise
- Both top-5 and bottom-5 reported — shows dispersion
- Median + win rate alongside mean — outlier protection
- No look-ahead: SMA at date t uses only data < t

Strategies tested:
- Trend-following: enter when close > SMA(n) AND SMA(n) > SMA(m), require n < m
- Mean-reversion: enter when close is k% below SMA(n), where n parameter
  controls lookback and m controls the threshold (m=1 → -1%, m=2 → -2%, etc.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SweepConfig:
    n_range: tuple = (3, 5, 7, 10, 14, 20)      # shorter MA lookbacks
    m_range: tuple = (10, 20, 30, 50)            # longer MA lookbacks
    holding_days: int = 5                         # hold position this many days
    mr_thresholds: tuple = (0.01, 0.02, 0.03, 0.05)  # -1%, -2%, -3%, -5% below SMA
    mr_lookbacks: tuple = (5, 10, 20, 50)
    min_observations: int = 15                    # drop (n,m) combos with fewer samples
    top_k: int = 5                                 # how many top/bottom results to return


# ============================================================================
# Strategy signal generators
# ============================================================================

def trend_following_signals(prices: pd.Series, n: int, m: int) -> pd.Series:
    """
    Returns boolean series: True when trend-following says long.
    Signal: close > SMA(n) AND SMA(n) > SMA(m)
    Requires n < m.
    """
    if n >= m:
        return pd.Series(False, index=prices.index)
    sma_n = prices.rolling(n).mean()
    sma_m = prices.rolling(m).mean()
    return (prices > sma_n) & (sma_n > sma_m)


def mean_reversion_signals(prices: pd.Series, lookback: int, threshold: float) -> pd.Series:
    """
    Returns boolean series: True when mean-reversion says long.
    Signal: close is `threshold` or more below SMA(lookback)
    (i.e. z-score-like, but simpler: percentage deviation)
    """
    sma = prices.rolling(lookback).mean()
    pct_below = (prices - sma) / sma
    return pct_below <= -threshold


# ============================================================================
# Backtest engine (regime-conditional)
# ============================================================================

def backtest_signals(
    prices: pd.Series,
    signals: pd.Series,
    regime_series: pd.Series,
    current_regime: str,
    holding_days: int = 5,
) -> dict:
    """
    For each date where signal=True AND regime=current_regime,
    record the forward holding_days return.
    """
    # Compute forward returns
    fwd_ret = prices.pct_change(holding_days).shift(-holding_days)

    # Align everything on a common index
    df = pd.DataFrame({
        'signal': signals,
        'regime': regime_series,
        'fwd_ret': fwd_ret,
    }).dropna()

    # Filter: signal triggered AND regime matches current
    triggered = df[(df['signal']) & (df['regime'] == current_regime)]

    if len(triggered) == 0:
        return None

    returns = triggered['fwd_ret']
    return {
        'n_obs': len(returns),
        'mean': returns.mean(),
        'median': returns.median(),
        'std': returns.std(),
        'win_rate': (returns > 0).mean(),
        'best': returns.max(),
        'worst': returns.min(),
        'sharpe_approx': returns.mean() / returns.std() if returns.std() > 0 else 0,
    }


# ============================================================================
# Parameter sweep
# ============================================================================

class StrategySweep:
    """
    Runs a grid sweep over trend-following and mean-reversion parameters,
    conditional on the current VIX regime from a fitted RegimeModel.
    """

    def __init__(self, config: Optional[SweepConfig] = None):
        self.config = config or SweepConfig()
        self.trend_results: Optional[pd.DataFrame] = None
        self.mr_results: Optional[pd.DataFrame] = None

    def run_trend_sweep(
        self,
        prices: pd.Series,
        regime_series: pd.Series,
        current_regime: str,
    ) -> pd.DataFrame:
        """Sweep (n, m) for trend-following."""
        cfg = self.config
        rows = []
        for n in cfg.n_range:
            for m in cfg.m_range:
                if n >= m:
                    continue
                signals = trend_following_signals(prices, n, m)
                result = backtest_signals(
                    prices, signals, regime_series, current_regime,
                    holding_days=cfg.holding_days,
                )
                if result is None or result['n_obs'] < cfg.min_observations:
                    continue
                rows.append({'strategy': 'trend', 'n': n, 'm': m, **result})
        self.trend_results = pd.DataFrame(rows)
        return self.trend_results

    def run_mr_sweep(
        self,
        prices: pd.Series,
        regime_series: pd.Series,
        current_regime: str,
    ) -> pd.DataFrame:
        """Sweep (lookback, threshold) for mean-reversion."""
        cfg = self.config
        rows = []
        for lookback in cfg.mr_lookbacks:
            for threshold in cfg.mr_thresholds:
                signals = mean_reversion_signals(prices, lookback, threshold)
                result = backtest_signals(
                    prices, signals, regime_series, current_regime,
                    holding_days=cfg.holding_days,
                )
                if result is None or result['n_obs'] < cfg.min_observations:
                    continue
                rows.append({
                    'strategy': 'mean_rev',
                    'lookback': lookback,
                    'threshold_pct': threshold * 100,
                    **result,
                })
        self.mr_results = pd.DataFrame(rows)
        return self.mr_results

    def run_all(
        self,
        prices: pd.Series,
        regime_series: pd.Series,
        current_regime: str,
    ) -> dict:
        """Run both strategy families, return structured output."""
        trend = self.run_trend_sweep(prices, regime_series, current_regime)
        mr = self.run_mr_sweep(prices, regime_series, current_regime)

        output = {
            'current_regime': current_regime,
            'trend_all': trend,
            'mr_all': mr,
            'trend_top': None,
            'trend_bottom': None,
            'mr_top': None,
            'mr_bottom': None,
        }

        k = self.config.top_k
        if trend is not None and len(trend) > 0:
            sorted_trend = trend.sort_values('mean', ascending=False)
            output['trend_top'] = sorted_trend.head(k).reset_index(drop=True)
            output['trend_bottom'] = sorted_trend.tail(k).sort_values('mean').reset_index(drop=True)

        if mr is not None and len(mr) > 0:
            sorted_mr = mr.sort_values('mean', ascending=False)
            output['mr_top'] = sorted_mr.head(k).reset_index(drop=True)
            output['mr_bottom'] = sorted_mr.tail(k).sort_values('mean').reset_index(drop=True)

        return output


# ============================================================================
# Convenience: run on asset(s) given a fitted regime model
# ============================================================================

def sweep_from_regime_model(
    regime_model,
    target_prices: dict,
    config: Optional[SweepConfig] = None,
) -> dict:
    """
    Convenience function: given a fitted RegimeModel and a dict of asset
    price series, run the sweep for each asset conditional on the regime
    model's current_state.

    Parameters
    ----------
    regime_model : fitted markov_regime.RegimeModel
    target_prices : dict like {'BZ=F': brent_series, 'GC=F': gold_series}
    config : SweepConfig (optional)

    Returns
    -------
    dict of {asset_name: sweep_output} where sweep_output follows the
    StrategySweep.run_all structure.
    """
    if regime_model.regime_series is None:
        raise RuntimeError("regime_model must be fitted first")

    results = {}
    for name, prices in target_prices.items():
        sweeper = StrategySweep(config)
        # Align prices to the regime series date range
        aligned_prices = prices.reindex(regime_model.regime_series.index).dropna()
        if len(aligned_prices) < 60:
            continue
        results[name] = sweeper.run_all(
            aligned_prices,
            regime_model.regime_series,
            regime_model.current_state,
        )
    return results


# ============================================================================
# Formatting helpers
# ============================================================================

def format_sweep_table(df: pd.DataFrame, show_cols: Optional[list] = None) -> pd.DataFrame:
    """Return a display-formatted version of sweep results."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    for col in ['mean', 'median', 'std', 'win_rate', 'best', 'worst']:
        if col in df.columns:
            df[col] = (df[col] * 100).round(2)
    if 'sharpe_approx' in df.columns:
        df['sharpe_approx'] = df['sharpe_approx'].round(2)
    if show_cols:
        df = df[[c for c in show_cols if c in df.columns]]
    return df


if __name__ == '__main__':
    # Smoke test
    import yfinance as yf
    from markov_regime import RegimeModel

    print("Fetching data for smoke test...")
    raw = yf.download(['^VIX', 'BZ=F', 'GC=F'], period='3y',
                      interval='1d', progress=False, auto_adjust=True)
    vix = raw['Close']['^VIX'].dropna()
    brent = raw['Close']['BZ=F'].dropna()
    gold = raw['Close']['GC=F'].dropna()

    model = RegimeModel().fit(
        driver_series=vix,
        target_series={'BZ=F': brent, 'GC=F': gold},
    )
    print(f"Current regime: {model.current_state}")

    results = sweep_from_regime_model(
        model, {'BZ=F': brent, 'GC=F': gold},
    )

    for asset, res in results.items():
        print(f"\n{'='*70}")
        print(f"{asset} — current regime: {res['current_regime']}")
        print(f"{'='*70}")
        print("\nTrend top 5:")
        print(format_sweep_table(res['trend_top']).to_string(index=False))
        print("\nMean-rev top 5:")
        print(format_sweep_table(res['mr_top']).to_string(index=False))