"""
Markov Regime Model — Tailored for Brent/Gold Geopolitical Trading
===================================================================
Adaptation of the state-transition library for the CME challenge.

Key design choices:
1. Regime is classified from VIX percentile (or any stress-driver series),
   NOT from the target asset's own returns. This produces a meaningful,
   persistent regime signal rather than a near-50/50 coin flip.

2. Three states (calm/normal/stressed) rather than binary. Middle state
   captures the majority of trading days and prevents transition matrix
   estimates from being dominated by regime-boundary noise.

3. Multi-step forward projections out to the contest horizon (5 days) so
   you can see regime persistence probabilities for each day of the week.

4. Conditional return/vol analysis per regime for your trading assets
   (Brent, Gold) so you can size positions based on what each regime
   historically pays.

Usage:
    from markov_regime import RegimeModel

    # Fit from historical data
    model = RegimeModel().fit(vix_series, target_series={'BZ=F': brent, 'GC=F': gold})

    # Print current state and forward projections
    print(model.report())

    # Or get probabilities programmatically
    probs = model.forecast(horizon=5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RegimeConfig:
    # How to bucket the regime driver
    n_states: int = 3
    state_labels: tuple = ('calm', 'normal', 'stressed')
    # Percentile thresholds for state boundaries (used when n_states=3)
    # E.g. (0.33, 0.67) means bottom third = calm, middle third = normal, top third = stressed
    percentile_thresholds: tuple = (0.33, 0.67)

    # Lookback for building transition matrix (days)
    training_window: int = 504  # 2 years

    # Percentile rolling window for regime classification
    classification_window: int = 252  # 1 year


# ============================================================================
# Core Regime Model
# ============================================================================

class RegimeModel:
    """
    Markov regime model driven by a stress indicator (typically VIX).

    The driver series is classified into N regimes via rolling percentile,
    then a transition matrix is estimated from the classified sequence.
    Forward projections use matrix powers, exactly as in the original library.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.transition_matrix: Optional[pd.DataFrame] = None
        self.regime_series: Optional[pd.Series] = None
        self.current_state: Optional[str] = None
        self.current_driver_value: Optional[float] = None
        self.current_driver_percentile: Optional[float] = None
        self.conditional_stats: dict = {}  # per-regime return/vol for each target

    # ------------------------------------------------------------------------
    # Classification (replaces the binary up/down logic from original library)
    # ------------------------------------------------------------------------
    def _classify_regimes(self, driver: pd.Series) -> pd.Series:
        """Convert a continuous driver series into discrete regime labels."""
        cfg = self.config
        # Rolling percentile of driver value
        pct_series = driver.rolling(cfg.classification_window).rank(pct=True)

        # Bucket into states
        labels = pd.Series(index=driver.index, dtype=object)
        if cfg.n_states == 2:
            labels[pct_series < 0.5] = cfg.state_labels[0]
            labels[pct_series >= 0.5] = cfg.state_labels[1]
        elif cfg.n_states == 3:
            t1, t2 = cfg.percentile_thresholds
            labels[pct_series < t1] = cfg.state_labels[0]
            labels[(pct_series >= t1) & (pct_series < t2)] = cfg.state_labels[1]
            labels[pct_series >= t2] = cfg.state_labels[2]
        else:
            raise ValueError(f"n_states={cfg.n_states} not supported (use 2 or 3)")
        return labels

    # ------------------------------------------------------------------------
    # Transition matrix (generalizes the original state_transition_matrix)
    # ------------------------------------------------------------------------
    def _build_transition_matrix(self, regime_series: pd.Series) -> pd.DataFrame:
        """
        Count state -> next-state transitions, normalize rows to probabilities.
        This is the same logic as the original library, generalized to N states.
        """
        cfg = self.config
        df = pd.DataFrame({
            'prev_regime': regime_series.shift(1),
            'regime': regime_series,
        }).dropna()

        # Crosstab gives us the raw counts
        counts = pd.crosstab(df['prev_regime'], df['regime'])

        # Ensure all states present in both dimensions (even if count=0)
        all_states = list(cfg.state_labels)
        counts = counts.reindex(index=all_states, columns=all_states, fill_value=0)

        # Normalize rows to probabilities
        row_sums = counts.sum(axis=1)
        # Avoid division by zero if a state never occurred
        row_sums[row_sums == 0] = 1
        tm = counts.div(row_sums, axis=0)
        return tm

    # ------------------------------------------------------------------------
    # Conditional statistics per regime (new)
    # ------------------------------------------------------------------------
    def _conditional_stats(
        self,
        regime_series: pd.Series,
        target_series: dict,
    ) -> dict:
        """
        For each target asset and each regime, compute historical:
          - mean daily return
          - daily vol
          - 5-day forward return (since contest holding is ~5d)
          - win rate of 5-day forward return
        This tells you whether each regime is actually profitable to trade.
        """
        stats = {}
        for name, series in target_series.items():
            returns = series.pct_change()
            fwd_5d = series.pct_change(5).shift(-5)  # 5-day forward return

            # Align with regime series
            aligned = pd.DataFrame({
                'regime': regime_series,
                'ret_1d': returns,
                'ret_5d_fwd': fwd_5d,
            }).dropna()

            per_regime = {}
            for state in self.config.state_labels:
                subset = aligned[aligned['regime'] == state]
                if len(subset) == 0:
                    continue
                per_regime[state] = {
                    'n_obs': len(subset),
                    'mean_ret_1d': subset['ret_1d'].mean(),
                    'vol_1d': subset['ret_1d'].std(),
                    'mean_ret_5d_fwd': subset['ret_5d_fwd'].mean(),
                    'median_ret_5d_fwd': subset['ret_5d_fwd'].median(),
                    'win_rate_5d_fwd': (subset['ret_5d_fwd'] > 0).mean(),
                    'best_5d_fwd': subset['ret_5d_fwd'].max(),
                    'worst_5d_fwd': subset['ret_5d_fwd'].min(),
                }
            stats[name] = per_regime
        return stats

    # ------------------------------------------------------------------------
    # Fit (main entry point for training)
    # ------------------------------------------------------------------------
    def fit(
        self,
        driver_series: pd.Series,
        target_series: Optional[dict] = None,
    ) -> 'RegimeModel':
        """
        Fit the regime model.

        Parameters
        ----------
        driver_series : pd.Series
            The stress indicator (e.g. VIX close) used to classify regimes.
        target_series : dict, optional
            Dict of {asset_name: price_series} for computing conditional
            return/vol stats per regime. E.g. {'BZ=F': brent, 'GC=F': gold}
        """
        cfg = self.config

        # Trim to training window
        driver = driver_series.dropna()
        if len(driver) > cfg.training_window:
            driver = driver.iloc[-cfg.training_window:]

        # Classify
        self.regime_series = self._classify_regimes(driver).dropna()

        # Guard against insufficient data
        if len(self.regime_series) < 30:
            raise ValueError(
                f"Insufficient data to fit regime model: only {len(self.regime_series)} "
                f"classified regime observations after warmup. "
                f"Driver series has {len(driver)} points, but classification_window="
                f"{cfg.classification_window} burns the first {cfg.classification_window-1} "
                f"for rolling percentile. Supply more history (need at least "
                f"{cfg.classification_window + 30} days) or reduce "
                f"classification_window in RegimeConfig."
            )

        # Current state (most recent non-NA)
        self.current_state = self.regime_series.iloc[-1]
        self.current_driver_value = driver.iloc[-1]
        pct_now = driver.rolling(cfg.classification_window).rank(pct=True).iloc[-1]
        self.current_driver_percentile = pct_now

        # Transition matrix
        self.transition_matrix = self._build_transition_matrix(self.regime_series)

        # Conditional stats per target
        if target_series:
            self.conditional_stats = self._conditional_stats(self.regime_series, target_series)

        return self

    # ------------------------------------------------------------------------
    # Forward projections (generalizes regime_probabilities_t)
    # ------------------------------------------------------------------------
    def forecast(self, horizon: int = 5, from_state: Optional[str] = None) -> pd.DataFrame:
        """
        Project regime probabilities forward for each day from 1 to `horizon`.

        Parameters
        ----------
        horizon : int
            Number of days ahead to project (default 5 = contest week).
        from_state : str, optional
            Override the starting state. Defaults to current_state.

        Returns
        -------
        DataFrame with columns = state labels, index = days ahead (1..horizon)
        """
        if self.transition_matrix is None:
            raise RuntimeError("Must call .fit() before .forecast()")

        start = from_state or self.current_state
        states = list(self.config.state_labels)
        tm = self.transition_matrix.loc[states, states].values

        # Initial state vector
        init = np.zeros(len(states))
        init[states.index(start)] = 1.0

        rows = []
        for t in range(1, horizon + 1):
            tm_t = np.linalg.matrix_power(tm, t)
            probs = init @ tm_t
            rows.append(dict(zip(states, probs)))
        df = pd.DataFrame(rows, index=range(1, horizon + 1))
        df.index.name = 'days_ahead'
        return df

    # ------------------------------------------------------------------------
    # Stationary distribution (useful context)
    # ------------------------------------------------------------------------
    def stationary_distribution(self) -> pd.Series:
        """
        Long-run regime probabilities (eigenvector of transition matrix).
        Tells you the unconditional frequency of each state.
        """
        if self.transition_matrix is None:
            raise RuntimeError("Must call .fit() before .stationary_distribution()")
        states = list(self.config.state_labels)
        tm = self.transition_matrix.loc[states, states].values.T

        # Solve for eigenvector with eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(tm)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat = np.real(eigvecs[:, idx])
        stat = stat / stat.sum()
        return pd.Series(stat, index=states)

    # ------------------------------------------------------------------------
    # Expected returns over horizon (given regime forecast)
    # ------------------------------------------------------------------------
    def expected_path_return(self, asset: str, horizon: int = 5) -> dict:
        """
        Combine regime forecast with conditional stats to get expected
        return for an asset over the horizon.

        For each day t in 1..horizon:
            E[ret_t] = sum over states of P(state at t) * mean_ret_1d(state)
        Total path return is approximately the sum (or compounded).
        """
        if asset not in self.conditional_stats:
            raise ValueError(f"No conditional stats for {asset}. "
                             f"Fit with target_series containing this asset.")

        forecast_df = self.forecast(horizon=horizon)
        asset_stats = self.conditional_stats[asset]

        daily_expected = []
        daily_vol = []
        for t in forecast_df.index:
            ret_t = 0.0
            var_t = 0.0
            for state in forecast_df.columns:
                prob = forecast_df.loc[t, state]
                if state in asset_stats:
                    mu = asset_stats[state]['mean_ret_1d']
                    sig = asset_stats[state]['vol_1d']
                    ret_t += prob * mu
                    var_t += prob * (sig ** 2 + mu ** 2)
            var_t -= ret_t ** 2
            daily_expected.append(ret_t)
            daily_vol.append(np.sqrt(max(var_t, 0)))

        cumulative_return = np.prod([1 + r for r in daily_expected]) - 1
        # Rough path vol (assumes independence day-to-day, which isn't strictly true)
        path_vol = np.sqrt(sum(v ** 2 for v in daily_vol))

        return {
            'daily_expected_returns': daily_expected,
            'daily_vols': daily_vol,
            'cumulative_expected_return': cumulative_return,
            'path_vol': path_vol,
            'expected_sharpe_annualized': (
                (cumulative_return / path_vol) * np.sqrt(252 / horizon)
                if path_vol > 0 else 0
            ),
        }

    # ------------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------------
    def report(self) -> str:
        """Pretty-print the full regime analysis."""
        if self.transition_matrix is None:
            return "Model not fitted. Call .fit() first."

        lines = []
        lines.append("=" * 76)
        lines.append("  MARKOV REGIME MODEL — Brent/Gold Geopolitical Trading")
        lines.append("=" * 76)

        # Current state
        lines.append(f"\n [ CURRENT STATE ]")
        lines.append(f"  State:              {self.current_state.upper()}")
        lines.append(f"  Driver value:       {self.current_driver_value:.2f}")
        lines.append(f"  Driver percentile:  {self.current_driver_percentile*100:.1f}% (1y rolling)")

        # Stationary distribution (context)
        stat = self.stationary_distribution()
        lines.append(f"\n [ LONG-RUN REGIME FREQUENCIES ]")
        for state, p in stat.items():
            lines.append(f"  {state:<10}: {p*100:>5.1f}%")

        # Transition matrix
        lines.append(f"\n [ TRANSITION MATRIX ]")
        lines.append(f"  (row = from state, col = to state, values = P(next | current))")
        tm = self.transition_matrix
        header = "  from\\to   " + "  ".join(f"{s:>9}" for s in tm.columns)
        lines.append(header)
        for idx, row in tm.iterrows():
            lines.append("  " + f"{idx:<9} " + "  ".join(f"{v:>9.3f}" for v in row.values))

        # Persistence (diagonal)
        lines.append(f"\n  Persistence (P(same state tomorrow)):")
        for state in tm.columns:
            lines.append(f"    {state:<10}: {tm.loc[state, state]*100:>5.1f}%")

        # Forward forecast
        lines.append(f"\n [ FORWARD FORECAST from {self.current_state.upper()} ]")
        fc = self.forecast(horizon=5)
        lines.append(f"  day   " + "  ".join(f"{s:>9}" for s in fc.columns))
        for t, row in fc.iterrows():
            lines.append(f"  t+{t}   " + "  ".join(f"{v*100:>8.1f}%" for v in row.values))

        # Conditional stats per target
        if self.conditional_stats:
            lines.append(f"\n [ CONDITIONAL RETURNS BY REGIME ]")
            for asset, stats in self.conditional_stats.items():
                lines.append(f"\n  -- {asset} --")
                lines.append(f"  {'regime':<10}{'n':>5}{'mean_1d':>10}{'vol_1d':>9}"
                             f"{'5d_fwd':>10}{'win%':>8}{'worst':>9}{'best':>9}")
                for state in self.config.state_labels:
                    if state in stats:
                        s = stats[state]
                        lines.append(
                            f"  {state:<10}"
                            f"{s['n_obs']:>5}"
                            f"{s['mean_ret_1d']*100:>+9.2f}%"
                            f"{s['vol_1d']*100:>8.2f}%"
                            f"{s['mean_ret_5d_fwd']*100:>+9.2f}%"
                            f"{s['win_rate_5d_fwd']*100:>7.1f}%"
                            f"{s['worst_5d_fwd']*100:>+8.1f}%"
                            f"{s['best_5d_fwd']*100:>+8.1f}%"
                        )

        # Expected path returns (5 day horizon)
        if self.conditional_stats:
            lines.append(f"\n [ 5-DAY EXPECTED PATH (from current state) ]")
            for asset in self.conditional_stats:
                path = self.expected_path_return(asset, horizon=5)
                lines.append(
                    f"  {asset}: cumulative expected = "
                    f"{path['cumulative_expected_return']*100:+.2f}%, "
                    f"path vol = {path['path_vol']*100:.2f}%, "
                    f"expected Sharpe (annz) = {path['expected_sharpe_annualized']:.2f}"
                )

        lines.append("\n" + "=" * 76)
        return "\n".join(lines)


# ============================================================================
# Integration helper: Plug into the geo_dashboard pipeline
# ============================================================================

def fit_from_dashboard_data(
    loader_data: dict,
    driver_ticker: str = '^VIX',
    target_tickers: tuple = ('BZ=F', 'GC=F'),
    config: Optional[RegimeConfig] = None,
) -> RegimeModel:
    """
    Convenience: fit a RegimeModel directly from the dashboard's data dict.

    Usage:
        from geo_dashboard import GeoTradeDashboard
        from markov_regime import fit_from_dashboard_data

        d = GeoTradeDashboard().refresh()
        model = fit_from_dashboard_data(d.loader.data)
        print(model.report())
    """
    if driver_ticker not in loader_data:
        raise ValueError(f"{driver_ticker} not in loader data")

    driver = loader_data[driver_ticker]['close']
    targets = {
        t: loader_data[t]['close']
        for t in target_tickers if t in loader_data
    }
    return RegimeModel(config).fit(driver, targets)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    # Example: load data via yfinance and fit model
    import yfinance as yf

    print("Fetching VIX, Brent, Gold...")
    raw = yf.download(['^VIX', 'BZ=F', 'GC=F'], period='3y',
                      interval='1d', progress=False, auto_adjust=True)

    vix = raw['Close']['^VIX'].dropna()
    brent = raw['Close']['BZ=F'].dropna()
    gold = raw['Close']['GC=F'].dropna()

    model = RegimeModel().fit(
        driver_series=vix,
        target_series={'BZ=F': brent, 'GC=F': gold},
    )
    print(model.report())