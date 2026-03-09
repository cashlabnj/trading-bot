"""
strategy.py
===========
Multi-strategy signal generation module.

Three independent signal generators are blended into a single combined
score per ticker using configurable weights defined in config.STRATEGY.

Signal scores are always in [0.0, 1.0]:
  0.0  -> no interest / bearish
  0.5  -> neutral / hold
  1.0  -> strong interest / bullish
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

import config
import data as data_module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual signal generators
# ---------------------------------------------------------------------------

def momentum_signal(
    prices:   pd.DataFrame,
    lookback: int,
    top_n:    int,
) -> pd.Series:
    """Rank-based momentum signal.

    Computes total return over the last *lookback* trading days for each
    ticker, then scores the top *top_n* performers as 1.0 and all others
    as 0.0.

    Parameters
    ----------
    prices   : price DataFrame (rows = dates, cols = tickers)
    lookback : number of trading days to measure return over
    top_n    : how many tickers receive a score of 1.0

    Returns
    -------
    pd.Series  index = tickers, values in {0.0, 1.0}
    """
    if len(prices) < lookback + 1:
        logger.warning(
            "momentum_signal: only %d rows available, need %d. "
            "Returning zero signal.", len(prices), lookback + 1
        )
        return pd.Series(0.0, index=prices.columns)

    window = prices.iloc[-lookback - 1:]
    returns = (window.iloc[-1] - window.iloc[0]) / window.iloc[0].replace(0, np.nan)
    returns.fillna(0.0, inplace=True)

    # Rank descending; ties broken by first occurrence
    ranked = returns.rank(ascending=False, method="first")
    signal = (ranked <= top_n).astype(float)

    logger.debug("Momentum signal (top %d over %d days):\n%s", top_n, lookback, signal)
    return signal


def mean_reversion_signal(
    prices:       pd.DataFrame,
    window:       int,
    entry_zscore: float,
    exit_zscore:  float,
) -> pd.Series:
    """Z-score mean-reversion signal.

    For each ticker, computes the z-score of the latest price relative to a
    rolling mean/std over *window* days:
        z = (price - rolling_mean) / rolling_std

    Scoring logic:
        z < entry_zscore  -> 1.0  (oversold, buy signal)
        z > exit_zscore   -> 0.0  (overbought, no position)
        otherwise         -> 0.5  (neutral, hold if already in)

    Parameters
    ----------
    prices       : price DataFrame
    window       : rolling window for mean and std
    entry_zscore : threshold below which asset is considered oversold
    exit_zscore  : threshold above which the signal is cleared

    Returns
    -------
    pd.Series  index = tickers, values in {0.0, 0.5, 1.0}
    """
    if len(prices) < window:
        logger.warning(
            "mean_reversion_signal: only %d rows, need %d. "
            "Returning neutral signal.", len(prices), window
        )
        return pd.Series(0.5, index=prices.columns)

    rolling_mean = prices.rolling(window=window).mean()
    rolling_std  = prices.rolling(window=window).std()

    # Avoid division by zero for zero-std tickers
    rolling_std.replace(0.0, np.nan, inplace=True)

    z_scores = (prices - rolling_mean) / rolling_std
    latest_z = z_scores.iloc[-1]

    signal = latest_z.apply(
        lambda z: 1.0 if z < entry_zscore
                  else (0.0 if z > exit_zscore else 0.5)
    )
    # NaN z-scores (insufficient data) -> neutral
    signal.fillna(0.5, inplace=True)

    logger.debug(
        "MeanReversion z-scores (window=%d, entry=%.2f, exit=%.2f):\n%s",
        window, entry_zscore, exit_zscore, latest_z.round(3)
    )
    return signal


def ma_crossover_signal(
    prices:       pd.DataFrame,
    short_window: int,
    long_window:  int,
) -> pd.Series:
    """Moving-average crossover signal.

    Scores 1.0 (bullish) when the short MA is above the long MA,
    0.0 (bearish) otherwise.

    Parameters
    ----------
    prices       : price DataFrame
    short_window : fast MA period in trading days
    long_window  : slow MA period in trading days

    Returns
    -------
    pd.Series  index = tickers, values in {0.0, 1.0}
    """
    if len(prices) < long_window:
        logger.warning(
            "ma_crossover_signal: only %d rows, need %d. "
            "Returning zero signal.", len(prices), long_window
        )
        return pd.Series(0.0, index=prices.columns)

    short_ma = prices.rolling(window=short_window).mean().iloc[-1]
    long_ma  = prices.rolling(window=long_window).mean().iloc[-1]

    signal = (short_ma > long_ma).astype(float)
    signal.fillna(0.0, inplace=True)

    logger.debug(
        "MA Crossover signal (short=%d, long=%d):\n%s",
        short_window, long_window, signal
    )
    return signal


# ---------------------------------------------------------------------------
# Combined signal
# ---------------------------------------------------------------------------

def combined_signal(
    prices:          pd.DataFrame,
    config_strategy: Dict,
) -> pd.Series:
    """Blend the three individual signals into a single weighted score.

    Uses weights from config_strategy["weights"] and filters out any ticker
    whose combined score falls below config_strategy["min_combined_score"].

    Parameters
    ----------
    prices          : price DataFrame
    config_strategy : dict matching config.STRATEGY structure

    Returns
    -------
    pd.Series  index = qualifying tickers, values in [0.0, 1.0],
               sorted descending by score.
    """
    weights = config_strategy["weights"]

    mom_sig = momentum_signal(
        prices,
        lookback = config_strategy["momentum_lookback"],
        top_n    = config_strategy["momentum_top_n"],
    )

    mr_sig = mean_reversion_signal(
        prices,
        window       = config_strategy["mr_zscore_window"],
        entry_zscore = config_strategy["mr_entry_zscore"],
        exit_zscore  = config_strategy["mr_exit_zscore"],
    )

    ma_sig = ma_crossover_signal(
        prices,
        short_window = config_strategy["ma_short"],
        long_window  = config_strategy["ma_long"],
    )

    # Align all signals to the same index (tickers present in all three)
    common_tickers = mom_sig.index.intersection(mr_sig.index).intersection(ma_sig.index)
    mom_sig = mom_sig.reindex(common_tickers).fillna(0.0)
    mr_sig  = mr_sig.reindex(common_tickers).fillna(0.5)
    ma_sig  = ma_sig.reindex(common_tickers).fillna(0.0)

    blended = (
        weights["momentum"]       * mom_sig
        + weights["mean_reversion"] * mr_sig
        + weights["ma_crossover"]   * ma_sig
    )

    # Filter by minimum threshold
    min_score = config_strategy["min_combined_score"]
    qualified = blended[blended >= min_score].sort_values(ascending=False)

    logger.info(
        "Combined signal: %d/%d tickers qualify (min_score=%.2f)",
        len(qualified), len(blended), min_score
    )
    logger.debug("Qualified signals:\n%s", qualified.round(4))

    return qualified


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_module.setup_logging()
    logger.info("=== strategy.py standalone run ===")

    prices = data_module.fetch_prices(
        tickers  = config.ALL_TICKERS,
        period   = config.DATA["period"],
        interval = config.DATA["interval"],
    )

    print("\n--- Momentum Signal ---")
    print(momentum_signal(prices, config.STRATEGY["momentum_lookback"],
                          config.STRATEGY["momentum_top_n"]))

    print("\n--- Mean Reversion Signal ---")
    print(mean_reversion_signal(prices, config.STRATEGY["mr_zscore_window"],
                                config.STRATEGY["mr_entry_zscore"],
                                config.STRATEGY["mr_exit_zscore"]))

    print("\n--- MA Crossover Signal ---")
    print(ma_crossover_signal(prices, config.STRATEGY["ma_short"],
                              config.STRATEGY["ma_long"]))

    print("\n--- Combined Signal ---")
    print(combined_signal(prices, config.STRATEGY))
