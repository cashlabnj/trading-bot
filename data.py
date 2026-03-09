"""
data.py
=======
Data fetching and preprocessing module using yfinance.
Provides price DataFrames and return series consumed by strategy, backtest,
and paper-trading modules.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

import pandas as pd
import yfinance as yf

import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure root logger from config.LOGGING and return a module logger."""
    os.makedirs(os.path.dirname(config.LOGGING["log_file"]), exist_ok=True)

    level = getattr(logging, config.LOGGING["level"].upper(), logging.INFO)
    fmt   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(config.LOGGING["log_file"]))
    except OSError as exc:
        print(f"[WARNING] Could not open log file: {exc}")

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core fetch functions
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers:  List[str],
    period:   Optional[str] = None,
    interval: str = "1d",
    start:    Optional[str] = None,
    end:      Optional[str] = None,
) -> pd.DataFrame:
    """Download adjusted-close prices for *tickers* via yfinance.

    Parameters
    ----------
    tickers  : list of ticker symbols
    period   : yfinance period string (e.g. "5y"); used when start/end are None
    interval : bar interval string (default "1d")
    start    : ISO date "YYYY-MM-DD" (overrides period when provided)
    end      : ISO date "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame  columns = tickers, index = Date (DatetimeIndex)
    """
    if not tickers:
        raise ValueError("fetch_prices: tickers list is empty.")

    kwargs: dict = {"tickers": tickers, "interval": interval,
                    "auto_adjust": True, "progress": False}
    if start:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end
        logger.info("Fetching %d tickers | start=%s end=%s interval=%s",
                    len(tickers), start, end, interval)
    else:
        kwargs["period"] = period or config.DATA["period"]
        logger.info("Fetching %d tickers | period=%s interval=%s",
                    len(tickers), kwargs["period"], interval)

    try:
        raw = yf.download(**kwargs, group_by="ticker")
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        raise

    # --- Extract close prices from potentially MultiIndex DataFrame ---
    # yfinance >= 0.2.31 swapped MultiIndex order from (field, ticker)
    # to (ticker, field). We detect which schema is in use and handle both.
    price_col = "Close"  # auto_adjust=True renames Adj Close -> Close

    if isinstance(raw.columns, pd.MultiIndex):
        level_0 = raw.columns.get_level_values(0).unique().tolist()
        level_1 = raw.columns.get_level_values(1).unique().tolist()

        if price_col in level_0:
            # Old schema: (field, ticker)
            prices = raw[price_col]
        elif price_col in level_1:
            # New schema: (ticker, field)
            prices = raw.xs(price_col, axis=1, level=1)
        else:
            # Fallback: use first available field
            logger.warning(
                "Column '%s' not found. level_0=%s  level_1=%s",
                price_col, level_0, level_1,
            )
            fallback = level_1[0] if level_1 else level_0[0]
            prices = raw.xs(fallback, axis=1, level=1)
    else:
        # Single ticker or flat columns
        if price_col in raw.columns:
            prices = raw[[price_col]].rename(columns={price_col: tickers[0]})
        else:
            close_cols = [c for c in raw.columns if "close" in str(c).lower()]
            col = close_cols[0] if close_cols else raw.columns[0]
            logger.warning("Expected '%s', using '%s' instead.", price_col, col)
            prices = raw[[col]].rename(columns={col: tickers[0]})

    # Ensure columns are plain strings
    prices.columns = [str(c) for c in prices.columns]

    # Drop tickers that are entirely NaN (e.g. bad symbols)
    before = prices.shape[1]
    prices.dropna(axis=1, how="all", inplace=True)
    dropped = before - prices.shape[1]
    if dropped:
        logger.warning("Dropped %d all-NaN tickers after download.", dropped)

    # Forward-fill small gaps (weekends / holidays already handled by yf,
    # but crypto may have isolated NaNs mid-series)
    prices.ffill(inplace=True)
    prices.dropna(how="all", inplace=True)

    logger.info("Price DataFrame shape: %s  date range: %s -> %s",
                prices.shape,
                prices.index[0].date() if len(prices) else "N/A",
                prices.index[-1].date() if len(prices) else "N/A")
    return prices


def fetch_latest(tickers: List[str]) -> pd.DataFrame:
    """Fetch the last 100 calendar days of daily prices.

    Convenience wrapper used by paper-trading and live-alert modules
    where only recent data is needed.

    Parameters
    ----------
    tickers : list of ticker symbols

    Returns
    -------
    pd.DataFrame  same shape as fetch_prices()
    """
    logger.info("fetch_latest: pulling 100-day window for %d tickers", len(tickers))
    return fetch_prices(tickers, period="100d", interval="1d")


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from a prices DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame  price series (columns = tickers)

    Returns
    -------
    pd.DataFrame  of daily pct_change values; first row is NaN and is dropped
    """
    returns = prices.pct_change()
    returns.dropna(how="all", inplace=True)
    logger.debug("Returns shape: %s", returns.shape)
    return returns


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    logger.info("=== data.py standalone run ===")

    prices = fetch_prices(
        tickers  = config.ALL_TICKERS,
        period   = config.DATA["period"],
        interval = config.DATA["interval"],
        start    = config.DATA["start"],
        end      = config.DATA["end"],
    )
    print("\nPrice head:")
    print(prices.head())
    print("\nPrice tail:")
    print(prices.tail())

    returns = compute_returns(prices)
    print("\nReturns head:")
    print(returns.head())
