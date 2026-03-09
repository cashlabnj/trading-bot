"""
config.py
=========
Central configuration for the modular stock trading bot.
All tuneable parameters live here - no magic numbers elsewhere.
"""

# ---------------------------------------------------------------------------
# ASSETS
# ---------------------------------------------------------------------------
ASSETS: dict = {
    "etfs":   ["SPY", "QQQ", "AGG", "GLD", "IWM"],
    "stocks": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "crypto": ["BTC-USD", "ETH-USD"],
}

# Flat list of every ticker used throughout the project
ALL_TICKERS: list = (
    ASSETS["etfs"] + ASSETS["stocks"] + ASSETS["crypto"]
)

# Reverse mapping: ticker -> asset class
TICKER_CLASS: dict = {
    ticker: cls
    for cls, tickers in ASSETS.items()
    for ticker in tickers
}

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
DATA: dict = {
    "interval":  "1d",        # yfinance interval string
    "period":    "5y",        # yfinance period string (used when start/end are None)
    "start":     None,        # ISO date string "YYYY-MM-DD" or None
    "end":       None,        # ISO date string "YYYY-MM-DD" or None
    "price_col": "Adj Close", # Column to extract from yfinance MultiIndex download
}

# ---------------------------------------------------------------------------
# STRATEGY
# ---------------------------------------------------------------------------
STRATEGY: dict = {
    # --- Signal blend weights (must sum to 1.0) ---
    "weights": {
        "momentum":       0.40,
        "mean_reversion": 0.30,
        "ma_crossover":   0.30,
    },

    # --- Momentum parameters ---
    "momentum_lookback": 90,   # calendar days of return to rank on
    "momentum_top_n":    5,    # number of top assets to score 1.0

    # --- Mean-reversion parameters ---
    "mr_zscore_window":  20,   # rolling window for z-score calculation
    "mr_entry_zscore":  -1.5,  # z-score below which asset is considered oversold
    "mr_exit_zscore":    0.5,  # z-score above which the signal clears

    # --- MA-crossover parameters ---
    "ma_short": 20,            # short moving-average window (days)
    "ma_long":  50,            # long  moving-average window (days)

    # --- Minimum combined score to act on ---
    "min_combined_score": 0.2,
}

# ---------------------------------------------------------------------------
# RISK
# ---------------------------------------------------------------------------
RISK: dict = {
    "portfolio_value":      100_000,  # baseline portfolio size ($)
    "max_position_pct":       0.10,   # max 10% of portfolio in any single ticker
    "stop_loss_pct":          0.05,   # exit if position drops 5% from entry
    "max_drawdown_halt_pct":  0.15,   # halt all trading if drawdown >= 15%
    "sector_cap_pct":         0.40,   # max 40% of portfolio in any one asset class
}

# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
BACKTEST: dict = {
    "initial_capital":  100_000,  # starting cash ($)
    "commission_pct":     0.001,  # 0.10% round-trip per trade
    "slippage_pct":      0.0005,  # 0.05% adverse fill per trade
    "rebalance_freq":      "ME",  # pandas offset alias: "ME" = month-end
}

# ---------------------------------------------------------------------------
# PAPER TRADING
# ---------------------------------------------------------------------------
PAPER: dict = {
    "initial_capital":    100_000,
    "trade_log_path":     "logs/paper_trades.csv",
    "portfolio_log_path": "logs/paper_portfolio.csv",
}

# ---------------------------------------------------------------------------
# ALERTS (email via SMTP / Gmail app-password)
# ---------------------------------------------------------------------------
ALERTS: dict = {
    "enabled":             True,
    "smtp_host":           "smtp.gmail.com",
    "smtp_port":           587,
    "sender_email":        "your_sender@gmail.com",    # <-- replace
    "sender_password":     "your_app_password",        # <-- replace (Gmail app-password)
    "recipient_email":     "your_recipient@email.com", # <-- replace
    "min_signal_to_alert": 0.3,  # only alert on tickers with score >= this
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOGGING: dict = {
    "level":    "INFO",
    "log_file": "logs/bot.log",
}
