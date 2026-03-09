# Trading Bot

A modular, event-driven Python trading bot that combines three independent signal strategies with robust risk management, full historical backtesting, paper trading simulation, and automated email alerts.

---

## Features

- **Multi-Strategy Signal Engine** -- blends Momentum, Mean Reversion, and MA Crossover signals into a single weighted score per ticker
- **Risk Management** -- enforces position-size caps, sector/asset-class caps, per-position stop-losses, and a portfolio-level drawdown halt
- **Backtesting** -- full walk-forward backtest with commission + slippage simulation, performance metrics, and equity curve charting
- **Paper Trading** -- persistent simulated portfolio backed by CSV logs; state survives restarts
- **Email Alerts** -- HTML signal alert emails via SMTP/Gmail with portfolio snapshot
- **Fully Configurable** -- all parameters live in `config.py`; no magic numbers elsewhere

---

## Project Structure

```
trading-bot/
├── config.py          # All tuneable parameters
├── data.py            # Price data fetching (yfinance)
├── strategy.py        # Signal generators + blending
├── risk.py            # RiskManager class
├── backtest.py        # Backtester class + metrics + charting
├── paper_trading.py   # PaperTrader class + CSV state
├── alerts.py          # Email alert formatting + sending
├── main.py            # CLI orchestrator (3 modes)
├── requirements.txt   # Python dependencies
└── logs/              # Auto-created: trade logs, charts, bot.log
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.py` and update:
- `ASSETS` -- tickers to trade
- `ALERTS` -- your Gmail sender address, app-password, and recipient
- `RISK`, `STRATEGY`, `BACKTEST`, `PAPER` -- tune as desired

### 3. Run

```bash
# Historical backtest (5 years by default)
python main.py --mode backtest

# One paper-trading cycle
python main.py --mode paper

# Live signal scan + email alert
python main.py --mode live_alerts
```

---

## Strategies

### Momentum
Ranks tickers by total return over the last `momentum_lookback` days. The top `momentum_top_n` performers receive a score of 1.0; all others receive 0.0.

### Mean Reversion
Computes a rolling z-score for each ticker over `mr_zscore_window` days.
- z < `mr_entry_zscore` (e.g. -1.5) -> score 1.0 (oversold, buy signal)
- z > `mr_exit_zscore`  (e.g.  0.5) -> score 0.0 (overbought)
- otherwise -> score 0.5 (neutral)

### MA Crossover
Scores 1.0 (bullish) when the `ma_short`-day MA is above the `ma_long`-day MA, 0.0 otherwise.

### Combined Score
The three signals are blended using configurable weights (default: 40% momentum, 30% mean reversion, 30% MA crossover). Only tickers with a combined score >= `min_combined_score` qualify for trading.

---

## Risk Management

| Guard | Parameter | Default |
|-------|-----------|--------|
| Max position size | `max_position_pct` | 10% of portfolio |
| Max sector exposure | `sector_cap_pct` | 40% of portfolio |
| Stop-loss per position | `stop_loss_pct` | 5% below entry |
| Portfolio drawdown halt | `max_drawdown_halt_pct` | 15% from peak |

---

## Backtest Metrics

The backtester reports:
- Total Return (%)
- Annualized Return (%)
- Annual Volatility (%)
- Sharpe Ratio (risk-free = 5%)
- Max Drawdown (%)
- Calmar Ratio
- Win Rate (% of profitable days)

An equity curve chart (vs SPY benchmark) is saved to `logs/backtest_chart.png`.

---

## Email Alerts

Alerts are sent via SMTP with STARTTLS. To enable:

1. Create a Gmail App Password (Google Account -> Security -> 2-Step Verification -> App passwords)
2. Set `sender_email`, `sender_password`, and `recipient_email` in `config.ALERTS`
3. Set `enabled: True`

Alerts include a signal strength table and a portfolio snapshot with unrealized P&L.

---

## Paper Trading Logs

All paper trades and portfolio snapshots are written to CSV files defined in `config.PAPER`:
- `logs/paper_trades.csv` -- individual trade records
- `logs/paper_portfolio.csv` -- timestamped portfolio snapshots

Portfolio state is automatically restored from the CSV on restart.

---

## Configuration Reference

All parameters are in `config.py`. Key sections:

```python
ASSETS       -- tickers grouped by class (etfs, stocks, crypto)
DATA         -- yfinance interval, period, date range
STRATEGY     -- signal weights and all strategy parameters
RISK         -- position caps, stop-loss, drawdown halt
BACKTEST     -- capital, commission, slippage, rebalance frequency
PAPER        -- capital, log file paths
ALERTS       -- SMTP settings, email addresses, min signal threshold
LOGGING      -- log level and log file path
```

---

## Requirements

- Python 3.10+
- yfinance >= 0.2.40
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.11.0

---

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Always conduct your own due diligence before making any investment decisions. Past backtest performance does not guarantee future results.
