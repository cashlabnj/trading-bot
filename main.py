"""
main.py
=======
Main orchestrator for the modular stock trading bot.

Three operating modes, selected via --mode flag:

  backtest     -- Full historical backtest on 5-year daily data
  paper        -- One paper-trading cycle on latest 100-day prices
  live_alerts  -- Fetch latest prices, generate signals, send email alerts

Usage
-----
  python main.py --mode backtest
  python main.py --mode paper
  python main.py --mode live_alerts
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional


def setup_logging() -> None:
    """Configure root logger from config.LOGGING; create logs/ dir if needed."""
    import config
    log_file = config.LOGGING.get("log_file", "logs/bot.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    level   = getattr(logging, config.LOGGING.get("level", "INFO").upper(), logging.INFO)
    fmt     = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except OSError as exc:
        print(f"[WARNING] Could not open log file {log_file!r}: {exc}", file=sys.stderr)
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)


def _import_modules():
    import config
    import data         as data_mod
    import strategy     as strategy_mod
    from backtest       import Backtester
    from paper_trading  import PaperTrader
    import alerts       as alerts_mod
    return dict(config=config, data=data_mod, strategy=strategy_mod,
                Backtester=Backtester, PaperTrader=PaperTrader, alerts=alerts_mod)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trading_bot",
        description="Modular multi-strategy stock trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode backtest      # historical backtest
  python main.py --mode paper         # one paper-trading cycle
  python main.py --mode live_alerts   # signal scan + email alert
        """,
    )
    parser.add_argument("--mode", choices=["backtest", "paper", "live_alerts"],
                        default="backtest", help="Operating mode (default: backtest)")
    return parser.parse_args()


def run_backtest() -> None:
    """Fetch 5-year daily prices, run full backtest, print metrics, save chart."""
    mods = _import_modules()
    cfg  = mods["config"]
    data = mods["data"]
    logger.info("========== MODE: BACKTEST ==========")

    try:
        prices = data.fetch_prices(
            tickers=cfg.ALL_TICKERS, period=cfg.DATA["period"],
            interval=cfg.DATA["interval"], start=cfg.DATA["start"], end=cfg.DATA["end"],
        )
    except Exception as exc:
        logger.critical("Failed to fetch price data: %s", exc)
        sys.exit(1)

    if prices.empty:
        logger.critical("Price DataFrame is empty. Check tickers and network.")
        sys.exit(1)

    bt = mods["Backtester"](config_backtest=cfg.BACKTEST, config_risk=cfg.RISK,
                             config_strategy=cfg.STRATEGY)
    logger.info("Running backtest on %d tickers, %d rows...", len(prices.columns), len(prices))

    try:
        results = bt.run(prices)
    except Exception as exc:
        logger.critical("Backtest failed: %s", exc, exc_info=True)
        sys.exit(1)

    metrics = results["metrics"]
    trades  = results["trades"]
    pv      = results["portfolio_values"]

    print("\n" + "=" * 54)
    print("  BACKTEST RESULTS")
    print("=" * 54)
    metric_labels = {
        "total_return":      ("Total Return",        "%"),
        "annualized_return": ("Annualized Return",   "%"),
        "volatility":        ("Annual Volatility",   "%"),
        "sharpe_ratio":      ("Sharpe Ratio",         ""),
        "max_drawdown":      ("Max Drawdown",        "%"),
        "calmar_ratio":      ("Calmar Ratio",         ""),
        "win_rate":          ("Win Rate (daily)",    "%"),
    }
    for key, (label, unit) in metric_labels.items():
        val = metrics.get(key, float("nan"))
        print(f"  {label:<25} {val:>10.2f}{unit}")
    print("-" * 54)
    print(f"  Total Trades:             {len(trades):>10}")
    print(f"  Start Value:              ${pv.iloc[0]:>10,.2f}")
    print(f"  End Value:                ${pv.iloc[-1]:>10,.2f}")
    print(f"  Period:  {pv.index[0].date()} -> {pv.index[-1].date()}")
    print("=" * 54)

    chart_path = "logs/backtest_chart.png"
    try:
        bt.plot_results(portfolio_values=pv, benchmark_prices=prices,
                        benchmark_ticker="SPY", output_path=chart_path)
        print(f"\n  Chart saved -> {chart_path}")
    except Exception as exc:
        logger.warning("Could not save chart: %s", exc)


def run_paper() -> None:
    """Fetch latest 100-day prices and run one paper-trading cycle."""
    mods = _import_modules()
    cfg  = mods["config"]
    data = mods["data"]
    logger.info("========== MODE: PAPER TRADING ==========")

    try:
        prices = data.fetch_latest(cfg.ALL_TICKERS)
    except Exception as exc:
        logger.critical("Failed to fetch prices: %s", exc)
        sys.exit(1)

    trader = mods["PaperTrader"](config_paper=cfg.PAPER, config_risk=cfg.RISK,
                                  config_strategy=cfg.STRATEGY)
    trades = trader.run_once(prices)

    print(f"\n  Executed {len(trades)} trade(s) this cycle:")
    if trades:
        header = f"  {'Action':<12} {'Ticker':<10} {'Shares':>10} {'Price':>12} {'Value':>14}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for t in trades:
            print(f"  {t['action']:<12} {t['ticker']:<10} {abs(t['shares']):>10.4f} "
                  f"${t['price']:>11.4f} ${abs(t['value']):>13,.2f}")
    trader.print_summary()


def run_live_alerts() -> None:
    """Fetch latest prices, generate signals, print them, and send email alerts."""
    mods = _import_modules()
    cfg  = mods["config"]
    data = mods["data"]
    logger.info("========== MODE: LIVE ALERTS ==========")

    try:
        prices = data.fetch_latest(cfg.ALL_TICKERS)
    except Exception as exc:
        logger.critical("Failed to fetch prices: %s", exc)
        sys.exit(1)

    signals = mods["strategy"].combined_signal(prices, cfg.STRATEGY)

    print("\n" + "=" * 46)
    print("  LIVE SIGNAL SCAN RESULTS")
    print("=" * 46)
    if signals.empty:
        print("  No qualifying signals found.")
    else:
        print(f"  {'Ticker':<12} {'Score':>8}   {'Strength':<14}")
        print("  " + "-" * 40)
        for ticker, score in signals.items():
            from alerts import _score_to_label
            label = _score_to_label(score)
            print(f"  {ticker:<12} {score:>8.4f}   {label:<14}")
    print("=" * 46)

    mods["alerts"].send_signal_alerts(
        signals=signals,
        portfolio_summary=None,
        config_alerts=cfg.ALERTS,
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    logger.info("Trading bot starting | mode=%s", args.mode)

    dispatch = {
        "backtest":    run_backtest,
        "paper":       run_paper,
        "live_alerts": run_live_alerts,
    }
    try:
        dispatch[args.mode]()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.critical("Unhandled exception in mode '%s': %s", args.mode, exc, exc_info=True)
        sys.exit(1)

    logger.info("Trading bot finished | mode=%s", args.mode)


if __name__ == "__main__":
    main()
