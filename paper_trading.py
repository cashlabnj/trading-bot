"""
paper_trading.py
================
Paper (simulated) trading module.

PaperTrader maintains a persistent simulated portfolio backed by CSV logs.
On each call to run_once() it:
  1. Generates signals from the latest prices
  2. Applies all risk checks
  3. Simulates order execution at last close price
  4. Appends the trade to the trade log CSV
  5. Writes the full portfolio snapshot to the portfolio log CSV

No real money is ever moved - this is purely for strategy validation.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
import data as data_module
import strategy as strategy_module
from risk import RiskManager

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulated paper-trading engine with persistent CSV-backed state."""

    def __init__(self, config_paper: Dict, config_risk: Dict, config_strategy: Dict,
                 asset_class_map: Optional[Dict[str, str]] = None) -> None:
        self.cfg_paper  = config_paper
        self.cfg_strat  = config_strategy
        self.trade_log_path     : str   = config_paper["trade_log_path"]
        self.portfolio_log_path : str   = config_paper["portfolio_log_path"]
        self.initial_capital    : float = config_paper["initial_capital"]
        _map = asset_class_map if asset_class_map is not None else config.TICKER_CLASS
        self.risk = RiskManager(config_risk, _map)
        os.makedirs(os.path.dirname(self.trade_log_path),     exist_ok=True)
        os.makedirs(os.path.dirname(self.portfolio_log_path), exist_ok=True)
        self.holdings    : Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}
        self.cash        : float            = self.initial_capital
        self.peak_value  : float            = self.initial_capital
        self._load_state()
        logger.info("PaperTrader initialised | cash=$%.2f | positions=%d",
                    self.cash, len(self.holdings))

    def _load_state(self) -> None:
        """Restore cash, holdings, and entry prices from the portfolio log."""
        if not os.path.exists(self.portfolio_log_path):
            logger.info("No existing portfolio log found. Starting fresh.")
            return
        try:
            df = pd.read_csv(self.portfolio_log_path, parse_dates=["timestamp"])
            if df.empty:
                return
            latest  = df.sort_values("timestamp").iloc[-1]
            self.cash       = float(latest.get("cash", self.initial_capital))
            self.peak_value = float(latest.get("peak_value", self.initial_capital))
            last_ts = latest["timestamp"]
            snap    = df[df["timestamp"] == last_ts]
            for _, row in snap.iterrows():
                tkr    = row.get("ticker")
                shares = float(row.get("shares", 0.0))
                ep     = float(row.get("entry_price", 0.0))
                if tkr and tkr != "CASH" and shares > 0:
                    self.holdings[tkr]     = shares
                    self.entry_prices[tkr] = ep
            logger.info("Loaded state: cash=$%.2f, %d positions (snapshot %s)",
                        self.cash, len(self.holdings), last_ts)
        except Exception as exc:
            logger.error("Failed to load portfolio state: %s", exc)

    def _save_portfolio_snapshot(self, prices: pd.Series, timestamp: datetime) -> None:
        """Append current portfolio state to the portfolio log CSV."""
        rows = [{
            "timestamp": timestamp, "ticker": "CASH", "shares": 0.0,
            "entry_price": 0.0, "current_price": 0.0, "market_value": self.cash,
            "unrealized_pnl": 0.0, "cash": self.cash, "peak_value": self.peak_value,
            "total_value": self._total_value(prices),
        }]
        for tkr, shares in self.holdings.items():
            cp  = float(prices.get(tkr, 0.0))
            ep  = self.entry_prices.get(tkr, 0.0)
            mv  = shares * cp
            pnl = (cp - ep) * shares if ep > 0 else 0.0
            rows.append({
                "timestamp": timestamp, "ticker": tkr, "shares": shares,
                "entry_price": ep, "current_price": cp, "market_value": mv,
                "unrealized_pnl": pnl, "cash": self.cash, "peak_value": self.peak_value,
                "total_value": self._total_value(prices),
            })
        snap_df = pd.DataFrame(rows)
        header  = not os.path.exists(self.portfolio_log_path)
        snap_df.to_csv(self.portfolio_log_path, mode="a", header=header, index=False)

    def _log_trade(self, trade: Dict) -> None:
        """Append a single trade record to the trade log CSV."""
        tdf    = pd.DataFrame([trade])
        header = not os.path.exists(self.trade_log_path)
        tdf.to_csv(self.trade_log_path, mode="a", header=header, index=False)

    def _total_value(self, prices: pd.Series) -> float:
        """Return total portfolio value (cash + mark-to-market holdings)."""
        return self.cash + sum(
            shares * float(prices.get(tkr, 0.0)) for tkr, shares in self.holdings.items()
        )

    def _current_allocations(self, prices: pd.Series) -> Dict[str, float]:
        return {tkr: shares * float(prices.get(tkr, 0.0)) for tkr, shares in self.holdings.items()}

    def run_once(self, prices: pd.DataFrame) -> List[Dict]:
        """Execute one paper-trading cycle."""
        if prices.empty:
            logger.warning("run_once: empty price DataFrame received.")
            return []

        now        = datetime.now(timezone.utc)
        day_prices = prices.iloc[-1]
        portfolio_value = self._total_value(day_prices)
        self.peak_value = max(self.peak_value, portfolio_value)
        executed_trades: List[Dict] = []

        if self.risk.check_max_drawdown(self.peak_value, portfolio_value):
            logger.warning("PaperTrader: drawdown halt active. No trades this cycle.")
            self._save_portfolio_snapshot(day_prices, now)
            return []

        for tkr in list(self.holdings.keys()):
            cp = float(day_prices.get(tkr, 0.0))
            ep = self.entry_prices.get(tkr, 0.0)
            if ep > 0 and cp > 0 and self.risk.check_stop_loss(tkr, ep, cp):
                shares    = self.holdings.pop(tkr)
                proceeds  = shares * cp
                self.cash += proceeds
                self.entry_prices.pop(tkr, None)
                trade = {"timestamp": now, "ticker": tkr, "action": "STOP_LOSS",
                         "shares": -shares, "price": cp, "value": -proceeds, "cash_after": self.cash}
                self._log_trade(trade)
                executed_trades.append(trade)

        signals = strategy_module.combined_signal(prices, self.cfg_strat)
        if signals.empty:
            self._save_portfolio_snapshot(day_prices, now)
            return executed_trades

        tickers_to_hold = set(signals.index)
        for tkr in list(self.holdings.keys()):
            if tkr not in tickers_to_hold:
                cp     = float(day_prices.get(tkr, 0.0))
                shares = self.holdings.pop(tkr, 0.0)
                self.entry_prices.pop(tkr, None)
                if cp <= 0 or shares <= 0:
                    continue
                proceeds  = shares * cp
                self.cash += proceeds
                trade = {"timestamp": now, "ticker": tkr, "action": "SELL",
                         "shares": -shares, "price": cp, "value": -proceeds, "cash_after": self.cash}
                self._log_trade(trade)
                executed_trades.append(trade)

        portfolio_value = self._total_value(day_prices)
        current_allocs  = self._current_allocations(day_prices)

        for tkr, score in signals.items():
            cp = float(day_prices.get(tkr, 0.0))
            if cp <= 0:
                continue
            raw_alloc   = (score / signals.sum()) * portfolio_value * 0.95
            risk_result = self.risk.apply_all_checks(
                ticker=tkr, proposed_value=raw_alloc, portfolio_value=portfolio_value,
                current_allocations=current_allocs, entry_price=self.entry_prices.get(tkr, 0.0),
                current_price=cp, peak_value=self.peak_value,
            )
            if risk_result["halt_trading"]:
                break
            if not risk_result["approved"]:
                continue

            target_value  = risk_result["capped_value"]
            target_shares = target_value / cp
            held_shares   = self.holdings.get(tkr, 0.0)
            delta         = target_shares - held_shares
            if abs(delta) < 1e-6:
                continue

            if delta > 0:
                cost = delta * cp
                if cost > self.cash:
                    delta = self.cash / cp * 0.995
                    cost  = delta * cp
                if delta < 1e-6:
                    continue
                self.cash -= cost
                self.holdings[tkr]     = held_shares + delta
                self.entry_prices[tkr] = cp
                current_allocs[tkr]    = self.holdings[tkr] * cp
                action, value = "BUY", cost
            else:
                proceeds   = abs(delta) * cp
                self.cash += proceeds
                self.holdings[tkr] = held_shares + delta
                if self.holdings[tkr] <= 1e-6:
                    self.holdings.pop(tkr, None)
                    self.entry_prices.pop(tkr, None)
                current_allocs[tkr] = self.holdings.get(tkr, 0.0) * cp
                action, value = "SELL", -proceeds

            trade = {"timestamp": now, "ticker": tkr, "action": action,
                     "shares": delta, "price": cp, "value": value, "cash_after": self.cash}
            self._log_trade(trade)
            executed_trades.append(trade)
            logger.info("%s: %s %.4f shares @ $%.4f", action, tkr, abs(delta), cp)

        self._save_portfolio_snapshot(day_prices, now)
        logger.info("run_once complete | trades=%d | portfolio=$%.2f",
                    len(executed_trades), self._total_value(day_prices))
        return executed_trades

    def get_portfolio_summary(self) -> Dict:
        """Return a snapshot dict of the current portfolio state."""
        positions, total_market_value, total_pnl = [], 0.0, 0.0
        for tkr, shares in self.holdings.items():
            ep      = self.entry_prices.get(tkr, 0.0)
            cp      = ep
            mv      = shares * cp
            pnl     = (cp - ep) * shares if ep > 0 else 0.0
            pnl_pct = (cp / ep - 1) * 100 if ep > 0 else 0.0
            positions.append({"ticker": tkr, "shares": round(shares, 6),
                               "entry_price": round(ep, 4), "current_price": round(cp, 4),
                               "market_value": round(mv, 2), "unrealized_pnl": round(pnl, 2),
                               "pnl_pct": round(pnl_pct, 2)})
            total_market_value += mv
            total_pnl          += pnl
        total_value = self.cash + total_market_value
        return {
            "cash":                 round(self.cash, 2),
            "holdings_value":       round(total_market_value, 2),
            "total_value":          round(total_value, 2),
            "unrealized_pnl_total": round(total_pnl, 2),
            "peak_value":           round(self.peak_value, 2),
            "drawdown_pct":         round(
                (self.peak_value - total_value) / self.peak_value * 100
                if self.peak_value > 0 else 0.0, 2),
            "positions": positions,
        }

    def print_summary(self) -> None:
        """Print a clean formatted portfolio summary table to stdout."""
        summary = self.get_portfolio_summary()
        sep = "-" * 75
        print(f"\n{'='*75}")
        print(f"  PAPER TRADING PORTFOLIO SUMMARY")
        print(f"{'='*75}")
        print(f"  Cash:             ${summary['cash']:>12,.2f}")
        print(f"  Holdings Value:   ${summary['holdings_value']:>12,.2f}")
        print(f"  Total Value:      ${summary['total_value']:>12,.2f}")
        print(f"  Unrealized P&L:   ${summary['unrealized_pnl_total']:>12,.2f}")
        print(f"  Peak Value:       ${summary['peak_value']:>12,.2f}")
        print(f"  Current Drawdown: {summary['drawdown_pct']:>11.2f}%")
        print(f"\n  {'Ticker':<10} {'Shares':>10} {'Entry':>10} {'Current':>10} "
              f"{'Mkt Val':>12} {'P&L':>10} {'P&L%':>8}")
        print(sep)
        if not summary["positions"]:
            print("  (no open positions)")
        else:
            for p in summary["positions"]:
                pnl_str = f"${p['unrealized_pnl']:,.2f}"
                print(f"  {p['ticker']:<10} {p['shares']:>10.4f} "
                      f"${p['entry_price']:>9.2f} ${p['current_price']:>9.2f} "
                      f"${p['market_value']:>11,.2f} {pnl_str:>10} {p['pnl_pct']:>7.2f}%")
        print(f"{'='*75}\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_module.setup_logging()
    logger.info("=== paper_trading.py standalone run ===")
    prices = data_module.fetch_latest(config.ALL_TICKERS)
    trader = PaperTrader(config_paper=config.PAPER, config_risk=config.RISK,
                         config_strategy=config.STRATEGY)
    trades = trader.run_once(prices)
    print(f"\nExecuted {len(trades)} paper trade(s) this cycle:")
    for t in trades:
        print(f"  [{t['action']}] {t['ticker']:>10}  {abs(t['shares']):>8.4f} shares "
              f"@ ${t['price']:>10.4f}  = ${abs(t['value']):>12,.2f}")
    trader.print_summary()
