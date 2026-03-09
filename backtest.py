"""
backtest.py
===========
Event-driven backtesting engine.

The Backtester class resamples a price DataFrame to the configured
rebalance frequency, runs the combined signal at each period boundary,
applies all risk checks, sizes positions using equal-weight allocation
among qualifying tickers, and tracks the full portfolio time-series.

At the end it computes standard performance metrics and optionally
plots the equity curve vs a benchmark.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import config
import data as data_module
import strategy as strategy_module
from risk import RiskManager

logger = logging.getLogger(__name__)


class Backtester:
    """Full walk-forward backtesting engine."""

    def __init__(self, config_backtest: Dict, config_risk: Dict, config_strategy: Dict,
                 asset_class_map: Optional[Dict[str, str]] = None) -> None:
        self.cfg_bt     = config_backtest
        self.cfg_strat  = config_strategy
        self.initial_capital : float = config_backtest["initial_capital"]
        self.commission_pct  : float = config_backtest["commission_pct"]
        self.slippage_pct    : float = config_backtest["slippage_pct"]
        self.rebalance_freq  : str   = config_backtest["rebalance_freq"]
        _map = asset_class_map if asset_class_map is not None else config.TICKER_CLASS
        self.risk = RiskManager(config_risk, _map)
        logger.info(
            "Backtester ready | capital=$%.0f commission=%.2f%% slippage=%.2f%% rebalance=%s",
            self.initial_capital, self.commission_pct * 100,
            self.slippage_pct * 100, self.rebalance_freq,
        )

    def _apply_costs(self, trade_value: float) -> float:
        """Return the net cash after commissions and slippage."""
        return abs(trade_value) * (self.commission_pct + self.slippage_pct)

    def run(self, prices: pd.DataFrame) -> Dict:
        """Execute the walk-forward backtest."""
        prices = prices.copy()
        prices.sort_index(inplace=True)

        rebalance_dates = prices.resample(self.rebalance_freq).last().index
        rebalance_dates = rebalance_dates[rebalance_dates >= prices.index[0]]
        logger.info("Backtest: %d rebalance dates | %s -> %s",
                    len(rebalance_dates), rebalance_dates[0].date(), rebalance_dates[-1].date())

        cash        : float            = self.initial_capital
        holdings    : Dict[str, float] = {}
        entry_prices: Dict[str, float] = {}
        peak_value  : float            = self.initial_capital
        trades      : List[Dict]       = []
        portfolio_values: Dict[pd.Timestamp, float] = {}

        all_dates = prices.index.tolist()
        rebal_set = set(rebalance_dates)

        for date in all_dates:
            day_prices = prices.loc[date]
            holdings_value = sum(shares * day_prices.get(tkr, 0.0) for tkr, shares in holdings.items())
            portfolio_value = cash + holdings_value
            portfolio_values[date] = portfolio_value
            peak_value = max(peak_value, portfolio_value)

            if self.risk.check_max_drawdown(peak_value, portfolio_value):
                logger.warning("Drawdown halt on %s -- no trading.", date.date())
                continue

            to_exit: List[str] = []
            for tkr, shares in list(holdings.items()):
                ep = entry_prices.get(tkr, 0.0)
                cp = day_prices.get(tkr, 0.0)
                if ep > 0 and cp > 0 and self.risk.check_stop_loss(tkr, ep, cp):
                    to_exit.append(tkr)

            for tkr in to_exit:
                shares   = holdings.pop(tkr, 0.0)
                ep       = entry_prices.pop(tkr, 0.0)
                cp       = day_prices.get(tkr, 0.0)
                proceeds = shares * cp
                cost     = self._apply_costs(proceeds)
                cash    += proceeds - cost
                trades.append({"date": date, "ticker": tkr, "action": "STOP_LOSS",
                               "shares": -shares, "price": cp, "value": -proceeds})

            if date not in rebal_set:
                continue

            hist_prices = prices.loc[:date]
            signals = strategy_module.combined_signal(hist_prices, self.cfg_strat)
            if signals.empty:
                continue

            tickers_to_hold = set(signals.index)
            for tkr in list(holdings.keys()):
                if tkr not in tickers_to_hold:
                    shares   = holdings.pop(tkr, 0.0)
                    entry_prices.pop(tkr, None)
                    cp       = day_prices.get(tkr, 0.0)
                    if cp <= 0:
                        continue
                    proceeds = shares * cp
                    cost     = self._apply_costs(proceeds)
                    cash    += proceeds - cost
                    trades.append({"date": date, "ticker": tkr, "action": "SELL",
                                   "shares": -shares, "price": cp, "value": -proceeds})

            holdings_value  = sum(shares * day_prices.get(t, 0.0) for t, shares in holdings.items())
            portfolio_value = cash + holdings_value
            current_allocs  = {t: s * day_prices.get(t, 0.0) for t, s in holdings.items()}

            for tkr, score in signals.items():
                cp = day_prices.get(tkr, 0.0)
                if cp <= 0:
                    continue
                raw_alloc = (score / signals.sum()) * portfolio_value * 0.95
                risk_result = self.risk.apply_all_checks(
                    ticker=tkr, proposed_value=raw_alloc, portfolio_value=portfolio_value,
                    current_allocations=current_allocs, entry_price=entry_prices.get(tkr, 0.0),
                    current_price=cp, peak_value=peak_value,
                )
                if risk_result["halt_trading"]:
                    break
                if not risk_result["approved"]:
                    continue

                target_value   = risk_result["capped_value"]
                target_shares  = target_value / cp
                current_shares = holdings.get(tkr, 0.0)
                delta_shares   = target_shares - current_shares

                if abs(delta_shares) < 1e-6:
                    continue

                trade_value = abs(delta_shares * cp)
                cost        = self._apply_costs(trade_value)

                if delta_shares > 0:
                    total_cost = delta_shares * cp + cost
                    if total_cost > cash:
                        affordable = max(0, cash - cost) / cp
                        if affordable < 1e-6:
                            continue
                        delta_shares = affordable
                        total_cost   = delta_shares * cp + self._apply_costs(delta_shares * cp)
                    cash -= total_cost
                    holdings[tkr]      = holdings.get(tkr, 0.0) + delta_shares
                    entry_prices[tkr]  = cp
                    current_allocs[tkr] = holdings[tkr] * cp
                    trades.append({"date": date, "ticker": tkr, "action": "BUY",
                                   "shares": delta_shares, "price": cp, "value": delta_shares * cp})
                else:
                    proceeds = abs(delta_shares) * cp - cost
                    cash    += proceeds
                    holdings[tkr] = holdings.get(tkr, 0.0) + delta_shares
                    if holdings[tkr] <= 1e-6:
                        holdings.pop(tkr, None)
                        entry_prices.pop(tkr, None)
                    current_allocs[tkr] = holdings.get(tkr, 0.0) * cp
                    trades.append({"date": date, "ticker": tkr, "action": "SELL",
                                   "shares": delta_shares, "price": cp, "value": delta_shares * cp})

        pv_series = pd.Series(portfolio_values).sort_index()
        pv_series.index.name = "date"
        metrics = self.compute_metrics(pv_series)
        logger.info("Backtest complete | trades=%d | final_value=$%.2f", len(trades), pv_series.iloc[-1])
        return {"portfolio_values": pv_series, "trades": trades, "metrics": metrics}

    def compute_metrics(self, portfolio_values: pd.Series) -> Dict:
        """Compute standard performance metrics from a NAV series."""
        pv = portfolio_values.dropna()
        if len(pv) < 2:
            return {}
        daily_returns = pv.pct_change().dropna()
        total_return  = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
        n_years       = len(pv) / 252
        ann_return    = ((pv.iloc[-1] / pv.iloc[0]) ** (1 / max(n_years, 0.001)) - 1) * 100
        volatility    = daily_returns.std() * np.sqrt(252) * 100
        risk_free     = 0.05
        rf_daily      = risk_free / 252
        excess_ret    = daily_returns - rf_daily
        sharpe        = (excess_ret.mean() / excess_ret.std() * np.sqrt(252)
                         if excess_ret.std() > 0 else 0.0)
        rolling_max   = pv.cummax()
        drawdown      = (pv - rolling_max) / rolling_max
        max_dd        = drawdown.min() * 100
        calmar        = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0
        win_rate      = (daily_returns > 0).sum() / len(daily_returns) * 100
        return {
            "total_return":      round(total_return,  2),
            "annualized_return": round(ann_return,    2),
            "volatility":        round(volatility,    2),
            "sharpe_ratio":      round(sharpe,        4),
            "max_drawdown":      round(max_dd,        2),
            "calmar_ratio":      round(calmar,        4),
            "win_rate":          round(win_rate,      2),
        }

    def plot_results(self, portfolio_values: pd.Series,
                     benchmark_prices: Optional[pd.DataFrame] = None,
                     benchmark_ticker: str = "SPY",
                     output_path: str = "logs/backtest_chart.png") -> None:
        """Plot portfolio equity curve vs an optional benchmark."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle("Trading Bot Backtest Results", fontsize=14, fontweight="bold")

        ax1 = axes[0]
        norm_pv = portfolio_values / portfolio_values.iloc[0] * 100
        ax1.plot(norm_pv.index, norm_pv.values, color="#2196F3", linewidth=1.8, label="Strategy")
        if benchmark_prices is not None and benchmark_ticker in benchmark_prices.columns:
            bm = benchmark_prices[benchmark_ticker].reindex(portfolio_values.index).ffill()
            bm_norm = bm / bm.iloc[0] * 100
            ax1.plot(bm_norm.index, bm_norm.values, color="#FF9800",
                     linewidth=1.2, linestyle="--", label=benchmark_ticker)
        ax1.axhline(100, color="grey", linewidth=0.8, linestyle=":")
        ax1.set_ylabel("Indexed Value (base=100)", fontsize=10)
        ax1.set_title("Portfolio Equity Curve", fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

        ax2 = axes[1]
        rolling_max = portfolio_values.cummax()
        drawdown    = (portfolio_values - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.6, label="Drawdown %")
        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.set_title("Portfolio Drawdown", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Backtest chart saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_module.setup_logging()
    logger.info("=== backtest.py standalone run ===")

    prices = data_module.fetch_prices(
        tickers=config.ALL_TICKERS, period=config.DATA["period"],
        interval=config.DATA["interval"], start=config.DATA["start"], end=config.DATA["end"],
    )
    bt = Backtester(config_backtest=config.BACKTEST, config_risk=config.RISK,
                    config_strategy=config.STRATEGY)
    results = bt.run(prices)

    print("\n" + "="*50)
    print("  BACKTEST METRICS")
    print("="*50)
    for metric, value in results["metrics"].items():
        label = metric.replace("_", " ").title()
        unit  = "%" if any(k in metric for k in ["return", "drawdown", "volatility", "rate"]) else ""
        print(f"  {label:<25} {value:>10.2f}{unit}")
    print("="*50)
    print(f"  Total trades: {len(results['trades'])}")
    print(f"  Final value:  ${results['portfolio_values'].iloc[-1]:,.2f}")
    bt.plot_results(results["portfolio_values"], benchmark_prices=prices,
                    benchmark_ticker="SPY", output_path="logs/backtest_chart.png")
    print("\nChart saved to logs/backtest_chart.png")
