"""
risk.py
=======
Risk management module.

The RiskManager class enforces four independent guardrails before any
order is placed:

1. Position-size cap        -- no single ticker exceeds max_position_pct
2. Sector / asset-class cap -- no asset class exceeds sector_cap_pct
3. Stop-loss check          -- exit if current price has fallen stop_loss_pct from entry
4. Max-drawdown halt        -- halt all trading if portfolio has drawn down
                               max_drawdown_halt_pct from its peak

All monetary values are in the same currency as portfolio_value (assumed USD).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import config

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces position, sector, stop-loss, and drawdown risk rules."""

    def __init__(self, config_risk: Dict, asset_class_map: Dict[str, str]) -> None:
        self.cfg             = config_risk
        self.asset_class_map = asset_class_map
        self.max_pos_pct    : float = config_risk["max_position_pct"]
        self.stop_loss_pct  : float = config_risk["stop_loss_pct"]
        self.max_dd_pct     : float = config_risk["max_drawdown_halt_pct"]
        self.sector_cap_pct : float = config_risk["sector_cap_pct"]

        logger.debug(
            "RiskManager initialised | max_pos=%.0f%% stop_loss=%.0f%% "
            "max_dd=%.0f%% sector_cap=%.0f%%",
            self.max_pos_pct * 100, self.stop_loss_pct * 100,
            self.max_dd_pct * 100, self.sector_cap_pct * 100,
        )

    def check_position_size(self, ticker: str, proposed_value: float,
                             portfolio_value: float) -> Tuple[bool, float]:
        """Enforce max_position_pct on a single proposed trade."""
        if portfolio_value <= 0:
            logger.error("check_position_size: portfolio_value must be > 0")
            return False, 0.0
        max_allowed = portfolio_value * self.max_pos_pct
        if proposed_value <= max_allowed:
            return True, proposed_value
        logger.info("%s position capped: $%.2f -> $%.2f (max_position_pct=%.0f%%)",
                    ticker, proposed_value, max_allowed, self.max_pos_pct * 100)
        return True, max_allowed

    def check_sector_cap(self, ticker: str, proposed_value: float,
                          current_allocations: Dict[str, float],
                          portfolio_value: float) -> Tuple[bool, float]:
        """Enforce sector_cap_pct per asset class."""
        if portfolio_value <= 0:
            return False, 0.0
        asset_class = self.asset_class_map.get(ticker, "unknown")
        max_sector  = portfolio_value * self.sector_cap_pct
        current_sector_exposure = sum(
            v for t, v in current_allocations.items()
            if self.asset_class_map.get(t, "unknown") == asset_class and t != ticker
        )
        remaining_capacity = max_sector - current_sector_exposure
        if remaining_capacity <= 0:
            logger.info("%s REJECTED: sector '%s' already at cap (exposure $%.2f >= max $%.2f)",
                        ticker, asset_class, current_sector_exposure, max_sector)
            return False, 0.0
        capped = min(proposed_value, remaining_capacity)
        if capped < proposed_value:
            logger.info("%s sector cap: $%.2f -> $%.2f (sector='%s', remaining_capacity=$%.2f)",
                        ticker, proposed_value, capped, asset_class, remaining_capacity)
        return True, capped

    def check_stop_loss(self, ticker: str, entry_price: float,
                         current_price: float) -> bool:
        """Return True if the stop-loss level has been breached."""
        if entry_price <= 0:
            logger.warning("check_stop_loss: invalid entry_price=%.4f for %s", entry_price, ticker)
            return False
        loss_pct = (entry_price - current_price) / entry_price
        if loss_pct >= self.stop_loss_pct:
            logger.warning(
                "STOP LOSS triggered for %s | entry=%.4f current=%.4f loss=%.2f%% (threshold=%.0f%%)",
                ticker, entry_price, current_price, loss_pct * 100, self.stop_loss_pct * 100
            )
            return True
        return False

    def check_max_drawdown(self, peak_value: float, current_value: float) -> bool:
        """Return True if portfolio drawdown exceeds the halt threshold."""
        if peak_value <= 0:
            return False
        drawdown = (peak_value - current_value) / peak_value
        if drawdown >= self.max_dd_pct:
            logger.critical(
                "MAX DRAWDOWN HALT: drawdown=%.2f%% >= threshold=%.0f%% (peak=$%.2f, current=$%.2f)",
                drawdown * 100, self.max_dd_pct * 100, peak_value, current_value
            )
            return True
        return False

    def apply_all_checks(self, ticker: str, proposed_value: float, portfolio_value: float,
                          current_allocations: Dict[str, float], entry_price: float,
                          current_price: float, peak_value: float) -> Dict:
        """Run all four risk checks in sequence and return a summary dict."""
        result = {"approved": False, "capped_value": 0.0, "stop_loss": False,
                  "halt_trading": False, "reason": ""}

        halt = self.check_max_drawdown(peak_value, current_value=portfolio_value)
        if halt:
            result["halt_trading"] = True
            result["reason"] = (f"Max drawdown halt: portfolio at ${portfolio_value:,.2f} "
                                f"vs peak ${peak_value:,.2f}")
            return result

        if entry_price > 0:
            sl = self.check_stop_loss(ticker, entry_price, current_price)
            if sl:
                result["stop_loss"] = True
                result["reason"] = (f"Stop-loss triggered: entry={entry_price:.4f} "
                                    f"current={current_price:.4f}")
                return result

        pos_ok, pos_value = self.check_position_size(ticker, proposed_value, portfolio_value)
        if not pos_ok:
            result["reason"] = f"Position-size check failed for {ticker}"
            return result

        sec_ok, sec_value = self.check_sector_cap(ticker, pos_value, current_allocations, portfolio_value)
        if not sec_ok:
            result["reason"] = (f"Sector cap exhausted for asset class "
                                f"'{self.asset_class_map.get(ticker, 'unknown')}'")
            return result

        result["approved"]     = True
        result["capped_value"] = sec_value
        result["reason"]       = f"Approved: ${sec_value:,.2f} (original proposal ${proposed_value:,.2f})"
        return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import data as data_module
    data_module.setup_logging()
    logger.info("=== risk.py standalone run ===")

    rm = RiskManager(config.RISK, config.TICKER_CLASS)
    portfolio_value     = 100_000.0
    peak_value          = 105_000.0
    current_allocations = {"SPY": 15_000.0, "QQQ": 12_000.0, "AAPL": 8_000.0}

    print("\n--- Position size check (AAPL, $12k proposed) ---")
    ok, val = rm.check_position_size("AAPL", 12_000, portfolio_value)
    print(f"  approved={ok}  capped_value=${val:,.2f}")

    print("\n--- Sector cap check (SPY, $18k proposed, etfs already $27k) ---")
    ok, val = rm.check_sector_cap("SPY", 18_000, current_allocations, portfolio_value)
    print(f"  approved={ok}  capped_value=${val:,.2f}")

    print("\n--- Stop-loss check (entry=150, current=140) ---")
    sl = rm.check_stop_loss("AAPL", entry_price=150.0, current_price=140.0)
    print(f"  stop_loss_triggered={sl}")

    print("\n--- Max drawdown check (peak=105k, current=87k) ---")
    halt = rm.check_max_drawdown(peak_value=105_000, current_value=87_000)
    print(f"  halt_trading={halt}")

    print("\n--- Full apply_all_checks (NVDA, $10k proposed) ---")
    result = rm.apply_all_checks(
        ticker="NVDA", proposed_value=10_000, portfolio_value=portfolio_value,
        current_allocations=current_allocations, entry_price=0.0,
        current_price=500.0, peak_value=peak_value,
    )
    for k, v in result.items():
        print(f"  {k}: {v}")
