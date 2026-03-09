"""
alerts.py
=========
Email alert module for the trading bot.

Formats signal and portfolio data into clean HTML emails and delivers
them via SMTP with TLS (default: Gmail + app-password).

Public API
----------
format_signal_email(signals, portfolio_summary) -> (subject, body)
send_email(subject, body, config_alerts)         -> bool
send_signal_alerts(signals, portfolio_summary, config_alerts)
"""

from __future__ import annotations

import logging
import smtplib
import traceback
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_CSS = """
<style>
  body       { font-family: Arial, sans-serif; font-size: 14px; color: #333; }
  h2         { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 6px; }
  h3         { color: #424242; margin-top: 20px; }
  table      { border-collapse: collapse; width: 100%; margin-top: 8px; }
  th         { background-color: #1565C0; color: white; padding: 8px 12px;
               text-align: left; font-size: 13px; }
  td         { padding: 7px 12px; border-bottom: 1px solid #e0e0e0; font-size: 13px; }
  tr:nth-child(even) td { background-color: #f5f5f5; }
  .bull      { color: #2E7D32; font-weight: bold; }
  .bear      { color: #C62828; font-weight: bold; }
  .neutral   { color: #E65100; font-weight: bold; }
  .badge-buy { background:#2E7D32; color:white; padding:2px 8px;
               border-radius:4px; font-size:12px; }
  .badge-watch { background:#1565C0; color:white; padding:2px 8px;
                 border-radius:4px; font-size:12px; }
  .summary-box { background:#E3F2FD; border-left:4px solid #1565C0;
                 padding:10px 16px; margin:12px 0; border-radius:4px; }
  .footer    { margin-top:24px; font-size:11px; color:#9E9E9E;
               border-top:1px solid #e0e0e0; padding-top:8px; }
</style>
"""


def _score_to_label(score: float) -> str:
    if score >= 0.8: return "Strong Buy"
    if score >= 0.6: return "Buy"
    if score >= 0.4: return "Watch"
    return "Weak"


def _score_to_badge(score: float) -> str:
    label = _score_to_label(score)
    css   = "badge-buy" if score >= 0.6 else "badge-watch"
    return f'<span class="{css}">{label}</span>'


def format_signal_email(
    signals: pd.Series,
    portfolio_summary: Optional[Dict] = None,
) -> Tuple[str, str]:
    """Build a subject line and HTML body for a signal alert email."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_sig   = len(signals)
    subject = f"[TradingBot] {n_sig} Signal{'s' if n_sig != 1 else ''} | {now_utc}"

    signal_rows = ""
    for ticker, score in signals.sort_values(ascending=False).items():
        bar_width = int(score * 100)
        bar_html  = (
            f'<div style="background:#e0e0e0;border-radius:3px;width:120px;display:inline-block;">'
            f'<div style="background:#1565C0;width:{bar_width}px;height:10px;border-radius:3px;"></div></div>'
        )
        signal_rows += f"""
        <tr>
          <td><strong>{ticker}</strong></td>
          <td>{score:.4f}</td>
          <td>{bar_html}</td>
          <td>{_score_to_badge(score)}</td>
        </tr>"""

    signals_table = f"""
    <h3>Trading Signals ({n_sig} tickers)</h3>
    <table>
      <thead><tr><th>Ticker</th><th>Score</th><th>Strength Bar</th><th>Action</th></tr></thead>
      <tbody>{signal_rows}
      </tbody>
    </table>"""

    portfolio_section = ""
    if portfolio_summary:
        total_val = portfolio_summary.get("total_value",    0.0)
        cash      = portfolio_summary.get("cash",           0.0)
        hold_val  = portfolio_summary.get("holdings_value", 0.0)
        upnl      = portfolio_summary.get("unrealized_pnl_total", 0.0)
        dd_pct    = portfolio_summary.get("drawdown_pct",   0.0)
        positions = portfolio_summary.get("positions",      [])
        upnl_class = "bull" if upnl >= 0 else "bear"
        dd_class   = "bull" if dd_pct <= 2 else ("neutral" if dd_pct <= 8 else "bear")
        portfolio_section = f"""
    <h3>Portfolio Snapshot</h3>
    <div class="summary-box">
      <strong>Total Value:</strong> ${total_val:,.2f} &nbsp;|&nbsp;
      <strong>Cash:</strong> ${cash:,.2f} &nbsp;|&nbsp;
      <strong>Holdings:</strong> ${hold_val:,.2f}<br/>
      <strong>Unrealized P&amp;L:</strong>
        <span class="{upnl_class}">${upnl:,.2f}</span> &nbsp;|&nbsp;
      <strong>Drawdown:</strong>
        <span class="{dd_class}">{dd_pct:.2f}%</span>
    </div>"""
        if positions:
            pos_rows = ""
            for p in positions:
                pnl_class = "bull" if p["unrealized_pnl"] >= 0 else "bear"
                pos_rows += f"""
        <tr>
          <td><strong>{p['ticker']}</strong></td>
          <td>{p['shares']:.4f}</td>
          <td>${p['entry_price']:,.2f}</td>
          <td>${p['current_price']:,.2f}</td>
          <td>${p['market_value']:,.2f}</td>
          <td class="{pnl_class}">${p['unrealized_pnl']:,.2f} ({p['pnl_pct']:.2f}%)</td>
        </tr>"""
            portfolio_section += f"""
    <table>
      <thead>
        <tr><th>Ticker</th><th>Shares</th><th>Entry</th>
            <th>Current</th><th>Mkt Value</th><th>Unrealized P&amp;L</th></tr>
      </thead>
      <tbody>{pos_rows}
      </tbody>
    </table>"""

    body = f"""<!DOCTYPE html>
<html>
<head>{_CSS}</head>
<body>
  <h2>Trading Bot Signal Alert</h2>
  <p>Generated at: <strong>{now_utc}</strong></p>
  {signals_table}
  {portfolio_section}
  <div class="footer">
    This is an automated alert from your trading bot.
    Always apply independent judgement before executing trades.
  </div>
</body>
</html>"""
    return subject, body


def send_email(subject: str, body: str, config_alerts: Dict) -> bool:
    """Send an HTML email via SMTP with STARTTLS."""
    if not config_alerts.get("enabled", True):
        logger.info("send_email: alerts disabled in config. Skipping.")
        return False

    smtp_host = config_alerts["smtp_host"]
    smtp_port = config_alerts["smtp_port"]
    sender    = config_alerts["sender_email"]
    password  = config_alerts["sender_password"]
    recipient = config_alerts["recipient_email"]

    if "your_" in sender or "your_" in password or "your_" in recipient:
        logger.warning("send_email: placeholder credentials detected. "
                       "Update ALERTS in config.py before sending real emails.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, [recipient], msg.as_string())
        logger.info("Email sent to %s | subject: %s", recipient, subject)
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed for %s.", sender)
    except smtplib.SMTPConnectError as exc:
        logger.error("SMTP connection error (%s:%d): %s", smtp_host, smtp_port, exc)
    except smtplib.SMTPException as exc:
        logger.error("SMTP error: %s", exc)
    except OSError as exc:
        logger.error("Network/OS error sending email: %s", exc)
    except Exception:
        logger.error("Unexpected error sending email:\n%s", traceback.format_exc())
    return False


def send_signal_alerts(
    signals: pd.Series,
    portfolio_summary: Optional[Dict],
    config_alerts: Dict,
) -> None:
    """Filter signals above threshold and send an alert email."""
    if not config_alerts.get("enabled", True):
        return
    min_score = config_alerts.get("min_signal_to_alert", 0.3)
    filtered  = signals[signals >= min_score]
    if filtered.empty:
        logger.info("send_signal_alerts: no signals above threshold %.2f.", min_score)
        return
    subject, body = format_signal_email(filtered, portfolio_summary)
    success       = send_email(subject, body, config_alerts)
    if success:
        logger.info("Signal alert email delivered successfully.")
    else:
        logger.warning("Signal alert email could not be delivered.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import data as data_module
    import config
    data_module.setup_logging()
    logger.info("=== alerts.py standalone run ===")

    mock_signals = pd.Series({
        "NVDA": 0.87, "AAPL": 0.72, "SPY": 0.65,
        "QQQ": 0.58, "BTC-USD": 0.41, "AGG": 0.22,
    })
    mock_portfolio = {
        "cash": 42_350.00, "holdings_value": 57_200.00, "total_value": 99_550.00,
        "unrealized_pnl_total": 1_230.50, "peak_value": 101_000.00, "drawdown_pct": 1.43,
        "positions": [
            {"ticker": "NVDA", "shares": 50.0, "entry_price": 480.00,
             "current_price": 510.00, "market_value": 25_500.00,
             "unrealized_pnl": 1_500.00, "pnl_pct": 6.25},
            {"ticker": "AAPL", "shares": 100.0, "entry_price": 175.00,
             "current_price": 177.00, "market_value": 17_700.00,
             "unrealized_pnl": 200.00, "pnl_pct": 1.14},
        ],
    }
    subject, body = format_signal_email(mock_signals, mock_portfolio)
    print("\n" + "="*60)
    print(f"Subject: {subject}")
    print("="*60)
    print("(HTML body generated -- set ALERTS config to send via email)")
