"""
Alert dispatcher — Slack webhook + optional email.
All alert calls are fire-and-forget; failures are logged but never raise.
"""
import json
import smtplib
import threading
from email.mime.text import MIMEText
from typing import Optional

import requests

import config
from utils.logger import log


def _send_slack(message: str) -> None:
    if not config.SLACK_WEBHOOK_URL:
        return
    try:
        payload = {"text": message, "username": "24/7 TradeBot", "icon_emoji": ":robot_face:"}
        resp = requests.post(
            config.SLACK_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if resp.status_code != 200:
            log.warning("Slack alert failed: {} {}", resp.status_code, resp.text)
    except Exception as e:
        log.warning("Slack alert error: {}", e)


def _send_email(subject: str, body: str) -> None:
    if not config.ALERT_EMAIL:
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = config.ALERT_EMAIL
        msg["To"] = config.ALERT_EMAIL
        with smtplib.SMTP("localhost", timeout=5) as s:
            s.sendmail(config.ALERT_EMAIL, [config.ALERT_EMAIL], msg.as_string())
    except Exception as e:
        log.warning("Email alert error: {}", e)


def _dispatch(message: str, subject: Optional[str] = None) -> None:
    threading.Thread(target=_send_slack, args=(message,), daemon=True).start()
    if subject:
        threading.Thread(target=_send_email, args=(subject, message), daemon=True).start()


def alert_trade(symbol: str, side: str, qty: float, price: float, strategy: str) -> None:
    if not config.ALERT_ON_TRADE:
        return
    msg = (
        f"*TRADE* `{side.upper()} {qty} {symbol}` @ ${price:.4f} "
        f"| strategy={strategy} | notional=${qty*price:,.0f}"
    )
    log.info(msg)
    _dispatch(msg)


def alert_circuit_break(reason: str, halt_type: str) -> None:
    if not config.ALERT_ON_CIRCUIT:
        return
    msg = f":rotating_light: *CIRCUIT BREAKER* `{halt_type}` — {reason}"
    log.warning(msg)
    _dispatch(msg, subject=f"[TradeBot] Circuit Breaker: {halt_type}")


def alert_error(error: str, context: str = "") -> None:
    if not config.ALERT_ON_ERROR:
        return
    msg = f":x: *ERROR* {context} — {error}"
    log.error(msg)
    _dispatch(msg, subject=f"[TradeBot] Error: {context}")


def alert_daily_summary(portfolio_value: float, daily_pnl: float, open_positions: int) -> None:
    pct = daily_pnl / (portfolio_value - daily_pnl) * 100 if portfolio_value != daily_pnl else 0
    sign = "+" if daily_pnl >= 0 else ""
    msg = (
        f":bar_chart: *Daily Summary* | "
        f"Portfolio=${portfolio_value:,.2f} | "
        f"PnL={sign}{daily_pnl:,.2f} ({sign}{pct:.2f}%) | "
        f"Positions={open_positions}"
    )
    log.info(msg)
    _dispatch(msg, subject="[TradeBot] Daily Summary")
