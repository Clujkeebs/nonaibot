"""
Audit delivery — sends the audit report text to configured channels.

Supported channels (all off by default, enabled via env vars):
  Slack:    set SLACK_WEBHOOK_URL
  Discord:  set DISCORD_WEBHOOK_URL
  Telegram: set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID

Delivery is best-effort: failures are logged but don't raise exceptions.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from core.config import BotConfig

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds per HTTP request


def send_report(text: str, config: BotConfig) -> None:
    """Send report text to all configured channels."""
    sent_to: list[str] = []

    if config.slack_webhook:
        try:
            _send_slack(text, config.slack_webhook)
            sent_to.append("Slack")
        except Exception as e:
            logger.warning("Slack delivery failed: %s", e)

    if config.discord_webhook:
        try:
            _send_discord(text, config.discord_webhook)
            sent_to.append("Discord")
        except Exception as e:
            logger.warning("Discord delivery failed: %s", e)

    if config.telegram_token and config.telegram_chat:
        try:
            _send_telegram(text, config.telegram_token, config.telegram_chat)
            sent_to.append("Telegram")
        except Exception as e:
            logger.warning("Telegram delivery failed: %s", e)

    if sent_to:
        logger.info("Audit report delivered to: %s", ", ".join(sent_to))
    else:
        logger.info("Audit report: no notification channels configured (set SLACK_WEBHOOK_URL etc.)")


def send_alert(message: str, config: BotConfig, level: str = "INFO") -> None:
    """Send a short alert message (trade, error, circuit breaker)."""
    prefix = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "🚨", "KILL": "🛑"}.get(level, "")
    text = f"{prefix} [{level}] {message}"

    if config.slack_webhook:
        try:
            _send_slack(text, config.slack_webhook)
        except Exception:
            pass

    if config.discord_webhook:
        try:
            _send_discord(text, config.discord_webhook)
        except Exception:
            pass

    if config.telegram_token and config.telegram_chat:
        try:
            _send_telegram(text, config.telegram_token, config.telegram_chat)
        except Exception:
            pass


def _send_slack(text: str, webhook_url: str) -> None:
    # Split long text into Slack's 3000-char block limit
    for chunk in _chunks(text, 3000):
        resp = requests.post(
            webhook_url,
            json={"text": f"```{chunk}```"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()


def _send_discord(text: str, webhook_url: str) -> None:
    # Discord limit: 2000 chars per message
    for chunk in _chunks(text, 1900):
        resp = requests.post(
            webhook_url,
            json={"content": f"```{chunk}```"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()


def _send_telegram(text: str, token: str, chat_id: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram limit: 4096 chars per message (we use Markdown code block)
    for chunk in _chunks(text, 4000):
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": f"```\n{chunk}\n```", "parse_mode": "Markdown"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()


def _chunks(text: str, size: int):
    for i in range(0, len(text), size):
        yield text[i: i + size]
