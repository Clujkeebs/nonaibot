"""
Circuit Breakers — hardware kill-switch for the bot.

Three levels of protection:
  1. TRADE_HALT  — stop opening new positions; hold existing ones
  2. FULL_HALT   — close ALL positions immediately; stop all activity
  3. KILL_SWITCH — permanent manual override (set via env var)

Triggers:
  - Daily P&L loss > DAILY_LOSS_LIMIT_PCT
  - Weekly P&L loss > WEEKLY_LOSS_LIMIT_PCT
  - Single position loss > 2× ATR stop (handled per-position in risk manager)
  - Manual kill switch (KILL_SWITCH=1 env var)

State is persisted to SQLite so a restart does NOT reset the breaker.
The breaker resets automatically at 9:00 AM ET each weekday (daily_reset).
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Optional

import pytz

import config
from utils.alerts import alert_circuit_break
from utils.logger import log

ET = pytz.timezone(config.TIMEZONE)


class HaltLevel(str, Enum):
    NONE       = "none"
    TRADE_HALT = "trade_halt"   # no new entries
    FULL_HALT  = "full_halt"    # close everything


class CircuitBreaker:
    """
    Thread-safe circuit breaker backed by SQLite.
    All public methods are safe to call from multiple scheduler threads.
    """

    _TABLE = """
    CREATE TABLE IF NOT EXISTS circuit_breaker (
        id          INTEGER PRIMARY KEY,
        level       TEXT    NOT NULL DEFAULT 'none',
        reason      TEXT,
        triggered_at TEXT,
        reset_at    TEXT
    )
    """

    def __init__(self, db_path: str = config.DB_PATH) -> None:
        self._db   = db_path
        self._kill = _env_kill()
        self._loss_timestamps: list = []   # recent losing trade timestamps
        self._init_db()
        log.info("CircuitBreaker initialised — kill_switch={}", self._kill)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(self._TABLE)
            if c.execute("SELECT COUNT(*) FROM circuit_breaker").fetchone()[0] == 0:
                c.execute(
                    "INSERT INTO circuit_breaker (level, reason) VALUES (?, ?)",
                    (HaltLevel.NONE, "init"),
                )

    # ── State accessors ───────────────────────────────────────────────────────

    def get_level(self) -> HaltLevel:
        if self._kill:
            return HaltLevel.FULL_HALT
        with self._conn() as c:
            row = c.execute("SELECT level FROM circuit_breaker WHERE id=1").fetchone()
            return HaltLevel(row["level"]) if row else HaltLevel.NONE

    def is_halted(self) -> bool:
        return self.get_level() != HaltLevel.NONE

    def trading_allowed(self) -> bool:
        return not self.is_halted()

    def full_halt_active(self) -> bool:
        return self.get_level() == HaltLevel.FULL_HALT

    # ── Trigger / reset ───────────────────────────────────────────────────────

    def trigger(self, level: HaltLevel, reason: str) -> None:
        # Alert cooldown: don't spam Slack/email if the same event fired recently
        now = datetime.now(ET)
        if hasattr(self, "_last_alert_at") and self._last_alert_at:
            if (now - self._last_alert_at).total_seconds() < 300:  # 5-min cooldown
                log.info("CircuitBreaker {} skipped (alert cooldown — {:.0f}s since last)",
                         level, (now - self._last_alert_at).total_seconds())
                # Still persist the state change even if we skip the alert
                with self._conn() as c:
                    c.execute(
                        "UPDATE circuit_breaker SET level=?, reason=?, triggered_at=?, reset_at=NULL WHERE id=1",
                        (level, reason, now.isoformat()),
                    )
                return
        self._last_alert_at = now

        with self._conn() as c:
            c.execute(
                "UPDATE circuit_breaker SET level=?, reason=?, triggered_at=?, reset_at=NULL WHERE id=1",
                (level, reason, now.isoformat()),
            )
        log.warning("CircuitBreaker TRIGGERED: {} — {}", level, reason)
        alert_circuit_break(reason, level)

    def reset(self) -> None:
        now = datetime.now(ET).isoformat()
        with self._conn() as c:
            c.execute(
                "UPDATE circuit_breaker SET level=?, reason='reset', reset_at=? WHERE id=1",
                (HaltLevel.NONE, now),
            )
        log.info("CircuitBreaker RESET at {}", now)

    # ── Auto-trigger checks ───────────────────────────────────────────────────

    def check_daily_loss(
        self,
        daily_pnl: float,
        portfolio_value: float,
    ) -> None:
        if portfolio_value <= 0:
            return
        loss_pct = -daily_pnl / portfolio_value
        if loss_pct >= config.DAILY_LOSS_LIMIT_PCT:
            # FULL_HALT — not just trade halt.  Stopping new entries while
            # existing losing positions remain open means the bad day keeps
            # getting worse.  Close everything and go to cash.
            self.trigger(
                HaltLevel.FULL_HALT,
                f"Daily loss {loss_pct:.2%} ≥ limit {config.DAILY_LOSS_LIMIT_PCT:.2%} — closing all positions",
            )

    def check_weekly_loss(
        self,
        weekly_pnl: float,
        portfolio_value: float,
    ) -> None:
        if portfolio_value <= 0:
            return
        loss_pct = -weekly_pnl / portfolio_value
        if loss_pct >= config.WEEKLY_LOSS_LIMIT_PCT:
            self.trigger(
                HaltLevel.FULL_HALT,
                f"Weekly loss {loss_pct:.2%} ≥ limit {config.WEEKLY_LOSS_LIMIT_PCT:.2%}",
            )

    # ── Drawdown tracker: halt on rapid consecutive losses ─────────────────
    # Track recent losing trades to detect momentum crush.
    def record_loss(self) -> None:
        """Call this after each closing trade that was a loss."""
        now = datetime.now(ET)
        self._loss_timestamps.append(now)
        # Prune entries older than 1 hour
        self._loss_timestamps = [
            t for t in self._loss_timestamps
            if (now - t).total_seconds() < 3600
        ]

    def check_drawdown_streak(self, min_losses: int = 3, window_hours: float = 1.0) -> None:
        """Halt if too many losing trades happened within a short window (momentum crush)."""
        if not hasattr(self, "_loss_timestamps"):
            return
        now = datetime.now(ET)
        recent = [
            t for t in self._loss_timestamps
            if (now - t).total_seconds() < window_hours * 3600
        ]
        if len(recent) >= min_losses:
            self.trigger(
                HaltLevel.TRADE_HALT,
                f"Drawdown streak: {len(recent)} losses in {window_hours:.0f}h (momentum crush)",
            )


def _env_kill() -> bool:
    import os
    return os.environ.get("KILL_SWITCH", "0") == "1"
