"""
Task Manager — APScheduler job definitions for all recurring tasks.

Schedule overview:
  Job                    Interval         Active
  ─────────────────────  ───────────────  ────────────────────────────────
  run_equity_strategies  every 5 min      Mon–Fri 09:35–15:55 ET
  run_crypto_strategies  every 15 min     24 / 7
  run_sector_rotation    every Sunday     00:00 ET (weekly rebalance)
  check_all_exits        every 2 min      Mon–Fri 09:35–15:55 + 24/7 crypto
  check_circuit_breakers every 1 min      24 / 7
  update_regime          every 60 min     24 / 7
  daily_open_tasks       daily 09:30      Mon–Fri
  daily_close_tasks      daily 16:05      Mon–Fri
  portfolio_heartbeat    every 30 min     24 / 7
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytz
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import config
from utils.logger import log

if TYPE_CHECKING:
    from engine import TradingEngine

ET = pytz.timezone(config.TIMEZONE)


def _is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    from datetime import time as dtime
    return dtime(9, 35) <= now.time() <= dtime(15, 55)


class TaskManager:
    def __init__(self, engine: "TradingEngine") -> None:
        self._engine = engine
        self._scheduler = BackgroundScheduler(
            executors={"default": ThreadPoolExecutor(max_workers=4)},
            timezone=ET,
        )

    def start(self) -> None:
        e = self._engine

        # ── Opening Range Breakout (market hours, needs 5-min bars) ─────────
        self._scheduler.add_job(
            self._safe(e.run_opening_range),
            IntervalTrigger(minutes=3, timezone=ET),
            id="opening_range",
            name="Opening Range Breakout",
            max_instances=1,
            coalesce=True,
        )

        # ── Equity strategies (market hours only, enforced inside job) ───────
        self._scheduler.add_job(
            self._safe(e.run_equity_strategies),
            IntervalTrigger(minutes=3, timezone=ET),
            id="equity_strategies",
            name="Equity Strategies",
            max_instances=1,
            coalesce=True,
        )

        # ── Crypto strategies (24/7) ──────────────────────────────────────────
        self._scheduler.add_job(
            self._safe(e.run_crypto_strategies),
            IntervalTrigger(minutes=5, timezone=ET),
            id="crypto_strategies",
            name="Crypto Strategies",
            max_instances=1,
            coalesce=True,
        )

        # ── Exit checks ───────────────────────────────────────────────────────
        self._scheduler.add_job(
            self._safe(e.check_all_exits),
            IntervalTrigger(minutes=2, timezone=ET),
            id="exit_checks",
            name="Exit Checks",
            max_instances=1,
            coalesce=True,
        )

        # ── Circuit breaker monitor ────────────────────────────────────────────
        self._scheduler.add_job(
            self._safe(e.check_circuit_breakers),
            IntervalTrigger(minutes=1, timezone=ET),
            id="circuit_breakers",
            name="Circuit Breakers",
            max_instances=1,
            coalesce=True,
        )

        # ── Regime update (hourly) ─────────────────────────────────────────────
        self._scheduler.add_job(
            self._safe(e.update_regime),
            IntervalTrigger(minutes=60, timezone=ET),
            id="regime_update",
            name="Regime Update",
            max_instances=1,
            coalesce=True,
        )

        # ── Sector rotation (weekly, Sunday midnight) ─────────────────────────
        self._scheduler.add_job(
            self._safe(e.run_sector_rotation),
            CronTrigger(day_of_week="sun", hour=0, minute=0, timezone=ET),
            id="sector_rotation",
            name="Sector Rotation",
            max_instances=1,
        )

        # ── Daily open tasks (market open prep) ───────────────────────────────
        self._scheduler.add_job(
            self._safe(e.daily_open),
            CronTrigger(day_of_week="mon-fri", hour=9, minute=30, timezone=ET),
            id="daily_open",
            name="Daily Open",
        )

        # ── Daily close tasks (EOD wrap-up) ───────────────────────────────────
        self._scheduler.add_job(
            self._safe(e.daily_close),
            CronTrigger(day_of_week="mon-fri", hour=16, minute=5, timezone=ET),
            id="daily_close",
            name="Daily Close",
        )

        # ── Portfolio heartbeat (30-min summary log) ──────────────────────────
        self._scheduler.add_job(
            self._safe(e.portfolio_heartbeat),
            IntervalTrigger(minutes=30, timezone=ET),
            id="portfolio_heartbeat",
            name="Portfolio Heartbeat",
            max_instances=1,
            coalesce=True,
        )

        # ── Active position management (pyramid / scale-out, every 30 min) ───
        self._scheduler.add_job(
            self._safe(e.manage_open_positions),
            IntervalTrigger(minutes=30, timezone=ET),
            id="manage_positions",
            name="Position Management",
            max_instances=1,
            coalesce=True,
        )

        # ── Weekly theme rebalance (Sunday 01:00 ET, after sector rotation) ──
        self._scheduler.add_job(
            self._safe(e.theme_rebalance),
            CronTrigger(day_of_week="sun", hour=1, minute=0, timezone=ET),
            id="theme_rebalance",
            name="Theme Rebalance",
            max_instances=1,
        )

        self._scheduler.start()
        log.info(
            "TaskManager started — {} jobs scheduled",
            len(self._scheduler.get_jobs()),
        )

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            log.info("TaskManager stopped")

    @staticmethod
    def _safe(fn):
        """Wrap any job in a try/except so one failure doesn't kill the scheduler."""
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception as e:
                log.error("Scheduled job {} raised: {}", fn.__name__, e)
                from utils.alerts import alert_error
                alert_error(str(e), fn.__name__)
        wrapper.__name__ = fn.__name__
        return wrapper
