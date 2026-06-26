"""
MarketClock — caches the Alpaca clock API response to avoid burning rate-limit
tokens on every loop iteration. Cache TTL = 60 seconds.
"""
from __future__ import annotations

import time
from datetime import datetime, time as dtime
from typing import Optional

import pytz

from core.config import BotConfig


class MarketClock:
    """
    Provides is_market_open() with a 60-second cache.
    Also supports a soft local check (no API call) based on known NYSE hours
    to avoid the API call entirely in obvious off-hours.
    """

    def __init__(self, config: BotConfig, broker) -> None:
        self._cfg = config
        self._broker = broker
        self._et = pytz.timezone(config.timezone)
        self._cached_open: Optional[bool] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 60.0  # seconds

    def is_market_open(self) -> bool:
        """Return True if US equity markets are currently open."""
        now = time.time()
        if now - self._cache_time < self._cache_ttl and self._cached_open is not None:
            return self._cached_open

        # Fast local check: definitely closed on weekends
        now_et = datetime.now(self._et)
        if now_et.weekday() >= 5:  # Saturday/Sunday
            self._cached_open = False
            self._cache_time = now
            return False

        # Fast local check: definitely before or after trading window
        t = now_et.time()
        open_t = dtime(self._cfg.equity_open_hour, self._cfg.equity_open_min)
        close_t = dtime(self._cfg.equity_close_hour, self._cfg.equity_close_min)
        if t < dtime(8, 0) or t > dtime(17, 0):
            self._cached_open = False
            self._cache_time = now
            return False

        # Within possible trading hours — consult Alpaca for holidays/early closes
        try:
            clock = self._broker.get_clock()
            is_open = bool(clock.is_open)
            self._cached_open = is_open
            self._cache_time = now
            return is_open
        except Exception:
            # Fallback: use local time check within configured hours
            self._cached_open = open_t <= t <= close_t
            self._cache_time = now
            return self._cached_open

    def now_et(self) -> datetime:
        return datetime.now(self._et)
