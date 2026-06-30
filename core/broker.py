"""
Broker client wrapper — single place for all Alpaca SDK calls.

Handles:
  - Rate limiting: 200 req/min free tier (token bucket)
  - HTTP 429 backoff: exponential, up to 4 retries
  - Lazy singleton clients (one TradingClient, one StockDataClient, one CryptoDataClient)

All other modules import from here rather than instantiating SDK clients directly.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from core.config import BotConfig

logger = logging.getLogger(__name__)

# ── Rate limiter (token bucket) ───────────────────────────────────────────────
# Free tier: 200 data requests/min. Trading API has separate (generous) limits.
# We share one bucket for data calls; trading calls are infrequent enough to skip.

_RATE_LIMIT_PER_MIN = 200
_RATE_LOCK = threading.Lock()
_tokens: float = _RATE_LIMIT_PER_MIN
_last_refill: float = time.time()


def _acquire_token() -> None:
    """Block until a rate-limit token is available."""
    global _tokens, _last_refill
    with _RATE_LOCK:
        now = time.time()
        elapsed = now - _last_refill
        refill = elapsed * (_RATE_LIMIT_PER_MIN / 60.0)
        _tokens = min(_RATE_LIMIT_PER_MIN, _tokens + refill)
        _last_refill = now

        if _tokens >= 1:
            _tokens -= 1
            return

        wait_secs = (1.0 - _tokens) / (_RATE_LIMIT_PER_MIN / 60.0)
    # Release lock before sleeping
    time.sleep(wait_secs)
    with _RATE_LOCK:
        _tokens = max(0.0, _tokens - 1)


def _with_retry(fn, *args, retries: int = 4, **kwargs) -> Any:
    """Call fn with exponential backoff on 429 / transient errors."""
    delay = 2.0
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            _acquire_token()
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg or "too many" in msg:
                if attempt < retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
            # Non-rate-limit errors: re-raise immediately
            raise
    raise last_exc  # type: ignore[misc]


class BrokerClient:
    """
    Thin wrapper that exposes Alpaca clients and applies rate limiting.
    One instance per process — pass it around rather than re-instantiating.
    """

    def __init__(self, config: BotConfig) -> None:
        self._cfg = config
        self._trading: Optional[TradingClient] = None
        self._stock_data: Optional[StockHistoricalDataClient] = None
        self._crypto_data: Optional[CryptoHistoricalDataClient] = None

    # ── Credential / mode auto-detection ───────────────────────────────────────

    def validate_credentials(self) -> Optional[str]:
        """
        Figure out whether the provided keys are LIVE or PAPER by actually calling
        the trading endpoint on both, and pin the bot to whichever authenticates.

        This removes the entire class of "paper key against live endpoint" (and
        vice-versa) failures — the operator just supplies their keys and the bot
        adapts. Live is tried first so real keys go live.

        Returns "live", "paper", or None (neither authenticated → bad keys).
        On success it pins cfg.paper, rebuilds the trading client, and writes the
        resolved mode back to TRADING_MODE so config.reload() stays consistent.
        """
        if not self._cfg.api_key or not self._cfg.secret_key:
            logger.error("No Alpaca API key/secret found in environment")
            return None

        for paper in (False, True):
            label = "paper" if paper else "live"
            try:
                client = TradingClient(
                    api_key=self._cfg.api_key,
                    secret_key=self._cfg.secret_key,
                    paper=paper,
                )
                acct = client.get_account()  # the real auth test
                # Success — pin this mode everywhere
                self._cfg.paper = paper
                self._trading = client
                os.environ["TRADING_MODE"] = label  # survive config.reload()
                logger.info(
                    "Alpaca auth OK in %s mode (account %s, status=%s)",
                    label.upper(), getattr(acct, "account_number", "?"),
                    getattr(acct, "status", "?"),
                )
                return label
            except Exception as e:
                logger.warning("Alpaca auth failed in %s mode: %s", label, str(e)[:200])
                continue

        # Neither worked — emit a masked diagnostic to spot typos/whitespace
        k = self._cfg.api_key
        s = self._cfg.secret_key
        masked = f"{k[:4]}…{k[-2:]}" if len(k) >= 6 else "(too short)"
        logger.error("=" * 70)
        logger.error("ALPACA AUTH FAILED ON BOTH LIVE AND PAPER ENDPOINTS")
        logger.error("  key id : %s  (length %d)", masked, len(k))
        logger.error("  secret : length %d", len(s))
        logger.error("  Likely: keys mistyped, key/secret swapped, or wrong account.")
        logger.error("  Alpaca keys: id starts 'PK' (paper) or 'AK' (live); secret is ~40 chars.")
        logger.error("=" * 70)
        return None

    # ── Lazily-constructed clients ─────────────────────────────────────────────

    @property
    def trading(self) -> TradingClient:
        if self._trading is None:
            self._trading = TradingClient(
                api_key=self._cfg.api_key,
                secret_key=self._cfg.secret_key,
                paper=self._cfg.paper,
            )
        return self._trading

    @property
    def stock_data(self) -> StockHistoricalDataClient:
        if self._stock_data is None:
            self._stock_data = StockHistoricalDataClient(
                api_key=self._cfg.api_key,
                secret_key=self._cfg.secret_key,
            )
        return self._stock_data

    @property
    def crypto_data(self) -> CryptoHistoricalDataClient:
        if self._crypto_data is None:
            self._crypto_data = CryptoHistoricalDataClient(
                api_key=self._cfg.api_key,
                secret_key=self._cfg.secret_key,
            )
        return self._crypto_data

    # ── Trading API wrappers ───────────────────────────────────────────────────

    def get_account(self):
        return _with_retry(self.trading.get_account)

    def get_all_positions(self):
        return _with_retry(self.trading.get_all_positions)

    def get_clock(self):
        return _with_retry(self.trading.get_clock)

    def submit_order(self, request):
        # Trading orders don't count toward the data rate limit, but wrap anyway
        return self.trading.submit_order(request)

    def get_order_by_id(self, order_id: str):
        return self.trading.get_order_by_id(order_id)

    def cancel_order_by_id(self, order_id: str) -> None:
        try:
            self.trading.cancel_order_by_id(order_id)
        except Exception:
            pass

    def close_all_positions(self, cancel_orders: bool = True) -> None:
        self.trading.close_all_positions(cancel_orders=cancel_orders)

    def close_position(self, symbol: str) -> None:
        self.trading.close_position(symbol)

    def get_all_assets(self):
        return _with_retry(self.trading.get_all_assets)

    # ── Data API wrappers ──────────────────────────────────────────────────────

    def get_stock_bars(self, request):
        return _with_retry(self.stock_data.get_stock_bars, request)

    def get_crypto_bars(self, request):
        return _with_retry(self.crypto_data.get_crypto_bars, request)
