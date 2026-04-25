"""
Order Engine — all order placement, tracking, retries, and cancellation.

Design decisions:
  - Market orders for liquid equities during market hours (fastest fill)
  - Limit orders for crypto (bid/ask spread can be wide)
  - Auto-retry up to ORDER_RETRY_LIMIT on transient errors
  - Cancel-and-resubmit if not filled within ORDER_FILL_TIMEOUT seconds
  - All orders logged to SQLite; all errors escalated to alerts

Alpaca order IDs are stored; we poll for fill status rather than streaming
(streaming is additive complexity; polling every 10s is sufficient for our cadence).
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from typing import Optional

import pytz

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
)

import config
from utils.alerts import alert_error, alert_trade
from utils.logger import log

ET = pytz.timezone(config.TIMEZONE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT,
    symbol      TEXT,
    side        TEXT,
    qty         REAL,
    price       REAL,
    notional    REAL,
    strategy    TEXT,
    status      TEXT,
    filled_qty  REAL DEFAULT 0,
    filled_price REAL DEFAULT 0,
    created_at  TEXT,
    updated_at  TEXT
)
"""


class OrderEngine:
    def __init__(self) -> None:
        self._client = TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=config.ALPACA_PAPER,
        )
        self._db = config.DB_PATH
        self._init_db()
        log.info("OrderEngine initialised (paper={})", config.ALPACA_PAPER)

    # ── Public interface ──────────────────────────────────────────────────────

    def buy(
        self,
        symbol: str,
        qty: float,
        strategy: str,
        price: float,
        is_crypto: bool = False,
    ) -> bool:
        return self._execute(
            symbol, "buy", qty, strategy, price, is_crypto
        )

    def sell(
        self,
        symbol: str,
        qty: float,
        strategy: str,
        price: float,
        is_crypto: bool = False,
    ) -> bool:
        return self._execute(
            symbol, "sell", qty, strategy, price, is_crypto
        )

    def close_position(self, symbol: str) -> bool:
        """Close an open position via Alpaca's close-position endpoint."""
        try:
            self._client.close_position(symbol)
            log.info("close_position submitted for {}", symbol)
            return True
        except Exception as e:
            log.error("close_position({}) failed: {}", symbol, e)
            alert_error(str(e), f"close_position {symbol}")
            return False

    def close_all_positions(self) -> None:
        """Emergency — close everything immediately."""
        try:
            self._client.close_all_positions(cancel_orders=True)
            log.warning("close_all_positions submitted")
        except Exception as e:
            log.error("close_all_positions failed: {}", e)
            alert_error(str(e), "close_all_positions")

    def cancel_all_orders(self) -> None:
        try:
            self._client.cancel_orders()
            log.info("All open orders cancelled")
        except Exception as e:
            log.error("cancel_all_orders failed: {}", e)

    # ── Internal execution ────────────────────────────────────────────────────

    def _execute(
        self,
        symbol: str,
        side: str,
        qty: float,
        strategy: str,
        price: float,
        is_crypto: bool,
    ) -> bool:
        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL

        for attempt in range(1, config.ORDER_RETRY_LIMIT + 1):
            try:
                order_id = self._submit_order(symbol, side_enum, qty, price, is_crypto)
                if not order_id:
                    continue

                self._persist_order(order_id, symbol, side, qty, price, strategy)
                filled = self._wait_for_fill(order_id, symbol)
                if filled:
                    alert_trade(symbol, side, qty, price, strategy)
                    return True
                else:
                    log.warning(
                        "Order {} not filled in {}s — cancelling (attempt {}/{})",
                        order_id, config.ORDER_FILL_TIMEOUT, attempt, config.ORDER_RETRY_LIMIT,
                    )
                    self._cancel_order(order_id)

            except Exception as e:
                log.error(
                    "Order attempt {}/{} for {} {} {} failed: {}",
                    attempt, config.ORDER_RETRY_LIMIT, side, qty, symbol, e,
                )
                if attempt == config.ORDER_RETRY_LIMIT:
                    alert_error(str(e), f"{side} {qty} {symbol}")

        return False

    def _submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        is_crypto: bool,
    ) -> Optional[str]:
        try:
            if is_crypto:
                # Limit order within 0.1% for crypto (taker fee saving)
                limit_price = round(
                    price * (1 + config.SLIPPAGE_LIMIT_PCT)
                    if side == OrderSide.BUY
                    else price * (1 - config.SLIPPAGE_LIMIT_PCT),
                    2,
                )
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    limit_price=limit_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )

            order = self._client.submit_order(req)
            log.info(
                "Order submitted: id={} {} {} {} @ ~{:.4f}",
                order.id, side.value, qty, symbol, price,
            )
            return str(order.id)
        except Exception as e:
            log.error("_submit_order({} {} {}) error: {}", side, qty, symbol, e)
            return None

    def _wait_for_fill(self, order_id: str, symbol: str) -> bool:
        deadline = time.time() + config.ORDER_FILL_TIMEOUT
        while time.time() < deadline:
            try:
                order = self._client.get_order_by_id(order_id)
                status = order.status
                if status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    filled_qty   = float(order.filled_qty or 0)
                    filled_price = float(order.filled_avg_price or 0)
                    self._update_order(order_id, str(status), filled_qty, filled_price)
                    log.info(
                        "Order {} filled: qty={} avg_px={}",
                        order_id, filled_qty, filled_price,
                    )
                    return True
                if status in (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
                    self._update_order(order_id, str(status), 0, 0)
                    log.warning("Order {} terminal status: {}", order_id, status)
                    return False
            except Exception as e:
                log.warning("_wait_for_fill poll error: {}", e)
            time.sleep(5)
        return False

    def _cancel_order(self, order_id: str) -> None:
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception as e:
            log.warning("_cancel_order({}) error: {}", order_id, e)

    # ── SQLite persistence ────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(_SCHEMA)

    def _persist_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        strategy: str,
    ) -> None:
        now = datetime.now(ET).isoformat()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO orders
                  (order_id, symbol, side, qty, price, notional, strategy, status, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (order_id, symbol, side, qty, price, qty * price, strategy,
                 "pending", now, now),
            )

    def _update_order(
        self,
        order_id: str,
        status: str,
        filled_qty: float,
        filled_price: float,
    ) -> None:
        now = datetime.now(ET).isoformat()
        with self._conn() as c:
            c.execute(
                """
                UPDATE orders
                SET status=?, filled_qty=?, filled_price=?, updated_at=?
                WHERE order_id=?
                """,
                (status, filled_qty, filled_price, now, order_id),
            )
