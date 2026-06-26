"""
Executor — places, tracks, and verifies orders on Alpaca.

Design principles:
  1. FAIL SAFE: any exception must NOT place an order — log and return False
  2. Idempotency: client_order_id prevents duplicate orders on restart
  3. Fractional rules: checks fractionable flag before notional orders
  4. No resting stops: fractional positions can't use resting stop orders →
     stops are managed in code by the exit checker in main.py
  5. Wash-trade guard: never have opposing limit orders on the same symbol

Order type selection:
  Crypto BUY:   limit order @ price + small slippage (to get maker/limit fee)
  Crypto SELL:  market order (instant fill, must exit immediately)
  Equity BUY fractionable:  notional market order (DAY, fractional allowed)
  Equity BUY non-fractionable: whole-share market order (DAY)
  Equity SELL:  market order (DAY, always sells full qty)

All orders are polled for fill status; unfilled orders are cancelled and
optionally retried.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import pytz

from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

import config as _cfg_module  # legacy import kept for compatibility
from core.broker import BrokerClient
from core.config import BotConfig
from core.state import SQLiteState
from execution.assets import AssetCache

logger = logging.getLogger(__name__)


def _client_order_id(symbol: str, side: str, strategy: str) -> str:
    """
    Generate idempotency key for an order.
    Scoped to a 5-minute window so a restart within the same cycle doesn't duplicate.
    """
    import hashlib
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    window = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    raw = f"{symbol}|{side}|{strategy}|{window.isoformat()}"
    # Alpaca client_order_id max length is 48 chars
    return "bot-" + hashlib.md5(raw.encode()).hexdigest()[:44]


class Executor:
    def __init__(
        self,
        broker: BrokerClient,
        assets: AssetCache,
        state: SQLiteState,
        config: BotConfig,
    ) -> None:
        self._broker = broker
        self._assets = assets
        self._state = state
        self._cfg = config

    # ── Public interface ──────────────────────────────────────────────────────

    def buy(
        self,
        symbol: str,
        qty: float,
        strategy: str,
        price: float,
        is_crypto: bool,
        reason: str = "",
    ) -> bool:
        """Place a buy order. Returns True if order was accepted and (eventually) filled."""
        return self._execute("buy", symbol, qty, strategy, price, is_crypto, reason)

    def sell(
        self,
        symbol: str,
        qty: float,
        strategy: str,
        price: float,
        is_crypto: bool,
        reason: str = "",
    ) -> bool:
        """Place a sell order for a specific qty."""
        return self._execute("sell", symbol, qty, strategy, price, is_crypto, reason)

    def close_position(self, symbol: str) -> bool:
        """Close a position using Alpaca's close-position endpoint (handles any qty)."""
        try:
            self._broker.close_position(symbol)
            logger.info("close_position submitted for %s", symbol)
            return True
        except Exception as e:
            logger.error("close_position(%s) failed: %s", symbol, e)
            self._state.log_error("executor", f"close_position({symbol}): {e}")
            return False

    def close_all_positions(self) -> None:
        """Emergency flatten — closes everything immediately."""
        try:
            self._broker.close_all_positions(cancel_orders=True)
            logger.warning("EMERGENCY: close_all_positions submitted")
        except Exception as e:
            logger.error("close_all_positions failed: %s", e)
            self._state.log_error("executor", f"close_all_positions: {e}")

    # ── Internal execution ────────────────────────────────────────────────────

    def _execute(
        self,
        side: str,
        symbol: str,
        qty: float,
        strategy: str,
        price: float,
        is_crypto: bool,
        reason: str = "",
    ) -> bool:
        """
        Core execution method.
        Returns False without placing an order if ANYTHING goes wrong.
        """
        try:
            return self._execute_inner(side, symbol, qty, strategy, price, is_crypto, reason)
        except Exception as e:
            # Fail safe: catch all exceptions, never let them propagate to place an order
            logger.error(
                "EXECUTOR EXCEPTION [%s %s %s]: %s — no order placed", side, qty, symbol, e
            )
            self._state.log_error("executor", f"{side} {qty} {symbol}: {e}")
            return False

    def _execute_inner(
        self,
        side: str,
        symbol: str,
        qty: float,
        strategy: str,
        price: float,
        is_crypto: bool,
        reason: str,
    ) -> bool:
        cfg = self._cfg

        if qty <= 0 or price <= 0:
            logger.warning("Executor: invalid qty=%s price=%s for %s %s", qty, price, side, symbol)
            return False

        # ── Idempotency ────────────────────────────────────────────────────
        client_oid = _client_order_id(symbol, side, strategy)
        if self._state.idempotency_check(client_oid):
            logger.info("Idempotency: skipping duplicate %s %s (coid=%s)", side, symbol, client_oid)
            return False

        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL

        # ── Build order request ────────────────────────────────────────────
        request = self._build_request(
            symbol=symbol,
            side=side_enum,
            qty=qty,
            price=price,
            is_crypto=is_crypto,
            client_oid=client_oid,
            strategy=strategy,
        )
        if request is None:
            return False

        # ── Submit with retry ──────────────────────────────────────────────
        max_attempts = 3
        delay = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                order = self._broker.submit_order(request)
                order_id = str(order.id)
                logger.info(
                    "Order submitted: %s %s %s qty=%.6f price=~%.4f id=%s",
                    side, symbol, strategy, qty, price, order_id
                )

                # Persist immediately (before waiting for fill — restart safe)
                notional = qty * price if is_crypto else None
                self._state.save_order(
                    order_id=order_id,
                    client_order_id=client_oid,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    notional=notional,
                    price=price,
                    strategy=strategy,
                    reason=reason,
                )

                # Wait for fill
                filled = self._wait_for_fill(order_id, symbol)
                if filled:
                    logger.info("Order FILLED: %s %s %s", side, symbol, order_id)
                    return True
                else:
                    # Cancel and retry (different request on next attempt)
                    self._broker.cancel_order_by_id(order_id)
                    self._state.update_order(order_id, "cancelled", 0, 0)
                    if attempt < max_attempts:
                        logger.warning(
                            "Order %s not filled in timeout — retrying (%d/%d)",
                            order_id, attempt, max_attempts
                        )
                        time.sleep(delay)
                        delay *= 2
                        # Rebuild as limit order on retry to guarantee fill
                        request = self._build_limit_retry(
                            symbol, side_enum, qty, price, is_crypto, client_oid + f"r{attempt}"
                        )
                        if request is None:
                            return False

            except Exception as e:
                err_msg = str(e).lower()
                logger.error("Order attempt %d/%d [%s %s]: %s", attempt, max_attempts, side, symbol, e)
                self._state.log_error("executor", f"attempt {attempt}: {side} {symbol}: {e}")

                # Wash-trade rejection → don't retry
                if "403" in err_msg or "wash" in err_msg or "forbidden" in err_msg:
                    logger.warning("Wash-trade or 403 rejection for %s — not retrying", symbol)
                    return False

                # Rate limit → backoff
                if "429" in err_msg or "rate" in err_msg:
                    time.sleep(delay * 3)
                    delay *= 2
                elif attempt == max_attempts:
                    return False
                else:
                    time.sleep(delay)
                    delay *= 2

        return False

    def _build_request(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        is_crypto: bool,
        client_oid: str,
        strategy: str,
    ):
        """
        Build the appropriate order request given Alpaca's constraints:
        - Fractional equity: notional + market + DAY
        - Whole-share equity: qty + market + DAY
        - Crypto buy: limit + GTC (saves taker fee)
        - Crypto sell: market + GTC (must exit fast)
        """
        try:
            if is_crypto:
                return self._build_crypto_request(symbol, side, qty, price, client_oid)
            else:
                return self._build_equity_request(symbol, side, qty, price, client_oid)
        except Exception as e:
            logger.error("_build_request(%s %s): %s", side, symbol, e)
            return None

    def _build_equity_request(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        client_oid: str,
    ):
        is_fractionable = self._assets.is_fractionable(symbol)

        # SELL: always whole-share market order
        if side == OrderSide.SELL:
            import math
            sell_qty = math.floor(abs(qty))
            if sell_qty <= 0:
                # Try fractional sell if symbol is fractionable
                if is_fractionable:
                    sell_qty = round(abs(qty), 4)
                else:
                    logger.warning("Cannot sell <1 share of non-fractionable %s qty=%.4f", symbol, qty)
                    return None
            return MarketOrderRequest(
                symbol=symbol,
                qty=sell_qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_oid,
            )

        # BUY: fractionable → notional market order; non-fractionable → whole shares
        if is_fractionable:
            notional = round(qty * price, 2)
            notional = max(notional, self._cfg.min_notional)
            return MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_oid,
            )
        else:
            import math
            whole_qty = math.floor(qty)
            if whole_qty <= 0:
                logger.warning(
                    "Non-fractionable %s: qty=%.4f rounds to 0 whole shares — skipping", symbol, qty
                )
                return None
            return MarketOrderRequest(
                symbol=symbol,
                qty=whole_qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_oid,
            )

    def _build_crypto_request(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        client_oid: str,
    ):
        """
        Crypto buys: limit order (gets maker rate 0.15% vs taker 0.25%).
        Crypto sells: market order (must exit immediately).
        NOTE: never have opposing limit orders on the same symbol (Alpaca 403).
        """
        qty = round(abs(qty), 6)
        if qty <= 0:
            return None

        if side == OrderSide.BUY:
            # Limit slightly above current price to get filled but potentially at maker rate
            limit_price = round(price * 1.002, 6)  # +0.2% tolerance
            return LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price,
                client_order_id=client_oid,
            )
        else:
            return MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                client_order_id=client_oid,
            )

    def _build_limit_retry(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        is_crypto: bool,
        client_oid: str,
    ):
        """On retry, use a more aggressive limit price to guarantee fill."""
        try:
            if is_crypto:
                if side == OrderSide.BUY:
                    limit_price = round(price * 1.005, 6)
                    import math
                    qty = round(qty, 6)
                    return LimitOrderRequest(
                        symbol=symbol, qty=qty, side=side,
                        time_in_force=TimeInForce.GTC,
                        limit_price=limit_price, client_order_id=client_oid,
                    )
                else:
                    return MarketOrderRequest(
                        symbol=symbol, qty=round(qty, 6), side=side,
                        time_in_force=TimeInForce.GTC, client_order_id=client_oid,
                    )
            else:
                # Equity retry: just use market order (should always fill)
                import math
                whole_qty = math.floor(qty)
                if whole_qty <= 0:
                    return None
                return MarketOrderRequest(
                    symbol=symbol, qty=whole_qty, side=side,
                    time_in_force=TimeInForce.DAY, client_order_id=client_oid,
                )
        except Exception as e:
            logger.error("_build_limit_retry: %s", e)
            return None

    def _wait_for_fill(self, order_id: str, symbol: str, timeout: int = 60) -> bool:
        """Poll for fill status. Return True if filled within timeout seconds."""
        deadline = time.time() + timeout
        poll_interval = 5

        while time.time() < deadline:
            try:
                order = self._broker.get_order_by_id(order_id)
                status = order.status
                if status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    filled_qty = float(order.filled_qty or 0)
                    filled_price = float(order.filled_avg_price or 0)
                    self._state.update_order(order_id, str(status), filled_qty, filled_price)
                    return True
                if status in (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
                    self._state.update_order(order_id, str(status), 0, 0)
                    logger.warning("Order %s terminal status: %s", order_id, status)
                    return False
            except Exception as e:
                logger.warning("_wait_for_fill poll error for %s: %s", order_id, e)
            time.sleep(poll_interval)

        return False
