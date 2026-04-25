"""
Risk Manager — central gatekeeper for every order.

Responsibilities:
  1. Position sizing   (ATR-based dollar risk)
  2. Exposure checks   (per-symbol, per-sector, total crypto)
  3. Portfolio heat    (total open risk as % of equity)
  4. Signal filtering  (reject duplicates, illiquid symbols)
  5. Regime scaling    (multiply size by regime weight)

The return value of check_signal() is (approved: bool, reason: str, qty: float).
qty is pre-calculated; the execution engine places the order for exactly qty shares/coins.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import config
from data.universe import Universe
from strategies.base import Signal
from strategies.regime_filter import RegimeFilter
from utils.logger import log

_universe = Universe()


class RiskManager:
    def __init__(self, regime_filter: RegimeFilter) -> None:
        self._regime = regime_filter

    # ── Public entry point ────────────────────────────────────────────────────

    def check_signal(
        self,
        signal: Signal,
        portfolio_value: float,
        buying_power: float,
        open_positions: Dict[str, dict],   # {symbol: {"qty": float, "market_value": float, "side": str}}
        daily_pnl: float = 0.0,
    ) -> Tuple[bool, str, float]:
        """
        Returns (approved, reason, qty).
        qty is 0 if not approved.
        """
        if portfolio_value <= 0:
            return False, "portfolio_value <= 0", 0.0

        # ── Hard reject conditions ────────────────────────────────────────────
        if signal.atr <= 0 or signal.price <= 0:
            return False, "invalid atr or price", 0.0

        if signal.stop_distance <= 0:
            return False, "invalid stop distance", 0.0

        # Already long in this symbol?
        if signal.side == "buy" and signal.symbol in open_positions:
            return False, f"already holding {signal.symbol}", 0.0

        # Max open positions
        if len(open_positions) >= config.MAX_OPEN_POSITIONS:
            return False, "max_open_positions reached", 0.0

        # Regime equity block
        if not signal.is_crypto and not self._regime.equity_trading_enabled():
            return False, "equity trading halted by regime", 0.0

        # ── Size calculation ──────────────────────────────────────────────────
        regime_weight  = self._regime.strategy_weight(signal.strategy)
        pos_scale      = self._regime.max_position_scale()

        dollar_risk    = portfolio_value * config.RISK_PER_TRADE_PCT * regime_weight
        qty_by_risk    = dollar_risk / signal.stop_distance

        # Cap at MAX_POSITION_PCT of portfolio
        max_dollars    = portfolio_value * config.MAX_POSITION_PCT * pos_scale
        qty_by_max_pos = max_dollars / signal.price

        raw_qty = min(qty_by_risk, qty_by_max_pos)

        # Round properly
        if signal.is_crypto:
            qty = round(raw_qty, 4)
        else:
            qty = math.floor(raw_qty)

        if qty <= 0:
            return False, "computed qty <= 0", 0.0

        notional = qty * signal.price
        if notional < config.MIN_NOTIONAL:
            return False, f"notional ${notional:.2f} below minimum", 0.0

        if notional > buying_power:
            # Scale down to available buying power (minus 5% buffer)
            affordable = buying_power * 0.95 / signal.price
            if signal.is_crypto:
                qty = round(affordable, 4)
            else:
                qty = math.floor(affordable)
            if qty <= 0:
                return False, "insufficient buying power", 0.0
            notional = qty * signal.price

        # ── Exposure checks ───────────────────────────────────────────────────
        ok, msg = self._check_exposure(signal, notional, portfolio_value, open_positions)
        if not ok:
            return False, msg, 0.0

        # ── Portfolio heat ────────────────────────────────────────────────────
        total_exposure = sum(
            p.get("market_value", 0) for p in open_positions.values()
        )
        if (total_exposure + notional) / portfolio_value > config.PORTFOLIO_HEAT_MAX:
            return False, "portfolio heat limit reached", 0.0

        log.info(
            "RiskManager APPROVED: {} {} {} @ {:.4f} | qty={} notional={:.2f} risk={:.2f}",
            signal.strategy, signal.side, signal.symbol,
            signal.price, qty, notional, dollar_risk,
        )
        return True, "approved", float(qty)

    # ── Exposure helpers ──────────────────────────────────────────────────────

    def _check_exposure(
        self,
        signal: Signal,
        new_notional: float,
        portfolio_value: float,
        open_positions: Dict[str, dict],
    ) -> Tuple[bool, str]:
        # Per-symbol cap
        sym_exposure = open_positions.get(signal.symbol, {}).get("market_value", 0)
        if (sym_exposure + new_notional) / portfolio_value > config.MAX_POSITION_PCT:
            return False, f"symbol {signal.symbol} exposure limit"

        # Sector cap
        sector = _universe.theme_for_symbol(signal.symbol)
        sector_exposure = sum(
            p.get("market_value", 0)
            for s, p in open_positions.items()
            if _universe.theme_for_symbol(s) == sector
        )
        if (sector_exposure + new_notional) / portfolio_value > config.MAX_SECTOR_PCT:
            return False, f"sector {sector} exposure limit"

        # Crypto cap
        if signal.is_crypto:
            crypto_exposure = sum(
                p.get("market_value", 0)
                for s, p in open_positions.items()
                if _universe.is_crypto(s)
            )
            if (crypto_exposure + new_notional) / portfolio_value > config.MAX_CRYPTO_PCT:
                return False, "crypto allocation limit"

        return True, "ok"

    # ── Exit size ─────────────────────────────────────────────────────────────

    def close_qty(self, position: dict) -> float:
        """Return the full quantity to close a position."""
        return abs(float(position.get("qty", 0)))
