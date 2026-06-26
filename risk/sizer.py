"""
PositionSizer — determines approved qty for each trade signal.

Sizing formula (ATR-based Kelly-like):
  risk_dollars = equity × risk_per_trade_pct
  qty = risk_dollars / stop_distance
  where stop_distance = atr_stop_mult × ATR

Then capped by:
  - max_position_pct × equity (no single position dominates)
  - available buying_power (can't spend what we don't have)
  - max_concurrent_positions (keep the book manageable)
  - max_crypto_allocation_pct (crypto cap to limit correlated crypto risk)
  - min_notional ($1 Alpaca rule)

Edge gate (critical for a $100 account):
  Expected profit = take_profit_ratio × stop_distance / price
  If expected_profit_pct < min_edge_pct, reject to avoid fee bleed.

Returns: (approved: bool, reason: str, qty: float)
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Tuple

from core.config import BotConfig
from strategy.base import Signal

logger = logging.getLogger(__name__)


class PositionSizer:
    def __init__(self, config: BotConfig) -> None:
        self._cfg = config

    def check_signal(
        self,
        signal: Signal,
        equity: float,
        buying_power: float,
        open_positions: Dict[str, dict],
        daily_pnl: float = 0.0,
        regime_scale: float = 1.0,
        confluence: int = 1,
    ) -> Tuple[bool, str, float]:
        """
        Approve or reject a signal, returning (approved, reason, qty).
        qty is 0 if not approved.
        """
        cfg = self._cfg

        # ── Basic guards ────────────────────────────────────────────────────
        if equity <= 0:
            return False, "equity <= 0", 0.0
        if signal.price <= 0:
            return False, "invalid price", 0.0
        if signal.side == "buy" and signal.symbol in open_positions:
            return False, f"already holding {signal.symbol}", 0.0
        if len(open_positions) >= cfg.max_concurrent_positions:
            return False, f"max_concurrent_positions={cfg.max_concurrent_positions} reached", 0.0

        # ── Crypto cap ─────────────────────────────────────────────────────
        if signal.is_crypto:
            crypto_mv = sum(
                p.get("market_value", 0) for s, p in open_positions.items()
                if "/" in s  # crypto symbols contain a slash
            )
            if crypto_mv / equity > cfg.max_crypto_allocation_pct:
                return False, f"crypto allocation {crypto_mv/equity:.1%} exceeds {cfg.max_crypto_allocation_pct:.1%}", 0.0

        # ── Portfolio heat check ───────────────────────────────────────────
        total_mv = sum(p.get("market_value", 0) for p in open_positions.values())
        if total_mv / equity > cfg.portfolio_heat_max:
            return False, f"portfolio heat {total_mv/equity:.1%} exceeds {cfg.portfolio_heat_max:.1%}", 0.0

        # ── ATR guard ──────────────────────────────────────────────────────
        if signal.atr <= 0 or not math.isfinite(signal.atr):
            return False, "invalid ATR", 0.0
        if signal.stop_distance <= 0:
            return False, "stop_distance <= 0", 0.0

        # ── Edge gate ──────────────────────────────────────────────────────
        stop_pct = signal.stop_distance / signal.price
        expected_profit_pct = cfg.take_profit_ratio * stop_pct
        min_edge = cfg.edge_crypto_min_pct if signal.is_crypto else cfg.edge_equity_min_pct
        if expected_profit_pct < min_edge:
            return (
                False,
                f"edge_gate: expected {expected_profit_pct:.2%} < min {min_edge:.2%} "
                f"(ATR={signal.atr:.4f} stop={stop_pct:.2%})",
                0.0,
            )

        # ── Position sizing ────────────────────────────────────────────────
        risk_dollars = equity * cfg.risk_per_trade_pct * regime_scale

        # Confluence bonus: scale up when multiple strategies agree
        if confluence >= 2:
            bonus = 1.0 + cfg.confluence_bonus_pct * (confluence - 1)
            risk_dollars *= bonus
            logger.debug("Confluence bonus %.2fx for %s (%d strategies agree)", bonus, signal.symbol, confluence)

        qty_by_risk = risk_dollars / signal.stop_distance

        # Cap by max position %
        max_notional = equity * cfg.max_position_pct
        qty_by_max_pos = max_notional / signal.price

        qty = min(qty_by_risk, qty_by_max_pos)

        # Cap by buying power (with 2% safety margin)
        notional = qty * signal.price
        if notional > buying_power * 0.98:
            qty = (buying_power * 0.98) / signal.price

        # Round: crypto keeps fractional precision, equity uses whole shares
        # (fractional equity is allowed via notional orders for fractionable assets,
        #  but sizer returns qty; executor decides notional vs qty order type)
        if signal.is_crypto:
            qty = round(qty, 6)
        else:
            qty = round(qty, 4)  # executor will handle whole vs fractional

        notional = qty * signal.price

        # ── Final checks ────────────────────────────────────────────────────
        if qty <= 0:
            return (
                False,
                f"qty={qty:.6f} <= 0 (risk=${risk_dollars:.2f} stop=${signal.stop_distance:.4f})",
                0.0,
            )

        if notional < cfg.min_notional:
            return False, f"notional ${notional:.2f} < min ${cfg.min_notional:.2f}", 0.0

        if notional > buying_power:
            return False, f"notional ${notional:.2f} > buying_power ${buying_power:.2f}", 0.0

        # Symbol-level cap
        sym_mv = open_positions.get(signal.symbol, {}).get("market_value", 0)
        if (sym_mv + notional) / equity > cfg.max_position_pct:
            return False, f"{signal.symbol} would exceed max_position_pct", 0.0

        logger.info(
            "APPROVED %s %s qty=%.4f notional=%.2f risk=%.2f stop_dist=%.4f edge=%.2f%%",
            signal.side, signal.symbol, qty, notional, risk_dollars,
            signal.stop_distance, expected_profit_pct * 100
        )
        return True, "approved", qty
