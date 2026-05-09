"""
Risk Manager — position sizing, correlation caps, and exposure checks.

New features:
  - Correlation-aware position capping: limits total exposure to highly
    correlated symbols (e.g. NVDA + AMD + AVGO all count toward one cap)
  - Strategy confluence bonus: when 2+ strategies agree on a symbol,
    position size can be increased by CONFLUENCE_BONUS%.
  - VIX-based risk scaling (via regime filter's max_position_scale).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import config
from data.universe import Universe
from strategies.base import Signal
from strategies.regime_filter import RegimeFilter
from utils.logger import log

_universe = Universe()


class RiskManager:
    def __init__(self, regime_filter: RegimeFilter) -> None:
        self._regime = regime_filter

    def check_signal(
        self,
        signal: Signal,
        portfolio_value: float,
        buying_power: float,
        open_positions: Dict[str, dict],
        daily_pnl: float = 0.0,
        confluence: int = 1,
    ) -> Tuple[bool, str, float]:

        if portfolio_value <= 0:
            log.warning("REJECT {}: portfolio_value={}", signal.symbol, portfolio_value)
            return False, "portfolio_value <= 0", 0.0

        if signal.price <= 0:
            log.warning("REJECT {}: price={}", signal.symbol, signal.price)
            return False, "invalid price", 0.0

        if signal.side == "buy" and signal.symbol in open_positions:
            log.info("REJECT {}: already holding", signal.symbol)
            return False, f"already holding {signal.symbol}", 0.0

        if len(open_positions) >= config.MAX_OPEN_POSITIONS:
            log.warning("REJECT {}: max_open_positions={}", signal.symbol, len(open_positions))
            return False, "max_open_positions reached", 0.0

        if not signal.is_crypto and not self._regime.equity_trading_enabled():
            log.info("REJECT {}: equity halted by regime ({})", signal.symbol, self._regime.regime)
            return False, "equity trading halted by regime", 0.0

        # ── Crypto: simple flat-dollar fast-path ──────────────────────────────
        if signal.is_crypto:
            return self._approve_crypto(signal, portfolio_value, buying_power, open_positions)

        # ── Equity: full ATR-based sizing ─────────────────────────────────────
        if signal.atr <= 0 or not math.isfinite(signal.atr):
            log.warning("REJECT {}: bad atr={}", signal.symbol, signal.atr)
            return False, "invalid atr", 0.0

        if signal.stop_distance <= 0 or not math.isfinite(signal.stop_distance):
            log.warning("REJECT {}: bad stop_distance={}", signal.symbol, signal.stop_distance)
            return False, "invalid stop distance", 0.0

        regime_weight = self._regime.strategy_weight(signal.strategy)
        pos_scale     = self._regime.max_position_scale()

        dollar_risk    = portfolio_value * config.RISK_PER_TRADE_PCT * regime_weight
        qty_by_risk    = dollar_risk / signal.stop_distance
        max_dollars    = portfolio_value * config.MAX_POSITION_PCT * pos_scale
        qty_by_max_pos = max_dollars / signal.price
        raw_qty        = min(qty_by_risk, qty_by_max_pos)

        # Strategy confluence bonus: when 2+ strategies agree, size up
        # Apply BEFORE the min cap to avoid defeating position limits
        if confluence >= 2:
            bonus = 1.0 + config.CONFLUENCE_BONUS * (confluence - 1)
            raw_qty *= bonus
            # Re-apply max position cap after bonus
            raw_qty = min(raw_qty, qty_by_max_pos * 1.5)
            log.info("Confluence bonus {}x for {} ({} strats agree)",
                     bonus, signal.symbol, confluence)

        # Use fractional shares for high-priced names so we can size correctly
        # (Alpaca supports fractional for most large caps)
        if signal.price > 200.0:
            qty = round(raw_qty, 4)
        else:
            qty = math.floor(raw_qty)

        if qty <= 0:
            log.warning("REJECT {}: qty=0 (risk={:.2f} stop={:.4f} price={:.2f})",
                        signal.symbol, dollar_risk, signal.stop_distance, signal.price)
            return False, "computed qty <= 0", 0.0

        notional = qty * signal.price
        if notional < config.MIN_NOTIONAL:
            log.warning("REJECT {}: notional ${:.2f} < MIN ${}", signal.symbol, notional, config.MIN_NOTIONAL)
            return False, f"notional ${notional:.2f} below minimum", 0.0

        if notional > buying_power:
            available = buying_power * 0.95 / signal.price
            qty = round(available, 4) if signal.price > 200.0 else math.floor(available)
            if qty <= 0:
                log.warning("REJECT {}: insufficient buying power", signal.symbol)
                return False, "insufficient buying power", 0.0
            notional = qty * signal.price

        sym_exp = open_positions.get(signal.symbol, {}).get("market_value", 0)
        if (sym_exp + notional) / portfolio_value > config.MAX_POSITION_PCT:
            log.warning("REJECT {}: symbol exposure limit", signal.symbol)
            return False, f"symbol {signal.symbol} exposure limit", 0.0

        sector = _universe.theme_for_symbol(signal.symbol)
        sector_exp = sum(
            p.get("market_value", 0)
            for s, p in open_positions.items()
            if _universe.theme_for_symbol(s) == sector
        )
        if (sector_exp + notional) / portfolio_value > config.MAX_SECTOR_PCT:
            log.warning("REJECT {}: sector {} exposure limit ({:.1%})",
                        signal.symbol, sector, (sector_exp + notional) / portfolio_value)
            return False, f"sector {sector} exposure limit", 0.0

        total_exp = sum(p.get("market_value", 0) for p in open_positions.values())
        if (total_exp + notional) / portfolio_value > config.PORTFOLIO_HEAT_MAX:
            log.warning("REJECT {}: portfolio heat {:.1%} > {:.1%}",
                        signal.symbol, (total_exp + notional) / portfolio_value, config.PORTFOLIO_HEAT_MAX)
            return False, "portfolio heat limit reached", 0.0

        # ── Correlation cap: prevent over-concentration in correlated names ───
        correlated_themes = {"ai_tech", "crypto_equity"}
        signal_theme = _universe.theme_for_symbol(signal.symbol)
        if signal_theme in correlated_themes:
            corr_exp = sum(
                p.get("market_value", 0)
                for s, p in open_positions.items()
                if _universe.theme_for_symbol(s) in correlated_themes
            )
            if (corr_exp + notional) / portfolio_value > config.MAX_CORRELATED_PCT:
                log.warning("REJECT {}: correlated exposure {:.1%} > {:.1%} (theme={})",
                            signal.symbol, (corr_exp + notional) / portfolio_value,
                            config.MAX_CORRELATED_PCT, signal_theme)
                return False, "correlated exposure limit", 0.0

        log.info("APPROVED equity {} qty={} notional={:.2f}", signal.symbol, qty, notional)
        return True, "approved", float(qty)

    # ── Crypto fast-path ──────────────────────────────────────────────────────

    def _approve_crypto(
        self,
        signal: Signal,
        portfolio_value: float,
        buying_power: float,
        open_positions: Dict[str, dict],
    ) -> Tuple[bool, str, float]:
        """
        Crypto flat-dollar sizing. No per-crypto allocation cap —
        the overall portfolio heat limit is the only ceiling.
        Target: 3.5% of portfolio per trade.
        Floor: MIN_NOTIONAL_CRYPTO ($25). Cap: $6000.
        """
        target = max(config.MIN_NOTIONAL_CRYPTO, min(portfolio_value * 0.035, 6000.0))

        # If buying power is tight, scale down rather than reject outright.
        if target > buying_power * 0.95:
            scaled = buying_power * 0.95
            if scaled < config.MIN_NOTIONAL_CRYPTO:
                log.warning("REJECT {}: buying power {:.2f} < min {}",
                            signal.symbol, buying_power, config.MIN_NOTIONAL_CRYPTO)
                return False, "insufficient buying power", 0.0
            log.info("Scaling {} target {:.2f} → {:.2f} (BP-limited)",
                     signal.symbol, target, scaled)
            target = scaled

        total_exp = sum(p.get("market_value", 0) for p in open_positions.values())
        if (total_exp + target) / portfolio_value > config.PORTFOLIO_HEAT_MAX:
            log.warning("REJECT {}: portfolio heat {:.1%} would exceed {:.1%}",
                        signal.symbol,
                        (total_exp + target) / portfolio_value,
                        config.PORTFOLIO_HEAT_MAX)
            return False, "portfolio heat limit", 0.0

        qty = round(target / signal.price, 4)
        if qty <= 0:
            log.warning("REJECT {}: qty=0 price={}", signal.symbol, signal.price)
            return False, "qty <= 0", 0.0

        notional = qty * signal.price
        log.info("APPROVED crypto {} qty={} notional={:.2f} heat={:.1%}",
                 signal.symbol, qty, notional,
                 (total_exp + notional) / portfolio_value)
        return True, "approved", qty

    def close_qty(self, position: dict) -> float:
        return abs(float(position.get("qty", 0)))
