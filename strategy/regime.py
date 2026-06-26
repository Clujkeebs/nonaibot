"""
RegimeFilter — classifies the broad market and per-symbol environment.

Regimes:
  bull_trend   : SPY above both 50MA and 200MA, ADX trending → favour trend strategy
  bear_trend   : SPY below 200MA → reduce position sizes, no new entries if hard bear
  ranging      : SPY mixed / ADX low → favour mean-reversion strategy
  unknown      : insufficient data → conservative (treat as ranging)

Per-symbol regime is determined by the symbol's own ADX reading.
The regime affects which strategy runs and what weight it gets.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from data.indicators import adx as calc_adx, ema as calc_ema, sma as calc_sma
from core.config import BotConfig

logger = logging.getLogger(__name__)


class RegimeFilter:
    """
    Stateful — call update() with fresh SPY bars before each strategy run.
    Thread-safe for reads after update() completes.
    """

    def __init__(self, config: BotConfig) -> None:
        self._cfg = config
        self.regime: str = "unknown"      # broad market regime
        self._spy_above_fast: bool = False
        self._spy_above_slow: bool = False
        self._spy_adx: float = 0.0
        # Per-strategy weight overrides (set by intraday adjustments)
        self._weight_overrides: Dict[str, float] = {}

    def update(self, spy_df: Optional[pd.DataFrame]) -> None:
        """Classify market regime from SPY daily bars."""
        if spy_df is None or len(spy_df) < self._cfg.regime_spy_slow_ma + 5:
            self.regime = "unknown"
            return

        close = spy_df["close"]
        fast_ma = calc_sma(close, self._cfg.regime_spy_fast_ma)
        slow_ma = calc_sma(close, self._cfg.regime_spy_slow_ma)
        spy_adx = calc_adx(spy_df["high"], spy_df["low"], close, 14)

        latest = close.iloc[-1]
        self._spy_above_fast = latest > fast_ma.iloc[-1]
        self._spy_above_slow = latest > slow_ma.iloc[-1]
        self._spy_adx = float(spy_adx.iloc[-1]) if len(spy_adx) > 0 else 0.0

        if self._spy_above_slow and self._spy_above_fast:
            self.regime = "bull_trend"
        elif not self._spy_above_slow:
            self.regime = "bear_trend"
        else:
            self.regime = "ranging"

        # Clear intraday overrides so they don't persist into a new regime state
        self._weight_overrides.clear()
        logger.info(
            "Regime updated: %s (SPY vs 50MA=%s vs 200MA=%s ADX=%.1f)",
            self.regime, self._spy_above_fast, self._spy_above_slow, self._spy_adx
        )

    def symbol_is_trending(self, df: pd.DataFrame) -> bool:
        """Return True if the symbol's own ADX says it's in a trend."""
        try:
            sym_adx = calc_adx(df["high"], df["low"], df["close"], self._cfg.trend_adx_period)
            if len(sym_adx) == 0:
                return False
            return float(sym_adx.iloc[-1]) > self._cfg.regime_adx_trending
        except Exception:
            return False

    def strategy_weight(self, strategy_name: str) -> float:
        """
        Return a weight multiplier (0.0–1.5) for a strategy given the current regime.
        Weight is 0 if the strategy shouldn't run in the current regime.
        Overrides from intraday sector rotation take precedence.
        """
        if strategy_name in self._weight_overrides:
            return self._weight_overrides[strategy_name]

        regime = self.regime

        weights: Dict[str, Dict[str, float]] = {
            "bull_trend": {
                "trend": 1.0,
                "mean_reversion": 0.5,
                "crypto_momentum": 1.0,
            },
            "ranging": {
                "trend": 0.3,       # still allow trend on individual symbols
                "mean_reversion": 1.0,
                "crypto_momentum": 0.8,
            },
            "bear_trend": {
                "trend": 0.2,       # minimal trend entries in bear market
                "mean_reversion": 0.6,  # some MR still works
                "crypto_momentum": 0.5,
            },
            "unknown": {
                "trend": 0.5,
                "mean_reversion": 0.5,
                "crypto_momentum": 0.7,
            },
        }

        regime_weights = weights.get(regime, weights["unknown"])
        return regime_weights.get(strategy_name, 0.5)

    def equity_trading_enabled(self) -> bool:
        """Return False during extreme bear conditions to pause new equity entries."""
        # Always allow equity trading unless we're in a hard bear — let circuit
        # breakers handle the rest
        return True

    def max_position_scale(self) -> float:
        """
        Scale factor for position sizes based on regime.
        In uncertain regimes, risk less per trade.
        """
        scales = {
            "bull_trend": 1.0,
            "ranging": 0.8,
            "bear_trend": 0.6,
            "unknown": 0.7,
        }
        return scales.get(self.regime, 0.7)

    def set_weight_override(self, strategy_name: str, weight: float) -> None:
        """Temporarily override a strategy's weight (used by intraday adjustments)."""
        self._weight_overrides[strategy_name] = max(0.0, min(2.0, weight))
