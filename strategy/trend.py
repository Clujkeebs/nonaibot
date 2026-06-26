"""
Trend / Momentum strategy — EMA crossover with ADX filter.

Entry logic (all conditions must hold):
  1. Fast EMA (20) crosses above slow EMA (50) OR is already above with fresh momentum
  2. ADX > threshold (trend is real, not chop)
  3. Slow EMA slope is positive (trend is rising, not just a dead-cat bounce)
  4. Price is above the slow EMA (we're on the right side of the trend)

Exit logic (any condition triggers):
  1. Fast EMA crosses back below slow EMA

WHY these parameters:
  EMA20/50: classic short/medium-term crossover; well-tested across asset classes
  ADX 20 threshold: academic consensus on "trending" vs "ranging" markets
  Slope filter: prevents entering at the very end of a trend

Works on: equities with daily bars.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from core.config import BotConfig
from data.indicators import adx as calc_adx, atr as calc_atr, ema as calc_ema, ema_slope
from strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class TrendStrategy(BaseStrategy):
    name = "trend"

    def generate_signals(
        self, bars: Dict[str, pd.DataFrame], config: BotConfig
    ) -> List[Signal]:
        if not config.trend_enabled:
            return []

        signals: List[Signal] = []
        min_bars = config.trend_slow_ema + 20

        for sym, df in bars.items():
            if not self._has_enough_bars(df, min_bars):
                continue
            try:
                sig = self._evaluate(sym, df, config)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.warning("TrendStrategy._evaluate(%s): %s", sym, e)

        logger.debug("TrendStrategy: %d signals from %d symbols", len(signals), len(bars))
        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame, config: BotConfig) -> Signal | None:
        close = df["close"]

        ema_fast = calc_ema(close, config.trend_fast_ema)
        ema_slow = calc_ema(close, config.trend_slow_ema)
        sym_adx = calc_adx(df["high"], df["low"], close, config.trend_adx_period)
        cur_atr = calc_atr(df["high"], df["low"], close, config.atr_period)

        if len(ema_fast) < 3 or len(ema_slow) < 3:
            return None

        cur_fast = float(ema_fast.iloc[-1])
        cur_slow = float(ema_slow.iloc[-1])
        prev_fast = float(ema_fast.iloc[-2])
        prev_slow = float(ema_slow.iloc[-2])
        cur_adx = float(sym_adx.iloc[-1]) if len(sym_adx) > 0 else 0.0
        cur_price = float(close.iloc[-1])
        cur_atr_val = float(cur_atr.iloc[-1]) if len(cur_atr) > 0 else 0.0

        if cur_atr_val <= 0 or pd.isna(cur_atr_val):
            return None

        # Condition 1: fresh crossover OR continuation with momentum
        fresh_cross = (prev_fast <= prev_slow) and (cur_fast > cur_slow)
        continuation = (cur_fast > cur_slow) and (cur_adx > config.trend_adx_threshold * 1.2)

        if not (fresh_cross or continuation):
            return None

        # Condition 2: ADX confirms trend strength
        if cur_adx < config.trend_adx_threshold:
            return None

        # Condition 3: slow EMA must be rising (slope > min threshold)
        slope = ema_slope(ema_slow, lookback=5)
        if slope < config.trend_min_slope_pct:
            return None

        # Condition 4: price must be above slow EMA (in the trend)
        if cur_price < cur_slow:
            return None

        # Strength: ADX contribution + fresh cross bonus
        strength = 0.5 + (cur_adx - config.trend_adx_threshold) / 60.0
        if fresh_cross:
            strength += 0.15
        strength = max(0.1, min(1.0, strength))

        stop_price = cur_price - config.atr_stop_mult * cur_atr_val

        logger.info(
            "TREND BUY %s: price=%.4f ema20=%.4f ema50=%.4f adx=%.1f slope=%.4f%% fresh=%s",
            sym, cur_price, cur_fast, cur_slow, cur_adx, slope * 100, fresh_cross
        )

        return Signal(
            symbol=sym,
            side="buy",
            strategy=self.name,
            price=cur_price,
            atr=cur_atr_val,
            stop_price=stop_price,
            strength=strength,
            is_crypto=False,
            reason=(
                f"EMA{config.trend_fast_ema}/EMA{config.trend_slow_ema} "
                f"{'crossover' if fresh_cross else 'continuation'}, "
                f"ADX={cur_adx:.1f}>{config.trend_adx_threshold}, "
                f"slope={slope*100:.3f}%"
            ),
            metadata={
                "ema_fast": round(cur_fast, 4),
                "ema_slow": round(cur_slow, 4),
                "adx": round(cur_adx, 2),
                "slope_pct": round(slope * 100, 4),
                "fresh_cross": fresh_cross,
            },
        )

    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        df: pd.DataFrame,
        config: BotConfig,
    ) -> bool:
        """Exit when fast EMA crosses back below slow EMA."""
        if not self._has_enough_bars(df, config.trend_slow_ema + 5):
            return False
        try:
            close = df["close"]
            ema_fast = calc_ema(close, config.trend_fast_ema)
            ema_slow = calc_ema(close, config.trend_slow_ema)
            if len(ema_fast) < 2:
                return False
            # Bearish cross: fast dropped below slow
            return float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])
        except Exception as e:
            logger.warning("TrendStrategy.check_exit(%s): %s", symbol, e)
            return False
