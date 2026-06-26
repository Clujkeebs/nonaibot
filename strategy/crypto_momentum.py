"""
Crypto Momentum strategy — EMA crossover + volume spike + RSI filter.

Runs 24/7 on hourly bars.

Entry logic (all must hold):
  1. Fast EMA (12) crosses above slow EMA (26) — momentum shift
  2. Volume ratio > 1.3× average — confirms the move is real, not noise
  3. RSI is in the "momentum zone" (40–70) — not oversold collapse, not overbought chase

Exit logic (any triggers):
  1. Fast EMA crosses back below slow EMA

WHY this for crypto:
  Crypto trends sharply and quickly. EMAs are excellent trend-following tools.
  Volume confirmation is critical — fake-outs on low volume are common.
  RSI band avoids buying crashes (RSI<40) and chasing tops (RSI>70).
  Round-trip taker fee ≈0.5% → only enter on clear momentum to earn back fees.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from core.config import BotConfig
from data.indicators import (
    atr as calc_atr,
    ema as calc_ema,
    rsi as calc_rsi,
    volume_ratio as calc_vol_ratio,
)
from strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class CryptoMomentumStrategy(BaseStrategy):
    name = "crypto_momentum"

    def generate_signals(
        self, bars: Dict[str, pd.DataFrame], config: BotConfig
    ) -> List[Signal]:
        if not config.crypto_enabled:
            return []

        signals: List[Signal] = []
        min_bars = config.crypto_slow_ema + 25

        for sym, df in bars.items():
            if not self._has_enough_bars(df, min_bars):
                continue
            try:
                sig = self._evaluate(sym, df, config)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.warning("CryptoMomentumStrategy._evaluate(%s): %s", sym, e)

        logger.debug("CryptoMomentumStrategy: %d signals from %d symbols", len(signals), len(bars))
        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame, config: BotConfig) -> Signal | None:
        close = df["close"]
        volume = df.get("volume", pd.Series(dtype=float))

        ema_fast = calc_ema(close, config.crypto_fast_ema)
        ema_slow = calc_ema(close, config.crypto_slow_ema)
        cur_rsi = calc_rsi(close, config.crypto_rsi_period)
        cur_atr = calc_atr(df["high"], df["low"], close, config.atr_period)

        if len(ema_fast) < 3 or len(ema_slow) < 3:
            return None

        cur_fast = float(ema_fast.iloc[-1])
        cur_slow = float(ema_slow.iloc[-1])
        prev_fast = float(ema_fast.iloc[-2])
        prev_slow = float(ema_slow.iloc[-2])
        rsi_val = float(cur_rsi.iloc[-1]) if len(cur_rsi) > 0 else 50.0
        atr_val = float(cur_atr.iloc[-1]) if len(cur_atr) > 0 else 0.0
        cur_price = float(close.iloc[-1])

        if atr_val <= 0 or pd.isna(atr_val) or pd.isna(rsi_val):
            return None

        # Condition 1: fresh EMA crossover (fast crossed above slow)
        fresh_cross = (prev_fast <= prev_slow) and (cur_fast > cur_slow)
        # Also allow continuation: fast already above slow but RSI freshly entered zone
        continuation = (cur_fast > cur_slow) and (config.crypto_rsi_min <= rsi_val <= config.crypto_rsi_max)

        if not (fresh_cross or continuation):
            return None

        # Condition 2: volume spike confirms the move
        vol_ok = True
        if not volume.empty and len(volume) >= 21:
            try:
                vol_r = calc_vol_ratio(volume, 20)
                cur_vol_ratio = float(vol_r.iloc[-1])
                if not pd.isna(cur_vol_ratio):
                    vol_ok = cur_vol_ratio >= config.crypto_vol_ratio_min
            except Exception:
                pass

        if not vol_ok:
            return None

        # Condition 3: RSI in momentum zone (not crashed, not overbought)
        if not (config.crypto_rsi_min <= rsi_val <= config.crypto_rsi_max):
            return None

        # Strength: RSI position within the zone + fresh cross bonus
        zone_width = config.crypto_rsi_max - config.crypto_rsi_min
        rsi_position = (rsi_val - config.crypto_rsi_min) / zone_width  # 0 at bottom, 1 at top
        # Sweet spot is middle of zone: RSI ~55
        strength = 0.4 + 0.4 * (1 - abs(rsi_position - 0.5) * 2)
        if fresh_cross:
            strength += 0.15
        strength = max(0.1, min(1.0, strength))

        stop_price = cur_price - config.atr_stop_mult * atr_val

        logger.info(
            "CRYPTO MOM BUY %s: price=%.6f ema12=%.6f ema26=%.6f rsi=%.1f fresh=%s",
            sym, cur_price, cur_fast, cur_slow, rsi_val, fresh_cross
        )

        return Signal(
            symbol=sym,
            side="buy",
            strategy=self.name,
            price=cur_price,
            atr=atr_val,
            stop_price=stop_price,
            strength=strength,
            is_crypto=True,
            reason=(
                f"EMA{config.crypto_fast_ema}/EMA{config.crypto_slow_ema} "
                f"{'crossover' if fresh_cross else 'continuation'}, "
                f"RSI={rsi_val:.1f} in [{config.crypto_rsi_min},{config.crypto_rsi_max}], "
                f"volume confirmed"
            ),
            metadata={
                "ema_fast": round(cur_fast, 6),
                "ema_slow": round(cur_slow, 6),
                "rsi": round(rsi_val, 2),
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
        if not self._has_enough_bars(df, config.crypto_slow_ema + 5):
            return False
        try:
            close = df["close"]
            ema_fast = calc_ema(close, config.crypto_fast_ema)
            ema_slow = calc_ema(close, config.crypto_slow_ema)
            if len(ema_fast) < 2:
                return False
            bearish_cross = float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])
            if bearish_cross:
                logger.info("CRYPTO EXIT %s: EMA%d crossed below EMA%d",
                            symbol, config.crypto_fast_ema, config.crypto_slow_ema)
            return bearish_cross
        except Exception as e:
            logger.warning("CryptoMomentumStrategy.check_exit(%s): %s", symbol, e)
            return False
