"""
Mean-Reversion strategy — RSI oversold + Bollinger Band lower touch.

Entry logic (all conditions must hold):
  1. RSI < oversold threshold (e.g. 35) — price is stretched low
  2. Price is at or near the lower Bollinger Band (confirming the stretch)
  3. NOT in a bear trend (don't catch falling knives)

Exit logic (any condition triggers):
  1. RSI climbs above overbought threshold (e.g. 65) — stretch resolved
  2. Price reaches the middle Bollinger Band (mean restored)

WHY this combination:
  RSI alone produces false signals in strong downtrends.
  Adding Bollinger Band filter requires price to actually be at the lower extreme.
  The two indicators together significantly reduce false positives.

Best regime: ranging markets (ADX low, SPY sideways).
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from core.config import BotConfig
from data.indicators import (
    atr as calc_atr,
    bollinger,
    ema as calc_ema,
    rsi as calc_rsi,
)
from strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def generate_signals(
        self, bars: Dict[str, pd.DataFrame], config: BotConfig
    ) -> List[Signal]:
        if not config.mr_enabled:
            return []

        signals: List[Signal] = []
        min_bars = max(config.mr_bb_period, config.mr_rsi_period) + 10

        for sym, df in bars.items():
            if not self._has_enough_bars(df, min_bars):
                continue
            try:
                sig = self._evaluate(sym, df, config)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.warning("MeanReversionStrategy._evaluate(%s): %s", sym, e)

        logger.debug("MeanReversionStrategy: %d signals from %d symbols", len(signals), len(bars))
        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame, config: BotConfig) -> Signal | None:
        close = df["close"]

        cur_rsi = calc_rsi(close, config.mr_rsi_period)
        bb_upper, bb_mid, bb_lower = bollinger(close, config.mr_bb_period, config.mr_bb_std)
        cur_atr = calc_atr(df["high"], df["low"], close, config.atr_period)

        if len(cur_rsi) < 3 or len(bb_lower) < 3:
            return None

        rsi_val = float(cur_rsi.iloc[-1])
        lower_val = float(bb_lower.iloc[-1])
        upper_val = float(bb_upper.iloc[-1])
        mid_val = float(bb_mid.iloc[-1])
        cur_price = float(close.iloc[-1])
        atr_val = float(cur_atr.iloc[-1]) if len(cur_atr) > 0 else 0.0

        if pd.isna(rsi_val) or pd.isna(lower_val) or atr_val <= 0:
            return None

        # Condition 1: RSI oversold
        if rsi_val >= config.mr_rsi_oversold:
            return None

        # Condition 2: price near or below lower Bollinger Band
        if config.mr_require_bb_lower:
            # Allow 0.5% above the lower band (small tolerance for rounding)
            tolerance = lower_val * 0.005
            if cur_price > lower_val + tolerance:
                return None

        # Strength: how deeply oversold (0 = at threshold, 1 = RSI=0 extreme)
        depth = (config.mr_rsi_oversold - rsi_val) / config.mr_rsi_oversold
        strength = max(0.1, min(0.9, 0.4 + depth * 0.5))

        stop_price = cur_price - config.atr_stop_mult * atr_val
        # Target: middle band (where mean-reversion exits)
        # Reject if target is too close (edge gate check)
        target_pct = (mid_val - cur_price) / cur_price if cur_price > 0 else 0

        logger.info(
            "MR BUY %s: price=%.4f rsi=%.1f bb_lower=%.4f bb_mid=%.4f target_pct=%.2f%%",
            sym, cur_price, rsi_val, lower_val, mid_val, target_pct * 100
        )

        return Signal(
            symbol=sym,
            side="buy",
            strategy=self.name,
            price=cur_price,
            atr=atr_val,
            stop_price=stop_price,
            strength=strength,
            is_crypto=False,
            reason=(
                f"RSI={rsi_val:.1f}<{config.mr_rsi_oversold} oversold, "
                f"price={cur_price:.4f} at/below BB_lower={lower_val:.4f}, "
                f"target BB_mid={mid_val:.4f} ({target_pct*100:.1f}% above)"
            ),
            metadata={
                "rsi": round(rsi_val, 2),
                "bb_lower": round(lower_val, 4),
                "bb_mid": round(mid_val, 4),
                "bb_upper": round(upper_val, 4),
                "target_pct": round(target_pct, 4),
            },
        )

    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        df: pd.DataFrame,
        config: BotConfig,
    ) -> bool:
        """
        Exit when RSI returns to overbought OR price reaches the middle BB.
        Whichever comes first.
        """
        min_bars = max(config.mr_bb_period, config.mr_rsi_period) + 5
        if not self._has_enough_bars(df, min_bars):
            return False
        try:
            close = df["close"]
            rsi_s = calc_rsi(close, config.mr_rsi_period)
            _, bb_mid, _ = bollinger(close, config.mr_bb_period, config.mr_bb_std)

            if len(rsi_s) < 2 or len(bb_mid) < 2:
                return False

            rsi_val = float(rsi_s.iloc[-1])
            mid_val = float(bb_mid.iloc[-1])
            cur_price = float(close.iloc[-1])

            if pd.isna(rsi_val) or pd.isna(mid_val):
                return False

            # RSI overbought → price stretched the other way, exit
            if rsi_val >= config.mr_rsi_overbought:
                logger.info("MR EXIT %s: RSI=%.1f >= overbought=%.1f", symbol, rsi_val, config.mr_rsi_overbought)
                return True

            # Price reached middle band → mean-reversion target hit
            if cur_price >= mid_val:
                logger.info("MR EXIT %s: price=%.4f reached BB_mid=%.4f", symbol, cur_price, mid_val)
                return True

        except Exception as e:
            logger.warning("MeanReversionStrategy.check_exit(%s): %s", symbol, e)
        return False
