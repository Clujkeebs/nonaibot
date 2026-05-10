"""
Strategy 1 — Dual EMA Crossover + ADX Trend Filter

Entry:  EMA(9) crosses above EMA(21)  AND  ADX(14) > 25  AND  close > EMA(50)
        → trend is real, not a whipsaw, and we are above the long-term mean

Strength bonus: weight higher for AI_TECH and CLEAN_ENERGY symbols (see Universe priority)

Exit:   EMA(9) crosses below EMA(21)  OR  trailing stop (2× ATR below running high)

Works on: equities + crypto (hourly bars for crypto, daily for equities)
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

import config
from data.universe import Universe
from strategies.base import BaseStrategy, Signal
from utils.logger import log

_universe = Universe()


class TrendFollowing(BaseStrategy):
    name = "trend_following"
    timeframe_days = 80

    # Tunable params
    FAST_EMA   = 9
    SLOW_EMA   = 21
    TREND_EMA  = 50
    ADX_PERIOD = 14
    ADX_MIN    = 20.0
    ATR_PERIOD = 14

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []

        for sym, df in bars.items():
            if not self._enough_bars(df, minimum=self.TREND_EMA + 10):
                continue
            try:
                sig = self._evaluate(sym, df)
                if sig:
                    signals.append(sig)
            except Exception as e:
                log.warning("TrendFollowing._evaluate({}) error: {}", sym, e)

        log.debug("TrendFollowing generated {} signals", len(signals))
        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame) -> Signal | None:
        close  = df["close"]
        volume = df.get("volume", pd.Series(dtype=float))

        ema_fast   = self._ema(close, self.FAST_EMA)
        ema_slow   = self._ema(close, self.SLOW_EMA)
        ema_trend  = self._ema(close, self.TREND_EMA)
        adx        = self._adx(df, self.ADX_PERIOD)
        atr        = self._atr(df, self.ATR_PERIOD)

        # ── Entry conditions ────────────────────────────────────────────────
        cur_fast   = ema_fast.iloc[-1]
        cur_slow   = ema_slow.iloc[-1]
        prev_fast  = ema_fast.iloc[-2]
        prev_slow  = ema_slow.iloc[-2]
        cur_adx    = adx.iloc[-1]
        cur_price  = close.iloc[-1]
        cur_trend  = ema_trend.iloc[-1]
        cur_atr    = atr.iloc[-1]

        # Fresh crossover: fast was below slow, now above
        fresh_cross = (prev_fast <= prev_slow) and (cur_fast > cur_slow)
        # Continuation: fast already above slow — ADX just needs to exceed minimum
        continuation = (cur_fast > cur_slow) and (cur_adx > self.ADX_MIN)

        above_trend = cur_price > cur_trend
        adx_strong  = cur_adx > self.ADX_MIN

        if not above_trend or not adx_strong:
            return None
        if not (fresh_cross or continuation):
            return None

        # Require momentum: price must be above EMA21 for continuation trades too
        if cur_price < cur_slow:
            return None

        # Volume bonus on fresh crossovers (no longer a hard filter — too restrictive
        # on slow days). We just reward high-volume breakouts with a strength boost.
        vol_boost = 1.0
        if fresh_cross and not volume.empty and len(volume) >= 20 and not _universe.is_crypto(sym):
            avg_vol = volume.iloc[-20:-1].mean()
            cur_vol = volume.iloc[-1]
            if avg_vol > 0:
                vol_ratio = cur_vol / avg_vol
                # Reward strong volume up to 1.3x strength bonus, mild dampener
                # (0.85x) for very low volume — don't kill the signal outright.
                vol_boost = max(0.85, min(1.3, 0.7 + 0.3 * vol_ratio))

        # Strength: 0.5 base + ADX contribution + theme priority + volume bonus
        strength = min(1.0, 0.5 + (cur_adx - self.ADX_MIN) / 50.0)
        strength *= _universe.priority_for_symbol(sym) * vol_boost
        strength  = min(1.0, strength)

        # Use AI exit optimizer's stop multiplier for adaptive stops
        stop_mult = 2.0
        if config.ENABLE_AI_LAYER and config.ENABLE_AI_EXIT_OPT and self._ai_exit_optimizer:
            stop_mult = self._ai_exit_optimizer.adjust_stop_mult(self.name, 2.0)
        stop = cur_price - stop_mult * cur_atr

        return Signal(
            symbol=sym,
            side="buy",
            strategy=self.name,
            strength=strength,
            price=cur_price,
            atr=cur_atr,
            stop_price=stop,
            is_crypto=_universe.is_crypto(sym),
            metadata={
                "ema_fast": round(cur_fast, 4),
                "ema_slow": round(cur_slow, 4),
                "adx": round(cur_adx, 2),
                "fresh_cross": fresh_cross,
            },
        )

    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        bars: pd.DataFrame,
        position_side: str = "long",
    ) -> bool:
        if not self._enough_bars(bars, minimum=25):
            return False
        try:
            close    = bars["close"]
            ema_fast = self._ema(close, self.FAST_EMA)
            ema_slow = self._ema(close, self.SLOW_EMA)
            atr      = self._atr(bars, self.ATR_PERIOD)
            cur_price = close.iloc[-1]

            # Bearish EMA cross
            if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                return True

            # ATR-based trailing stop
            if atr.iloc[-1] > 0:
                stop_mult = 2.0
                if config.ENABLE_AI_LAYER and config.ENABLE_AI_EXIT_OPT:
                    try:
                        stop_mult = self._ai_exit_optimizer.adjust_stop_mult(self.name, 2.0)
                    except Exception:
                        pass
                trail_stop = close.rolling(len(close)).max().iloc[-1] - stop_mult * atr.iloc[-1]
                if cur_price < trail_stop:
                    return True
        except Exception as e:
            log.warning("TrendFollowing.check_exit({}) error: {}", symbol, e)
        return False
