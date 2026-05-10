"""
Strategy 3 — ATR Volatility Breakout (Keltner-style)

Concept: price has been coiling inside a low-volatility range.
A breakout above the upper Keltner Channel signals expansion — ride the move.

Entry:  close > EMA(20) + ATR_MULT × ATR(14)   (upper Keltner)
        AND today's ATR > 1.5× 20-period average ATR  (vol is expanding)
        AND RSI(14) between 50–75               (momentum but not overbought)
        AND volume > 1.5× 20-day avg volume     (confirmation)

Exit:   close < EMA(20)   OR  trailing stop 1.5× ATR

This strategy fires in both trending AND range-breaking scenarios.
"""
from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd

import config
from data.universe import Universe
from strategies.base import BaseStrategy, Signal
from utils.logger import log

_universe = Universe()


class VolatilityBreakout(BaseStrategy):
    name = "volatility_breakout"
    timeframe_days = 60

    KC_PERIOD   = 20
    ATR_PERIOD  = 14
    ATR_MULT    = 1.5        # Keltner upper multiplier
    VOL_EXPAND  = 1.3        # ATR vs avg ATR ratio (was 1.5 — too strict)
    RSI_LO      = 50.0
    RSI_HI      = 78.0       # was 75 — let strong breakouts through
    VOL_RATIO   = 1.2        # volume vs avg (was 1.5)
    STOP_MULT   = 1.5

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []
        for sym, df in bars.items():
            if not self._enough_bars(df, minimum=self.KC_PERIOD + 20):
                continue
            try:
                sig = self._evaluate(sym, df)
                if sig:
                    signals.append(sig)
            except Exception as e:
                log.warning("VolatilityBreakout._evaluate({}) error: {}", sym, e)
        log.debug("VolatilityBreakout generated {} signals", len(signals))
        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame) -> Signal | None:
        close  = df["close"]
        volume = df.get("volume", pd.Series(dtype=float))

        ema20  = self._ema(close, self.KC_PERIOD)
        atr    = self._atr(df, self.ATR_PERIOD)
        rsi    = self._rsi(close, self.ATR_PERIOD)

        avg_atr = atr.rolling(self.KC_PERIOD).mean()

        cur_close  = close.iloc[-1]
        cur_ema    = ema20.iloc[-1]
        cur_atr    = atr.iloc[-1]
        cur_avg_atr = avg_atr.iloc[-1]
        cur_rsi    = rsi.iloc[-1]

        keltner_upper = cur_ema + self.ATR_MULT * cur_atr

        if cur_close <= keltner_upper:
            return None
        if cur_avg_atr <= 0 or not math.isfinite(cur_avg_atr):
            return None
        if cur_atr < self.VOL_EXPAND * cur_avg_atr:
            return None   # vol not expanding — likely noise
        if not (self.RSI_LO <= cur_rsi <= self.RSI_HI):
            return None

        # Volume check (optional — crypto may not have volume)
        if not volume.empty and len(volume) >= self.KC_PERIOD:
            avg_vol = volume.iloc[-self.KC_PERIOD:-1].mean()
            if avg_vol > 0 and volume.iloc[-1] < self.VOL_RATIO * avg_vol:
                return None

        # Require price to be above 20-day SMA (trend confirmation)
        ema20_val = ema20.iloc[-1]
        if cur_close < ema20_val:
            return None  # no business being in a breakout below trend

        # Previous bar must NOT have already been above upper channel
        prev_close = close.iloc[-2]
        prev_upper = ema20.iloc[-2] + self.ATR_MULT * atr.iloc[-2]
        if prev_close > prev_upper:
            return None   # already broke out — stale signal

        strength = min(1.0, 0.6 + (cur_rsi - self.RSI_LO) / 100)
        strength *= _universe.priority_for_symbol(sym)
        strength  = min(1.0, strength)

        # Stop at ATR-based distance (use AI exit optimizer's stop multiplier if available)
        stop_mult = 2.0
        if config.ENABLE_AI_LAYER and config.ENABLE_AI_EXIT_OPT and self._ai_exit_optimizer:
            stop_mult = self._ai_exit_optimizer.adjust_stop_mult(self.name, 2.0)
        stop = cur_close - stop_mult * cur_atr

        return Signal(
            symbol=sym,
            side="buy",
            strategy=self.name,
            strength=strength,
            price=cur_close,
            atr=cur_atr,
            stop_price=stop,
            is_crypto=_universe.is_crypto(sym),
            metadata={
                "keltner_upper": round(keltner_upper, 4),
                "ema20":         round(cur_ema, 4),
                "atr":           round(cur_atr, 4),
                "avg_atr":       round(cur_avg_atr, 4),
                "rsi":           round(cur_rsi, 2),
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
            close = bars["close"]
            ema20 = self._ema(close, self.KC_PERIOD)
            atr   = self._atr(bars, self.ATR_PERIOD)
            cur   = close.iloc[-1]

            if cur < ema20.iloc[-1]:
                return True
            # ATR-based exit
            if atr.iloc[-1] > 0:
                stop_mult = 2.0
                if config.ENABLE_AI_LAYER and config.ENABLE_AI_EXIT_OPT:
                    try:
                        stop_mult = self._ai_exit_optimizer.adjust_stop_mult(self.name, 2.0)
                    except Exception:
                        pass
                if cur < entry_price - stop_mult * atr.iloc[-1]:
                    return True
        except Exception as e:
            log.warning("VolatilityBreakout.check_exit({}) error: {}", symbol, e)
        return False
