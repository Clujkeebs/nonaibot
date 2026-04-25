"""
Strategy 2 — Bollinger Band Mean Reversion + RSI Oversold Filter

Entry:  close < lower Bollinger Band(20, 2σ)
        AND RSI(14) < 35
        AND close > 200-day SMA  (we only buy dips in uptrends)
        AND ADX < 30             (avoid strong downtrends)

Exit:   close > middle BB  OR  RSI > 55  OR  hard stop 1.5× ATR

Best on: liquid large-caps, ETFs. Not used for highly volatile single-asset crypto.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

import config
from data.universe import Universe
from strategies.base import BaseStrategy, Signal
from utils.logger import log

_universe = Universe()


class MeanReversion(BaseStrategy):
    name = "mean_reversion"
    timeframe_days = 120

    BB_PERIOD   = 20
    BB_STD      = 2.0
    RSI_PERIOD  = 14
    RSI_ENTRY   = 35.0
    ADX_MAX     = 30.0
    TREND_SMA   = 200
    ATR_PERIOD  = 14
    STOP_MULT   = 1.5

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []
        for sym, df in bars.items():
            if not self._enough_bars(df, minimum=self.TREND_SMA + 10):
                continue
            try:
                sig = self._evaluate(sym, df)
                if sig:
                    signals.append(sig)
            except Exception as e:
                log.warning("MeanReversion._evaluate({}) error: {}", sym, e)
        log.debug("MeanReversion generated {} signals", len(signals))
        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame) -> Signal | None:
        close = df["close"]

        bb_lower, bb_mid, _ = self._bollinger(close, self.BB_PERIOD, self.BB_STD)
        rsi    = self._rsi(close, self.RSI_PERIOD)
        sma200 = self._sma(close, self.TREND_SMA)
        adx    = self._adx(df, 14)
        atr    = self._atr(df, self.ATR_PERIOD)

        cur_close  = close.iloc[-1]
        cur_lower  = bb_lower.iloc[-1]
        cur_mid    = bb_mid.iloc[-1]
        cur_rsi    = rsi.iloc[-1]
        cur_sma200 = sma200.iloc[-1]
        cur_adx    = adx.iloc[-1]
        cur_atr    = atr.iloc[-1]

        # All conditions must hold
        if cur_close >= cur_lower:
            return None
        if cur_rsi >= self.RSI_ENTRY:
            return None
        if cur_close < cur_sma200:
            return None
        if cur_adx > self.ADX_MAX:
            return None  # in a real downtrend — skip
        if cur_atr <= 0:
            return None

        # Extra quality filter: RSI two bars ago was not also oversold
        # (we want fresh oversold, not falling-knife stuck)
        if len(rsi) >= 3 and rsi.iloc[-3] < self.RSI_ENTRY:
            return None

        # Strength inversely proportional to how far below lower BB
        deviation = (cur_lower - cur_close) / cur_atr
        strength  = min(1.0, max(0.2, 1.0 - deviation * 0.3))
        strength *= _universe.priority_for_symbol(sym)
        strength  = min(1.0, strength)

        stop = cur_close - self.STOP_MULT * cur_atr

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
                "bb_lower": round(cur_lower, 4),
                "bb_mid":   round(cur_mid, 4),
                "rsi":      round(cur_rsi, 2),
                "sma200":   round(cur_sma200, 4),
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
            bb_lower, bb_mid, _ = self._bollinger(close, self.BB_PERIOD, self.BB_STD)
            rsi  = self._rsi(close, self.RSI_PERIOD)
            atr  = self._atr(bars, self.ATR_PERIOD)
            cur  = close.iloc[-1]

            # Target: mean reversion achieved
            if cur >= bb_mid.iloc[-1]:
                return True
            # RSI recovered
            if rsi.iloc[-1] > 55:
                return True
            # Hard stop
            if cur < entry_price - self.STOP_MULT * atr.iloc[-1]:
                return True
        except Exception as e:
            log.warning("MeanReversion.check_exit({}) error: {}", symbol, e)
        return False
