"""
Strategy 5 — 24/7 Crypto Momentum

Trades on hourly bars — operates around the clock.

Entry:  RSI(14) > 55
        AND close > SMA(50)
        AND MACD histogram > 0  (momentum is accelerating)
        AND close > prior 4-hour high  (micro breakout)
        AND 24h return > 1%  (not dead money)

Weekend/Overnight enhancement:
  - On Saturdays/Sundays multiply signal strength by 1.2 for BTC/ETH
    because institutional equity markets are closed — crypto vol concentrates here

Exit:   RSI < 45   OR   close < SMA(50)   OR   MACD histogram < 0 for 2 bars

Sizing note: crypto positions are capped at MAX_CRYPTO_PCT / len(active_crypto)
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd
import pytz

import config
from data.universe import Universe
from strategies.base import BaseStrategy, Signal
from utils.logger import log

ET = pytz.timezone(config.TIMEZONE)
_universe = Universe()


class CryptoMomentum(BaseStrategy):
    name = "crypto_momentum"
    timeframe_days = 60   # hourly bars, so ~60 days = ~1440 bars

    RSI_ENTRY  = 55.0
    RSI_EXIT   = 45.0
    SMA_PERIOD = 50
    MACD_FAST  = 12
    MACD_SLOW  = 26
    MACD_SIG   = 9
    ATR_PERIOD = 14
    MIN_24H_RET = 0.01   # 1% minimum 24h return

    # Weekend boost for BTC/ETH
    WEEKEND_SYMBOLS = {"BTC/USD", "ETH/USD"}
    WEEKEND_MULT    = 1.2

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []
        for sym, df in bars.items():
            if not _universe.is_crypto(sym):
                continue
            if not self._enough_bars(df, minimum=self.SMA_PERIOD + 30):
                continue
            try:
                sig = self._evaluate(sym, df)
                if sig:
                    signals.append(sig)
            except Exception as e:
                log.warning("CryptoMomentum._evaluate({}) error: {}", sym, e)
        log.debug("CryptoMomentum generated {} signals", len(signals))
        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame) -> Signal | None:
        close = df["close"]

        rsi    = self._rsi(close, self.RSI_ENTRY)
        sma50  = self._sma(close, self.SMA_PERIOD)
        macd_line, macd_signal = self._macd(close)
        macd_hist = macd_line - macd_signal
        atr    = self._atr(df, self.ATR_PERIOD)

        cur_close   = close.iloc[-1]
        cur_rsi     = rsi.iloc[-1]
        cur_sma50   = sma50.iloc[-1]
        cur_hist    = macd_hist.iloc[-1]
        prev_hist   = macd_hist.iloc[-2]
        cur_atr     = atr.iloc[-1]

        if cur_rsi < self.RSI_ENTRY:
            return None
        if cur_close < cur_sma50:
            return None
        if cur_hist <= 0:
            return None
        if prev_hist >= cur_hist:
            return None   # histogram must be accelerating

        # 24h return (24 hourly bars)
        if len(close) >= 24:
            ret_24h = close.iloc[-1] / close.iloc[-24] - 1
            if ret_24h < self.MIN_24H_RET:
                return None

        # Micro breakout: above prior 4-bar high
        if len(close) >= 5:
            prior_high = close.iloc[-5:-1].max()
            if cur_close <= prior_high:
                return None

        strength = min(1.0, 0.5 + (cur_rsi - self.RSI_ENTRY) / 50)

        # Weekend / overnight bonus
        now = datetime.now(ET)
        if sym in self.WEEKEND_SYMBOLS and now.weekday() >= 5:
            strength = min(1.0, strength * self.WEEKEND_MULT)

        stop = cur_close - config.ATR_STOP_MULT * cur_atr

        return Signal(
            symbol=sym,
            side="buy",
            strategy=self.name,
            strength=strength,
            price=cur_close,
            atr=cur_atr,
            stop_price=stop,
            is_crypto=True,
            metadata={
                "rsi":      round(cur_rsi, 2),
                "sma50":    round(cur_sma50, 4),
                "macd_hist": round(float(cur_hist), 6),
            },
        )

    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        bars: pd.DataFrame,
        position_side: str = "long",
    ) -> bool:
        if not self._enough_bars(bars, minimum=30):
            return False
        try:
            close     = bars["close"]
            rsi       = self._rsi(close, self.RSI_ENTRY)
            sma50     = self._sma(close, self.SMA_PERIOD)
            macd_line, macd_signal = self._macd(close)
            macd_hist = macd_line - macd_signal
            atr       = self._atr(bars, self.ATR_PERIOD)
            cur       = close.iloc[-1]

            if rsi.iloc[-1] < self.RSI_EXIT:
                return True
            if cur < sma50.iloc[-1]:
                return True
            if macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] < 0:
                return True
            if cur < entry_price - config.ATR_STOP_MULT * atr.iloc[-1]:
                return True
        except Exception as e:
            log.warning("CryptoMomentum.check_exit({}) error: {}", symbol, e)
        return False

    def _macd(self, series: pd.Series):
        fast = self._ema(series, self.MACD_FAST)
        slow = self._ema(series, self.MACD_SLOW)
        line = fast - slow
        signal = self._ema(line, self.MACD_SIG)
        return line, signal
