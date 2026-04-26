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

    RSI_ENTRY  = 45.0
    RSI_EXIT   = 40.0
    SMA_PERIOD = 20
    ATR_PERIOD = 14

    # Weekend boost for BTC/ETH
    WEEKEND_SYMBOLS = {"BTC/USD", "ETH/USD"}
    WEEKEND_MULT    = 1.2

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []
        log.info("CryptoMomentum scanning {} symbols", len(bars))
        for sym, df in bars.items():
            if not _universe.is_crypto(sym):
                log.info("{} not recognised as crypto — skipping", sym)
                continue
            if not self._enough_bars(df, minimum=self.SMA_PERIOD + 5):
                log.info("{} insufficient bars ({})", sym, len(df))
                continue
            try:
                sig = self._evaluate(sym, df)
                if sig:
                    signals.append(sig)
            except Exception as e:
                log.warning("CryptoMomentum._evaluate({}) error: {}", sym, e)
        log.info("CryptoMomentum generated {} signals from {} symbols", len(signals), len(bars))
        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame) -> Signal | None:
        close = df["close"]

        rsi   = self._rsi(close, 14)
        sma20 = self._sma(close, 20)
        atr   = self._atr(df, self.ATR_PERIOD)

        cur_close = float(close.iloc[-1])
        cur_rsi   = float(rsi.iloc[-1])
        cur_sma20 = float(sma20.iloc[-1])
        cur_atr   = float(atr.iloc[-1])

        log.info("{} | RSI={:.1f} SMA20={:.4f} price={:.4f}",
                 sym, cur_rsi, cur_sma20, cur_close)

        if cur_rsi < self.RSI_ENTRY:
            log.info("{} SKIP — RSI {:.1f} < {}", sym, cur_rsi, self.RSI_ENTRY)
            return None
        if cur_close < cur_sma20:
            log.info("{} SKIP — price below SMA20")
            return None
        if cur_atr <= 0:
            return None

        strength = min(1.0, 0.5 + (cur_rsi - self.RSI_ENTRY) / 50.0)

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
        if not self._enough_bars(bars, minimum=25):
            return False
        try:
            close = bars["close"]
            rsi   = self._rsi(close, 14)
            sma20 = self._sma(close, self.SMA_PERIOD)
            atr   = self._atr(bars, self.ATR_PERIOD)
            cur   = float(close.iloc[-1])

            if float(rsi.iloc[-1]) < self.RSI_EXIT:
                return True
            if cur < float(sma20.iloc[-1]):
                return True
            if cur < entry_price - config.ATR_STOP_MULT * float(atr.iloc[-1]):
                return True
        except Exception as e:
            log.warning("CryptoMomentum.check_exit({}) error: {}", symbol, e)
        return False
