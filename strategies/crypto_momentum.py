"""
Strategy 5 — 24/7 Crypto Momentum

Trades on hourly bars — operates around the clock.

Entry:  RSI(14) > 50 AND RSI < 72
        AND close > SMA(20)
        AND close > SMA(50) (multi-timeframe trend alignment)
        AND 24h return > 2% (only enter when momentum is real)
        AND ADX(14) > 20 (trend strength confirmation)

Weekend/Overnight enhancement:
  - On Saturdays/Sundays multiply signal strength by 1.2 for BTC/ETH
    because institutional equity markets are closed — crypto vol concentrates here

Exit:   RSI < 40   OR   close < SMA(20)   OR   profit-take at +8%
        OR ADX falling below 18 (trend dying)
"""
from __future__ import annotations

import math
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
    timeframe_days = 30   # hourly bars, so ~30 days = ~720 bars

    RSI_ENTRY  = 50.0    # was 45 — too low, entering weak momentum
    RSI_EXIT   = 40.0
    RSI_MAX    = 72.0    # don't chase overbought
    SMA_FAST   = 20
    SMA_SLOW   = 50      # new — multi-timeframe trend filter
    ADX_PERIOD = 14
    ADX_MIN    = 20.0    # new — trend strength confirmation
    ADX_EXIT   = 18.0    # exit when trend weakens
    ATR_PERIOD = 14
    MOM_LOOKBACK = 24    # hours for return check
    MOM_MIN     = 0.02   # 2% minimum 24h return — was 1%

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
            if not self._enough_bars(df, minimum=self.SMA_SLOW + 10):
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

        rsi    = self._rsi(close, 14)
        sma20  = self._sma(close, self.SMA_FAST)
        sma50  = self._sma(close, self.SMA_SLOW)
        adx    = self._adx(df, self.ADX_PERIOD)
        atr    = self._atr(df, self.ATR_PERIOD)

        cur_close  = float(close.iloc[-1])
        cur_rsi    = float(rsi.iloc[-1])
        cur_sma20  = float(sma20.iloc[-1])
        cur_sma50  = float(sma50.iloc[-1])
        cur_adx    = float(adx.iloc[-1])
        cur_atr    = float(atr.iloc[-1])

        log.info("{} | RSI={:.1f} SMA20={:.4f} SMA50={:.4f} ADX={:.1f} price={:.4f}",
                 sym, cur_rsi, cur_sma20, cur_sma50, cur_adx, cur_close)

        # Entry conditions
        if cur_rsi < self.RSI_ENTRY:
            log.info("{} SKIP — RSI {:.1f} < {:.0f} (not enough momentum)", sym, cur_rsi, self.RSI_ENTRY)
            return None
        if cur_rsi > self.RSI_MAX:
            log.info("{} SKIP — RSI {:.1f} > {:.0f} (overbought)", sym, cur_rsi, self.RSI_MAX)
            return None
        if cur_close < cur_sma20:
            log.info("{} SKIP — price {:.4f} below SMA20 {:.4f}", sym, cur_close, cur_sma20)
            return None
        if cur_close < cur_sma50:
            log.info("{} SKIP — price {:.4f} below SMA50 {:.4f} (no medium-term trend)", sym, cur_close, cur_sma50)
            return None
        if cur_adx < self.ADX_MIN:
            log.info("{} SKIP — ADX {:.1f} < {:.0f} (no trend)", sym, cur_adx, self.ADX_MIN)
            return None
        if cur_atr <= 0 or not math.isfinite(cur_atr):
            return None

        # 24h return check — only enter when momentum is real
        ret_24h = 0.0
        if len(close) >= self.MOM_LOOKBACK:
            ret_24h = (cur_close / float(close.iloc[-self.MOM_LOOKBACK]) - 1)
            if ret_24h < self.MOM_MIN:
                log.info("{} SKIP — 24h return {:.1%} < {:.1%}", sym, ret_24h, self.MOM_MIN)
                return None

        # Strength: based on RSI momentum, ADX trend strength, and 24h return
        strength = min(1.0, 0.4 + (cur_rsi - self.RSI_ENTRY) / 60.0 + (cur_adx - self.ADX_MIN) / 80.0)

        # Reward strong 24h returns (capped at 1.3x)
        if len(close) >= self.MOM_LOOKBACK:
            strength *= min(1.3, 1.0 + ret_24h * 3)

        # Weekend / overnight bonus
        now = datetime.now(ET)
        if sym in self.WEEKEND_SYMBOLS and now.weekday() >= 5:
            strength = min(1.0, strength * self.WEEKEND_MULT)

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
            is_crypto=True,
            metadata={
                "rsi":    round(cur_rsi, 2),
                "sma20":  round(cur_sma20, 4),
                "sma50":  round(cur_sma50, 4),
                "adx":    round(cur_adx, 2),
            },
        )

    PROFIT_TAKE_PCT = 0.08   # close when up 8%

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
            sma20 = self._sma(close, self.SMA_FAST)
            adx   = self._adx(bars, self.ADX_PERIOD)
            atr   = self._atr(bars, self.ATR_PERIOD)
            cur   = float(close.iloc[-1])

            # Profit-take: up 8% from entry
            if entry_price > 0 and cur >= entry_price * (1 + self.PROFIT_TAKE_PCT):
                log.info("{} EXIT — profit-take {:.1%}", symbol, (cur / entry_price) - 1)
                return True
            # Momentum fading
            if float(rsi.iloc[-1]) < self.RSI_EXIT:
                log.info("{} EXIT — RSI {:.1f} < {}", symbol, float(rsi.iloc[-1]), self.RSI_EXIT)
                return True
            # Trend broken
            if cur < float(sma20.iloc[-1]):
                log.info("{} EXIT — price below SMA20", symbol)
                return True
            # Trend dying (ADX falling)
            if float(adx.iloc[-1]) < self.ADX_EXIT:
                log.info("{} EXIT — ADX {:.1f} < {} (trend dying)", symbol, float(adx.iloc[-1]), self.ADX_EXIT)
                return True
            # ATR-based hard stop (use AI exit optimizer's stop multiplier if available)
            stop_mult = 2.0
            if config.ENABLE_AI_LAYER and config.ENABLE_AI_EXIT_OPT and self._ai_exit_optimizer:
                try:
                    stop_mult = self._ai_exit_optimizer.adjust_stop_mult(self.name, 2.0)
                except Exception:
                    pass
            if cur < entry_price - stop_mult * float(atr.iloc[-1]):
                log.info("{} EXIT — stop loss hit", symbol)
                return True
        except Exception as e:
            log.warning("CryptoMomentum.check_exit({}) error: {}", symbol, e)
        return False
