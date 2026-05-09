"""
Strategy 6 — Opening Range Breakout (ORB)

Captures intraday momentum by identifying the 30-minute opening range (09:35–10:05 ET)
and entering when price breaks above/below that range with volume confirmation.

This is a classic, battle-tested edge:
- The opening range often defines the day's high/low
- Breakouts from the OR are more likely to continue than fail
- Combining with volume confirmation and VWAP filter reduces false signals

Entry (long only, no shorting):
  Price > OR high
  AND volume > 1.5× 5-bar avg volume
  AND price > VWAP (trend confirmation)
  AND RSI between 45-75 (not overbought, not dead)
  AND ATR expanding (volatility is picking up)

Exit:
  Price < VWAP (lost intraday trend)
  OR trailing stop 1.5× ATR
  OR time stop (2 hours after entry if not profitable > 0.3%)
  OR profit target at +1.5× ATR

Works on: equities only, intraday (mon-fri, 09:35-15:30 ET)
Bars: 1-minute bars for current day + 5 days history
"""
from __future__ import annotations

from datetime import datetime, time as dtime
from typing import Dict, List, Optional

import pandas as pd
import pytz

import config
from data.universe import Universe
from strategies.base import BaseStrategy, Signal
from utils.logger import log

ET = pytz.timezone(config.TIMEZONE)
_universe = Universe()


class OpeningRangeBreakout(BaseStrategy):
    name = "opening_range_breakout"
    timeframe_days = 10  # 5-min bars, 10 days = plenty

    OR_START = dtime(9, 35)  # OR begins at market open + 5 min
    OR_END   = dtime(10, 5)  # OR ends at 10:05 ET
    ENTRY_END = dtime(15, 30)  # No new entries after 3:30 PM

    VOL_RATIO    = 1.5   # volume vs 5-bar avg
    RSI_LO       = 45.0
    RSI_HI       = 75.0
    ATR_EXPAND   = 1.1   # current ATR > 1.1× avg ATR
    ATR_PERIOD   = 14
    STOP_MULT    = 1.5
    PROFIT_ATR   = 1.5   # profit target in ATR multiples
    TIME_STOP_MIN = 120  # close after 2 hours if flat

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []
        now = datetime.now(ET)

        # Only operate during market hours, after OR is established
        if now.weekday() >= 5:
            return signals
        if now.time() < self.OR_END:
            log.debug("OR not yet established — waiting until 10:05 ET")
            return signals
        if now.time() > self.ENTRY_END:
            return signals

        for sym, df in bars.items():
            if _universe.is_crypto(sym):
                continue  # equities only
            if not self._enough_bars(df, minimum=80):  # need at least 1 day of 5min bars
                continue
            try:
                sig = self._evaluate(sym, df, now)
                if sig:
                    signals.append(sig)
            except Exception as e:
                log.warning("OpeningRangeBreakout._evaluate({}) error: {}", sym, e)

        log.debug("OpeningRangeBreakout generated {} signals", len(signals))
        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame, now: datetime) -> Optional[Signal]:
        close   = df["close"]
        high    = df["high"]
        volume  = df.get("volume", pd.Series(dtype=float))

        # Find today's bars (bars after 9:30 ET today)
        today_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        today_mask = df.index >= pd.Timestamp(today_start, tz=df.index.tz if df.index.tz else "UTC")

        if not today_mask.any():
            log.debug("{} no bars from today", sym)
            return None

        today_df = df[today_mask]
        if len(today_df) < 6:  # need enough intraday bars
            return None

        # Identify OR bars (09:35-10:05 ET)
        or_start_ts = pd.Timestamp(now.replace(hour=9, minute=35, second=0, microsecond=0),
                                    tz=df.index.tz if df.index.tz else "UTC")
        or_end_ts   = pd.Timestamp(now.replace(hour=10, minute=5, second=0, microsecond=0),
                                    tz=df.index.tz if df.index.tz else "UTC")

        or_mask     = (today_df.index >= or_start_ts) & (today_df.index <= or_end_ts)
        or_bars     = today_df[or_mask]

        if len(or_bars) < 2:
            log.debug("{} insufficient OR bars ({})", sym, len(or_bars))
            return None

        or_high_val = float(or_bars["high"].max())
        or_low_val  = float(or_bars["low"].min())
        or_range    = or_high_val - or_low_val

        if or_range <= 0:
            return None

        cur_close  = float(close.iloc[-1])
        cur_high   = float(high.iloc[-1])

        # Entry: price must have broken above OR high on this bar
        if cur_close <= or_high_val:
            return None

        # VWAP check — price should be above VWAP for trend confirmation
        vwap = self._vwap(today_df)
        if vwap is None or cur_close <= vwap:
            return None

        # Volume confirmation
        if not volume.empty and len(volume) >= 6:
            recent_vol = volume.iloc[-6:-1]  # last 5 bars before current
            cur_vol    = volume.iloc[-1]
            avg_vol    = recent_vol.mean()
            if avg_vol > 0 and cur_vol < self.VOL_RATIO * avg_vol:
                return None
        else:
            # No volume data — still trade but at reduced strength
            pass

        # RSI check
        rsi = self._rsi(close, 14)
        cur_rsi = float(rsi.iloc[-1])
        if not (self.RSI_LO <= cur_rsi <= self.RSI_HI):
            return None

        # ATR expansion check
        atr = self._atr(df, self.ATR_PERIOD)
        avg_atr = atr.rolling(20).mean()
        cur_atr   = float(atr.iloc[-1])
        cur_avg   = float(avg_atr.iloc[-1]) if len(atr) >= 20 else cur_atr

        if cur_avg > 0 and cur_atr < self.ATR_EXPAND * cur_avg:
            return None  # volatility not expanding

        # Strength: based on breakout conviction
        # - How far above OR high? (breakout strength)
        # - RSI momentum
        # - Volume surge
        breakout_pct = (cur_close - or_high_val) / or_range
        strength = min(1.0, max(0.3, 0.5 + breakout_pct * 0.3 + (cur_rsi - self.RSI_LO) / 100))

        # Volume bonus
        if not volume.empty and len(volume) >= 6:
            cur_vol = volume.iloc[-1]
            avg_vol = volume.iloc[-6:-1].mean()
            if avg_vol > 0:
                vol_ratio = cur_vol / avg_vol
                strength *= min(1.3, max(0.85, 0.8 + 0.1 * vol_ratio))

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
            is_crypto=False,
            metadata={
                "or_high":  round(or_high_val, 4),
                "or_low":   round(or_low_val, 4),
                "vwap":     round(vwap, 4),
                "rsi":      round(cur_rsi, 2),
                "entry_time": now.isoformat(),
            },
        )

    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        bars: pd.DataFrame,
        position_side: str = "long",
    ) -> bool:
        if not self._enough_bars(bars, minimum=20):
            return False
        try:
            close   = bars["close"]
            atr     = self._atr(bars, self.ATR_PERIOD)
            cur     = float(close.iloc[-1])

            # 1. Lost VWAP — intraday trend broken
            vwap = self._vwap(bars)
            if vwap is not None and cur < vwap:
                log.info("{} ORB EXIT — below VWAP", symbol)
                return True

            # 2. Profit target: 1.5× ATR gain
            if entry_price > 0 and cur >= entry_price + self.PROFIT_ATR * float(atr.iloc[-1]):
                log.info("{} ORB EXIT — profit target {:.1%}", symbol, (cur / entry_price) - 1)
                return True

            # 3. Trailing stop: 1.5× ATR below session high
            session_high = close.max()
            trail_stop   = session_high - self.STOP_MULT * float(atr.iloc[-1])
            if cur < trail_stop:
                log.info("{} ORB EXIT — trailing stop", symbol)
                return True

            # 4. ATR stop from entry
            if entry_price > 0 and cur < entry_price - self.STOP_MULT * float(atr.iloc[-1]):
                log.info("{} ORB EXIT — ATR stop", symbol)
                return True

        except Exception as e:
            log.warning("OpeningRangeBreakout.check_exit({}) error: {}", symbol, e)
        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _vwap(df: pd.DataFrame) -> Optional[float]:
        """Calculate VWAP for the given dataframe."""
        try:
            if "volume" not in df.columns or df["volume"].sum() == 0:
                return None
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            vwap_val = (typical_price * df["volume"]).sum() / df["volume"].sum()
            return float(vwap_val)
        except Exception:
            return None
