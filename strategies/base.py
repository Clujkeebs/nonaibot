"""
Strategy interface — every strategy implements BaseStrategy.

Signal flow:
    MarketData → Strategy.generate_signals() → [Signal] → RiskManager → OrderEngine
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

import config

ET = pytz.timezone(config.TIMEZONE)


@dataclass
class Signal:
    symbol: str
    side: str                    # "buy" | "sell"
    strategy: str
    strength: float              # 0.0 – 1.0 (used for sizing weight)
    price: float                 # current market price
    atr: float                   # ATR value — drives stop distance
    stop_price: float            # hard stop level
    is_crypto: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(ET))

    @property
    def stop_distance(self) -> float:
        return abs(self.price - self.stop_price)

    def __repr__(self) -> str:
        return (
            f"Signal({self.side.upper()} {self.symbol} "
            f"@ {self.price:.4f} | stop={self.stop_price:.4f} "
            f"| str={self.strength:.2f} | strat={self.strategy})"
        )


class BaseStrategy(ABC):
    """Abstract base — subclasses implement generate_signals() and check_exit()."""

    name: str = "base"

    # Override in subclasses to set preferred bar timeframe
    timeframe_days: int = 60      # how many days of history required

    def __init__(self) -> None:
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @abstractmethod
    def generate_signals(
        self, bars: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """
        Receive a dict of {symbol: OHLCV DataFrame} and return entry signals.
        DataFrames always sorted ascending by timestamp.
        """

    @abstractmethod
    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        bars: pd.DataFrame,
        position_side: str = "long",
    ) -> bool:
        """Return True if the position should be closed."""

    # ── Shared indicator helpers ──────────────────────────────────────────────

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h = df["high"]
        l = df["low"]
        c = df["close"].shift(1)
        tr = pd.concat(
            [h - l, (h - c).abs(), (l - c).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([(h - l), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        atr_s  = tr.ewm(span=period, adjust=False).mean()
        di_pos = 100 * pos_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, float("nan"))
        di_neg = 100 * neg_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, float("nan"))
        dx = (100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, float("nan")))
        return dx.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _bollinger(
        series: pd.Series, period: int = 20, std_mult: float = 2.0
    ):
        """Return (lower, middle, upper)."""
        mid = series.rolling(period).mean()
        std = series.rolling(period).std()
        return mid - std_mult * std, mid, mid + std_mult * std

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def _last(series: pd.Series, n: int = 1):
        return series.iloc[-n]

    def _enough_bars(self, df: pd.DataFrame, minimum: int = 50) -> bool:
        return df is not None and len(df) >= minimum
