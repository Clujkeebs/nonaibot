"""
Base types for the strategy layer.

Signal: immutable value object describing a single trade intent.
BaseStrategy: abstract class all strategies implement.

Strategies ONLY produce Signals — they never call the broker directly.
All broker interaction is handled by execution/executor.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from core.config import BotConfig


@dataclass
class Signal:
    symbol: str
    side: str               # 'buy' | 'sell'
    strategy: str           # name of strategy that generated this
    price: float            # current price at signal time
    atr: float              # current ATR (used for sizing)
    stop_price: float       # computed stop-loss price
    strength: float         # 0.0–1.0 conviction score
    is_crypto: bool
    reason: str             # human-readable explanation of why signal fired
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def stop_distance(self) -> float:
        """Dollar distance from entry to stop (always positive)."""
        if self.side == "buy":
            return max(0.0, self.price - self.stop_price)
        return 0.0


class BaseStrategy(ABC):
    """
    All strategies inherit from this.
    generate_signals: produce entry/exit signals from bar data
    check_exit: poll open positions for exit conditions
    """

    name: str = "base"
    enabled: bool = True

    @abstractmethod
    def generate_signals(
        self, bars: Dict[str, pd.DataFrame], config: BotConfig
    ) -> List[Signal]:
        """
        Produce signals from OHLCV bars.
        bars: {symbol: DataFrame with columns open/high/low/close/volume}
        Returns list of Signal objects (may be empty).
        Must NOT call the broker.
        Must NOT raise exceptions — catch internally and return [].
        """
        ...

    @abstractmethod
    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        df: pd.DataFrame,
        config: BotConfig,
    ) -> bool:
        """
        Return True if the strategy's own exit rule fires for this position.
        Called by the exit checker on every open position.
        Hard stops (ATR, trailing, time) are handled by the exit checker, not here.
        """
        ...

    # ── Shared helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _has_enough_bars(df: pd.DataFrame, minimum: int) -> bool:
        return df is not None and len(df) >= minimum
