"""
Regime Filter — adjusts strategy weights based on detected market regime.

No VIX API needed. We derive a volatility proxy from SPY daily bars:
  - realized_vol = std of 20-day daily returns × sqrt(252)
  - trend_state  = SPY close vs SMA(200)

Additionally applies VIX-based risk scaling: when VIX is elevated,
position sizes are automatically reduced across all strategies.

VIX scaling (uses realized vol as VIX proxy):
  vol < 15% → 100% sizing (low vol, full size)
  vol 15-25% → 90% sizing (normal)
  vol 25-35% → 75% sizing (elevated)
  vol > 35% → 60% sizing (high vol, defensive)

Regime matrix:
  BULL_LOW_VOL   → all strategies at full weight
  BULL_HIGH_VOL  → reduce trend-following weight, increase mean-reversion
  BEAR_LOW_VOL   → only mean-reversion and sector rotation
  BEAR_HIGH_VOL  → all equity strategies halted; crypto-only, reduced size
  UNKNOWN        → conservative (0.8× weights)

The regime is recomputed every hour and cached.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pandas as pd

import config
from strategies.base import BaseStrategy
from utils.logger import log


def _validate_config() -> None:
    """Sanity-check config values at startup — fail fast on bad parameters."""
    issues: list[str] = []
    if config.MAX_POSITION_PCT > 0.20:
        issues.append(f"MAX_POSITION_PCT={config.MAX_POSITION_PCT} is dangerously high (max 0.20)")
    if config.RISK_PER_TRADE_PCT > 0.03:
        issues.append(f"RISK_PER_TRADE_PCT={config.RISK_PER_TRADE_PCT} risks blowing up (max 0.03)")
    if config.HARD_STOP_PCT < 0.01:
        issues.append(f"HARD_STOP_PCT={config.HARD_STOP_PCT} is too tight — set at least 0.01")
    if config.COOLDOWN_HOURS_EQUITY < 1:
        issues.append(f"COOLDOWN_HOURS_EQUITY={config.COOLDOWN_HOURS_EQUITY} is too short — min 1h")
    if config.MAX_OPEN_POSITIONS < 5:
        issues.append(f"MAX_OPEN_POSITIONS={config.MAX_OPEN_POSITIONS} is too low — min 5")
    if issues:
        for issue in issues:
            log.error("CONFIG VALIDATION FAILED: {}", issue)
        raise ValueError(f"Config validation failed: {'; '.join(issues)}")


_validate_config()


class Regime(str, Enum):
    BULL_LOW_VOL  = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL  = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    UNKNOWN       = "unknown"


@dataclass
class RegimeWeights:
    trend_following:            float = 1.0
    mean_reversion:             float = 1.0
    volatility_breakout:        float = 1.0
    sector_rotation:            float = 1.0
    crypto_momentum:            float = 1.0
    opening_range_breakout:     float = 1.0
    max_position_scale:         float = 1.0   # multiply MAX_POSITION_PCT by this


_REGIME_TABLE: Dict[Regime, RegimeWeights] = {
    Regime.BULL_LOW_VOL:  RegimeWeights(1.0,  0.6,  1.0,  1.0,  1.0,  1.0,  1.0),
    Regime.BULL_HIGH_VOL: RegimeWeights(0.7,  1.2,  1.2,  0.8,  1.0,  0.8,  0.9),
    Regime.BEAR_LOW_VOL:  RegimeWeights(0.5,  1.2,  0.6,  1.0,  0.8,  0.6,  0.8),
    Regime.BEAR_HIGH_VOL: RegimeWeights(0.4,  0.8,  0.4,  0.6,  0.8,  0.4,  0.7),
    Regime.UNKNOWN:       RegimeWeights(0.8,  0.8,  0.8,  0.8,  0.9,  0.8,  0.9),
}

VOL_HIGH_THRESHOLD = 0.35   # 35% annualised vol → "high vol"
VOL_LOW_THRESHOLD  = 0.18


class RegimeFilter:
    """Detect market regime; expose weight multipliers to the engine."""

    def __init__(self) -> None:
        self._regime: Regime = Regime.UNKNOWN
        self._weights: RegimeWeights = _REGIME_TABLE[Regime.UNKNOWN]
        self._realized_vol: float = 0.0

    @property
    def regime(self) -> Regime:
        return self._regime

    @property
    def weights(self) -> RegimeWeights:
        return self._weights

    def update(self, spy_bars: Optional[pd.DataFrame]) -> None:
        if spy_bars is None or len(spy_bars) < 210:
            log.warning("RegimeFilter: insufficient SPY bars — using UNKNOWN regime")
            self._regime  = Regime.UNKNOWN
            self._weights = _REGIME_TABLE[Regime.UNKNOWN]
            return

        close  = spy_bars["close"]
        sma200 = close.rolling(200).mean()
        daily_ret = close.pct_change()
        realized_vol = daily_ret.iloc[-20:].std() * (252 ** 0.5)

        self._realized_vol = float(realized_vol)
        is_bull  = float(close.iloc[-1]) > float(sma200.iloc[-1])
        is_high_vol = realized_vol > VOL_HIGH_THRESHOLD

        if is_bull and not is_high_vol:
            self._regime = Regime.BULL_LOW_VOL
        elif is_bull and is_high_vol:
            self._regime = Regime.BULL_HIGH_VOL
        elif not is_bull and not is_high_vol:
            self._regime = Regime.BEAR_LOW_VOL
        else:
            self._regime = Regime.BEAR_HIGH_VOL

        self._weights = _REGIME_TABLE[self._regime]
        log.info(
            "Regime updated → {} | SPY={:.2f} SMA200={:.2f} Vol={:.1%}",
            self._regime,
            float(close.iloc[-1]),
            float(sma200.iloc[-1]),
            realized_vol,
        )

    def strategy_weight(self, strategy_name: str) -> float:
        # Check for AI override before returning default weight
        if hasattr(self, "_ai_weight_override") and strategy_name in self._ai_weight_override:
            return self._ai_weight_override[strategy_name]
        return getattr(self._weights, strategy_name, 1.0)

    def _override_weight(self, strategy_name: str, weight: float) -> None:
        """AI-driven override: temporarily adjust a strategy's weight."""
        if not hasattr(self, "_ai_weight_override"):
            self._ai_weight_override: Dict[str, float] = {}
        self._ai_weight_override[strategy_name] = weight
        log.debug("Regime AI override: {} weight → {:.3f}", strategy_name, weight)

    def equity_trading_enabled(self) -> bool:
        return True  # weights reduce size in bad regimes; never hard-block

    def max_position_scale(
        self,
        exit_optimizer=None,
    ) -> float:
        """
        Return position-size multiplier combining regime weight + VIX-based
        volatility scaling. When vol is high, we trade smaller.
        """
        regime_scale = self._weights.max_position_scale
        if not config.VIX_RISK_SCALE:
            return regime_scale

        # VIX scaling based on realized vol
        rv = self._realized_vol
        if rv < 0.15:
            vol_scale = 1.0
        elif rv < 0.25:
            vol_scale = 0.90
        elif rv < 0.35:
            vol_scale = 0.75
        else:
            vol_scale = 0.60

        scale = regime_scale * vol_scale

        # ── AI Exit Optimizer: override time_stop_days if available ─────────
        if (
            config.ENABLE_AI_LAYER and config.ENABLE_AI_EXIT_OPT
            and exit_optimizer is not None
        ):
            # Use the average time_stop_days across strategies as regime heat signal
            all_params = exit_optimizer.get_all_params()
            if all_params:
                days_values = [p.get("time_stop_days", 15) for p in all_params.values()]
                avg_days = sum(days_values) / len(days_values)
                # If avg recommended time_stop < 10 days → market is hostile, reduce further
                if avg_days < 10:
                    scale = min(scale, 0.75)
                # If avg recommended time_stop > 20 days → market is favorable, slight boost
                elif avg_days > 20:
                    scale = min(1.0, scale * 1.05)

        return scale
