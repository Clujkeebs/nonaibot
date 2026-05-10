"""
AI Position Sizer — online learner that adjusts position sizes based on
actual trade outcomes.

Reads from TradeJournal, groups closed trades by (strategy, regime),
and computes a size multiplier in [0.5, 1.5].

Logic:
  - Start at neutral 1.0 (no opinion)
  - If recent trades in this (strategy, regime) are winning → scale up
  - If recent trades are losing → scale down
  - Uses a dampened exponential moving average of win_rate and avg_r
  - Requires minimum 5 trades to form an opinion
  - Regime is bucketed: "bull", "bear", "transition", "unknown"

No external ML dependencies — pure statistics on the trade journal.
The bot works perfectly without this module; with it, position sizing
becomes adaptive.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from utils.logger import log

# Multiplier bounds
_MIN_MULT = 0.5
_MAX_MULT = 1.5
_MIN_TRADES = 5  # need at least this many to form an opinion


class AIPositionSizer:
    """Learns optimal position-sizing multipliers from trade history."""

    def __init__(self, journal) -> None:
        """
        Args:
            journal: TradeJournal instance for querying trade outcomes.
        """
        self._journal = journal
        self._cache: Dict[str, float] = {}  # key: "strategy|regime" → multiplier
        log.info("AI PositionSizer initialised")

    def size_multiplier(self, strategy: str, regime: str) -> float:
        """
        Return a position-size multiplier for the given strategy + regime.

        Returns 1.0 when:
          - Insufficient trade history (< 5 closed trades)
          - Trade journal unavailable
          - All trades are very mixed (neutral signal)
        """
        key = f"{strategy}|{regime}"

        # Return cached value (refreshed periodically by refresh())
        if key in self._cache:
            return self._cache[key]

        return 1.0

    def refresh(self, strategies: List[str], regime: str) -> None:
        """
        Recompute multipliers for all strategies in the current regime.
        Call this periodically (e.g., every 30 min or after batch of exits).
        """
        for strat in strategies:
            key = f"{strat}|{regime}"
            mult = self._compute_multiplier(strat, regime)
            self._cache[key] = mult

        if self._cache:
            log.debug(
                "PositionSizer refreshed {} strategy×regime multipliers",
                len(self._cache),
            )

    def get_all_multipliers(self) -> Dict[str, float]:
        """Return all cached multipliers for inspection."""
        return dict(self._cache)

    # ── Internal ─────────────────────────────────────────────────────────

    def _compute_multiplier(self, strategy: str, regime: str) -> float:
        """Compute size multiplier from recent closed trades."""
        trades = self._journal.get_trades(
            strategy=strategy, closed_only=True, limit=50,
        )
        if not trades or len(trades) < _MIN_TRADES:
            return 1.0

        # Filter to trades matching this regime (fuzzy match)
        regime_trades = [
            t for t in trades
            if t.get("regime", "").lower() == regime.lower()
        ]
        if len(regime_trades) < _MIN_TRADES:
            # Use all trades for this strategy as fallback
            regime_trades = trades

        if len(regime_trades) < _MIN_TRADES:
            return 1.0

        # Compute metrics with recency weighting (newer trades matter more)
        n = len(regime_trades)
        wins = 0
        r_sum = 0.0
        r_sq_sum = 0.0

        for i, t in enumerate(regime_trades):
            # Weight: most recent (i=0) gets weight=1.0, oldest gets weight=0.5
            weight = 1.0 - 0.5 * (i / max(n - 1, 1))
            is_win = t.get("is_win", 0) or 0
            r_val = t.get("r_multiple", 0) or 0.0

            wins += weight * is_win
            r_sum += weight * r_val
            r_sq_sum += weight * r_val * r_val

        total_weight = sum(
            1.0 - 0.5 * (i / max(n - 1, 1)) for i in range(n)
        )
        win_rate = wins / total_weight if total_weight > 0 else 0.5
        avg_r = r_sum / total_weight if total_weight > 0 else 0.0
        var_r = max(0, r_sq_sum / total_weight - avg_r * avg_r)
        std_r = var_r ** 0.5

        # Sharpe-like score: avg_r / std_r, scaled to [-1, 1]
        sharpe = avg_r / (std_r + 0.01)

        # Combine win_rate and sharpe into a single score
        # win_rate in [0, 1], sharpe roughly in [-2, 2] for most strategies
        score = 0.5 * (win_rate - 0.5) * 2 + 0.5 * max(-1.0, min(1.0, sharpe * 0.5))
        # score in [-1, 1]

        # Map score to multiplier: -1 → 0.5, 0 → 1.0, +1 → 1.5
        multiplier = 1.0 + score * 0.5
        multiplier = max(_MIN_MULT, min(_MAX_MULT, multiplier))

        log.debug(
            "PositionSizer: {}|{} → wr={:.1%} avgR={:.2f} sharpe={:.2f} mult={:.2f}",
            strategy, regime, win_rate, avg_r, sharpe, multiplier,
        )
        return multiplier
