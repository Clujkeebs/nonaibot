"""
AI Exit Optimizer — learns optimal stop-loss and take-profit levels from
trade history.

Reads from TradeJournal, analyzes completed trades per strategy, and
recommends adjusted exit parameters:

  - ATR stop multiplier: how many ATRs to use for the initial stop
  - Take-profit ratio: profit_target / stop_distance
  - Time stop days: optimal max holding period

Logic:
  - For losing trades: what was the average/median adverse excursion?
    → Set stop slightly wider than median loser's drawdown.
  - For winning trades: what was the average favorable excursion?
    → Set take-profit to capture the median winner without cutting it short.
  - Recommends holding time based on when winners typically peak.

All recommendations are smoothed toward defaults to prevent overfitting.
No external ML dependencies.

Outputs are used by the engine to override strategy-level stop parameters
before they're passed to RiskManager.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from utils.logger import log

_MIN_TRADES = 6  # need at least this many closed trades per strategy


class AIExitOptimizer:
    """Learns optimal exit parameters per strategy from trade history."""

    def __init__(self, journal) -> None:
        self._journal = journal
        # strategy → {atr_stop_mult, tp_ratio, time_stop_days}
        self._params: Dict[str, dict] = {}
        log.info("AI ExitOptimizer initialised")

    def adjust_stop_mult(
        self, strategy: str, default_mult: float = 1.5,
    ) -> float:
        """
        Return an adjusted ATR stop multiplier for the strategy.
        Falls back to default when insufficient data.
        """
        p = self._params.get(strategy, {})
        return p.get("atr_stop_mult", default_mult)

    def adjust_tp_ratio(
        self, strategy: str, default_ratio: float = 2.0,
    ) -> float:
        """
        Return an adjusted take-profit ratio (R:R) for the strategy.
        """
        p = self._params.get(strategy, {})
        return p.get("tp_ratio", default_ratio)

    def adjust_time_stop_days(
        self, strategy: str, default_days: int = 15,
    ) -> int:
        """
        Return an adjusted time-stop in days for the strategy.
        """
        p = self._params.get(strategy, {})
        return p.get("time_stop_days", default_days)

    def get_all_params(self) -> Dict[str, dict]:
        """Return all current per-strategy exit parameters."""
        return dict(self._params)

    def refresh(self, strategies: List[str]) -> None:
        """
        Recompute exit parameters for all strategies.
        Call this periodically (e.g., after daily close).
        """
        for strat in strategies:
            self._params[strat] = self._compute_params(strat)

        if self._params:
            log.debug("ExitOptimizer refreshed {} strategies", len(self._params))

    # ── Internal ─────────────────────────────────────────────────────────

    def _compute_params(self, strategy: str) -> dict:
        """Compute exit params from closed trades for a strategy."""
        trades = self._journal.get_trades(
            strategy=strategy, closed_only=True, limit=60,
        )
        if not trades or len(trades) < _MIN_TRADES:
            # Return neutral defaults
            return {"atr_stop_mult": 1.5, "tp_ratio": 2.0, "time_stop_days": 15}

        losers = [t for t in trades if not t.get("is_win")]
        winners = [t for t in trades if t.get("is_win")]

        # ── ATR stop multiplier ──────────────────────────────────────────
        # Look at loser R-multiples: how negative did they go?
        if losers:
            loser_r = sorted([
                t.get("r_multiple", 0) or 0 for t in losers
            ])
            # Median adverse excursion (absolute value)
            med_loser = abs(loser_r[len(loser_r) // 2]) if loser_r else 1.5
            # We want the stop to be wider than the median loser's drawdown
            # but not excessively wide. Target: 1.3× median loser |R|, bounded.
            raw_stop = med_loser * 1.3
            # Smooth toward default (1.5) with 70% weight on data, 30% on prior
            atr_stop = 0.7 * raw_stop + 0.3 * 1.5
            atr_stop = max(1.0, min(3.0, atr_stop))
        else:
            atr_stop = 1.5

        # ── Take-profit ratio ────────────────────────────────────────────
        if winners:
            winner_r = sorted([
                t.get("r_multiple", 0) or 0 for t in winners
            ])
            # Median favorable excursion
            med_winner = winner_r[len(winner_r) // 2] if winner_r else 2.0
            # Target tp ratio at 85% of median winner R (don't be greedy)
            raw_tp = med_winner * 0.85
            # Smooth toward default (2.0)
            tp_ratio = 0.7 * raw_tp + 0.3 * 2.0
            tp_ratio = max(1.0, min(4.0, tp_ratio))
        else:
            tp_ratio = 2.0

        # ── Time stop days ───────────────────────────────────────────────
        all_days = [
            t.get("holding_days", 0) or 0 for t in trades
            if (t.get("holding_days") or 0) > 0
        ]
        if len(all_days) >= _MIN_TRADES:
            sorted_days = sorted(all_days)
            # 75th percentile holding time (winners hold longer than losers)
            p75 = sorted_days[int(len(sorted_days) * 0.75)]
            # Time stop at 1.5× the 75th percentile holding time
            raw_time = p75 * 1.5
            time_stop = int(0.7 * raw_time + 0.3 * 15)
            time_stop = max(5, min(30, time_stop))
        else:
            time_stop = 15

        log.debug(
            "ExitOptimizer: {} → stop={:.1f} tp={:.1f} time={}d ({} trades)",
            strategy, atr_stop, tp_ratio, time_stop, len(trades),
        )
        return {
            "atr_stop_mult": round(atr_stop, 1),
            "tp_ratio": round(tp_ratio, 1),
            "time_stop_days": time_stop,
        }
