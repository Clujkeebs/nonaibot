"""
AI Strategy Allocator — dynamic strategy weighting based on rolling performance.

Reads from TradeJournal, evaluates each strategy's recent performance,
and produces recommended weights that shift capital toward hot strategies
and away from cold ones.

Key behavior:
  - Strategies with win_rate > 55% and avg_r > 0.3 → overweight
  - Strategies with win_rate < 35% or avg_r < -0.3 → underweight
  - Weight adjustments are dampened (max ±40% from equal weight)
  - Requires minimum 8 trades to form an opinion
  - Strategies with insufficient data keep equal weight
  - All weights sum to 1.0

This is a meta-layer: even when a strategy is underweighted, it still runs
and can generate signals — its signals just get a reduced regime weight
applied. When overperforming, it gets more capital allocation.

No external ML dependencies — pure performance statistics.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from utils.logger import log

_MIN_TRADES = 8  # need at least this many per strategy to form an opinion
_MAX_DEVIATION = 0.40  # max weight deviation from equal weight


class AIStrategyAllocator:
    """Recommends dynamic strategy weights from trade history."""

    def __init__(self, journal) -> None:
        """
        Args:
            journal: TradeJournal instance for querying trade outcomes.
        """
        self._journal = journal
        self._weights: Dict[str, float] = {}  # strategy → weight
        self._performance: Dict[str, dict] = {}  # strategy → {wr, avgR, sharpe, n}
        log.info("AI StrategyAllocator initialised")

    def get_weight(self, strategy: str) -> float:
        """
        Return the recommended weight for a strategy (0.0 to 1.0).
        Returns -1 if no recommendation is available (caller should use default).
        """
        return self._weights.get(strategy, -1.0)

    def get_all_weights(self) -> Dict[str, float]:
        """Return all current strategy weights."""
        return dict(self._weights)

    def refresh(self, strategies: List[str]) -> Dict[str, float]:
        """
        Recompute weights for all strategies. Returns updated weight dict.
        Call this periodically (e.g., every 4 hours or after daily close).
        """
        if not strategies:
            return {}

        # Gather performance per strategy
        perf: Dict[str, dict] = {}
        for strat in strategies:
            trades = self._journal.get_trades(
                strategy=strat, closed_only=True, limit=40,
            )
            if not trades or len(trades) < 3:
                perf[strat] = {"wr": 0.5, "avg_r": 0.0, "sharpe": 0.0, "n": 0}
                continue

            n = len(trades)
            wins = sum(1 for t in trades if t.get("is_win"))
            r_vals = [t.get("r_multiple", 0) or 0 for t in trades]
            avg_r = sum(r_vals) / n if n else 0.0
            var_r = max(0, sum(r * r for r in r_vals) / n - avg_r * avg_r)
            std_r = var_r ** 0.5
            sharpe = avg_r / (std_r + 0.01) if std_r > 0 else avg_r

            perf[strat] = {
                "wr": wins / n,
                "avg_r": avg_r,
                "sharpe": sharpe,
                "n": n,
            }

        self._performance = perf

        # Filter to strategies with enough data
        active = {
            s: p for s, p in perf.items()
            if p["n"] >= _MIN_TRADES
        }

        n_strats = len(strategies)
        if n_strats == 0:
            return {}

        equal_weight = 1.0 / n_strats

        if not active:
            # Not enough data for any strategy — use equal weights
            self._weights = {s: equal_weight for s in strategies}
            log.debug("StrategyAllocator: insufficient data — equal weights")
            return self._weights

        # Compute a score for each strategy with data
        scores: Dict[str, float] = {}
        for strat, p in active.items():
            # Score combines win_rate and sharpe
            wr_score = (p["wr"] - 0.5) * 2  # [-1, 1]
            sharpe_score = max(-1.0, min(1.0, p["sharpe"] * 0.5))  # [-1, 1]
            score = 0.5 * wr_score + 0.5 * sharpe_score  # [-1, 1]
            scores[strat] = score

        # Build weights: start from equal, adjust by score
        weights: Dict[str, float] = {}
        for strat in strategies:
            if strat in scores:
                deviation = scores[strat] * _MAX_DEVIATION  # [-0.4, 0.4]
                weights[strat] = equal_weight + deviation
            else:
                weights[strat] = equal_weight

        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        # Floor: no strategy below 25% of equal weight
        floor = equal_weight * 0.25
        weights = {s: max(floor, w) for s, w in weights.items()}

        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        self._weights = weights

        # Log the allocation
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        log.info(
            "StrategyAllocator refreshed: {}",
            ", ".join(f"{s}={w:.1%}" for s, w in sorted_w),
        )
        return weights

    @property
    def performance_report(self) -> Dict[str, dict]:
        """Return per-strategy performance data for logging/inspection."""
        return dict(self._performance)
