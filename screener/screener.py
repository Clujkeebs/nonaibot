"""
Dynamic Screener — discovers new watchlist candidates and rotates stale ones.

Screen criteria for equities (all must pass):
  1. Price > min_price (no penny stocks)
  2. Average dollar volume > min_avg_dollar_volume (liquidity)
  3. Fractionable (can trade notional amounts)
  4. 20-day return > min_return_20d (must be trending up)
  5. ATR% in range [min_atr_pct, max_atr_pct] (tradable but not meme)

Scoring: symbols that pass all filters are ranked by momentum score.
  momentum_score = 20d_return × volume_ratio × (1 + adx/100)

Every add/drop is logged to the screener_log table with the metric that triggered it.
Core watchlist symbols are NEVER touched by the screener.

Writes dynamic symbols to the state DB (not to watchlist.yaml directly —
the config.dynamic_equities/crypto fields are read from DB on each reload).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import pandas as pd

from core.config import BotConfig
from core.state import SQLiteState
from data.fetcher import DataFetcher
from data.indicators import atr as calc_atr, ema as calc_ema, sma as calc_sma
from execution.assets import AssetCache

from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


class Screener:
    def __init__(
        self,
        fetcher: DataFetcher,
        assets: AssetCache,
        state: SQLiteState,
        config: BotConfig,
    ) -> None:
        self._fetcher = fetcher
        self._assets = assets
        self._state = state
        self._cfg = config

    def run(self) -> Tuple[List[str], List[str]]:
        """
        Run full screener pass. Returns (added, removed) symbol lists.
        Updates state DB with changes.
        """
        cfg = self._cfg
        added: List[str] = []
        removed: List[str] = []

        core_set: Set[str] = set(cfg.core_equities + cfg.core_crypto)
        current_dynamic_eq = set(self._state.get_dynamic_symbols("equity"))
        current_dynamic_cr = set(self._state.get_dynamic_symbols("crypto"))

        # ── Screen equities ─────────────────────────────────────────────
        try:
            eq_add, eq_remove = self._screen_equities(
                cfg.screener_equity_pool, current_dynamic_eq, core_set
            )
            for sym, metric in eq_remove:
                self._state.remove_dynamic_symbol(sym, metric)
                removed.append(sym)
                logger.info("SCREENER DROP equity %s: %s", sym, metric)

            for sym, metric in eq_add:
                if len(current_dynamic_eq) - len(eq_remove) + len(eq_add[:eq_add.index((sym, metric)) + 1]) <= cfg.max_dynamic_slots:
                    self._state.add_dynamic_symbol(sym, "equity", metric)
                    added.append(sym)
                    logger.info("SCREENER ADD equity %s: %s", sym, metric)
        except Exception as e:
            logger.error("Screener equity scan error: %s", e)
            self._state.log_error("screener", f"equity scan: {e}")

        # ── Screen crypto ────────────────────────────────────────────────
        try:
            cr_add, cr_remove = self._screen_crypto(
                cfg.screener_crypto_pool, current_dynamic_cr, core_set
            )
            for sym, metric in cr_remove:
                self._state.remove_dynamic_symbol(sym, metric)
                removed.append(sym)
                logger.info("SCREENER DROP crypto %s: %s", sym, metric)

            for sym, metric in cr_add:
                n_current = len(current_dynamic_cr) - len(cr_remove)
                n_added_so_far = len([a for a in added if "/" in a])
                if n_current + n_added_so_far < cfg.max_dynamic_crypto_slots:
                    self._state.add_dynamic_symbol(sym, "crypto", metric)
                    added.append(sym)
                    logger.info("SCREENER ADD crypto %s: %s", sym, metric)
        except Exception as e:
            logger.error("Screener crypto scan error: %s", e)
            self._state.log_error("screener", f"crypto scan: {e}")

        if added or removed:
            logger.info(
                "Screener complete: %d added %s, %d removed %s",
                len(added), added, len(removed), removed
            )
        else:
            logger.info("Screener complete: no changes")

        self._state.mark_screener_ran()
        return added, removed

    # ── Equity screening ───────────────────────────────────────────────────────

    def _screen_equities(
        self,
        pool: List[str],
        current_dynamic: Set[str],
        core_set: Set[str],
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Screen equity pool. Returns (to_add, to_remove) lists of (symbol, metric_reason)."""
        cfg = self._cfg
        to_add: List[Tuple[str, str]] = []
        to_remove: List[Tuple[str, str]] = []

        # Fetch bars for all candidates + current dynamic symbols
        all_to_check = list(set(pool) | current_dynamic)
        if not all_to_check:
            return [], []

        bars = self._fetcher.get_stock_bars(all_to_check, TimeFrame.Day, lookback_days=60)

        # Score each symbol
        scored: List[Tuple[str, float, str]] = []  # (symbol, score, metric_reason)
        for sym in all_to_check:
            if sym in core_set or sym in cfg.veto_symbols:
                continue
            df = bars.get(sym)
            if df is None or len(df) < 25:
                continue

            result = self._score_equity(sym, df, cfg)
            if result is None:
                # Symbol failed a filter — if it's currently dynamic, flag for removal
                if sym in current_dynamic:
                    to_remove.append((sym, "failed_screener_filters"))
                continue

            score, reason = result
            scored.append((sym, score, reason))

        # Sort by score descending, pick top N that aren't already dynamic
        scored.sort(key=lambda x: x[1], reverse=True)
        slots_available = cfg.max_dynamic_slots - (len(current_dynamic) - len(to_remove))

        for sym, score, reason in scored:
            if sym in current_dynamic:
                continue  # already watching, no action needed
            if slots_available <= 0:
                break
            to_add.append((sym, f"score={score:.3f} {reason}"))
            slots_available -= 1

        return to_add, to_remove

    def _score_equity(
        self, sym: str, df: pd.DataFrame, cfg: BotConfig
    ):
        """
        Score an equity symbol. Returns (score, reason_str) or None if filters fail.
        """
        close = df["close"]
        volume = df.get("volume", pd.Series(dtype=float))
        high = df.get("high", close)
        low = df.get("low", close)

        cur_price = float(close.iloc[-1])
        if cur_price < cfg.scr_min_price:
            return None

        # Fractionable check
        if cfg.scr_require_fractionable and not self._assets.is_fractionable(sym):
            return None

        # Liquidity: average dollar volume
        if not volume.empty and len(volume) >= 20:
            avg_dv = float((close * volume).tail(20).mean())
            if avg_dv < cfg.scr_min_dollar_vol:
                return None
        else:
            return None  # no volume data → skip

        # Momentum: 20-day return
        if len(close) >= 21:
            ret_20d = float((close.iloc[-1] / close.iloc[-21]) - 1)
        else:
            return None
        if ret_20d < cfg.scr_min_return_20d:
            return None

        # ATR% filter
        if len(high) >= 15 and len(low) >= 15:
            atr_s = calc_atr(high, low, close, 14)
            if len(atr_s) > 0:
                atr_val = float(atr_s.iloc[-1])
                atr_pct = atr_val / cur_price if cur_price > 0 else 0
                if atr_pct < cfg.scr_min_atr_pct or atr_pct > cfg.scr_max_atr_pct:
                    return None
            else:
                return None
        else:
            return None

        # Momentum score: higher return × higher relative volume × not meme
        vol_ratio = 1.0
        if not volume.empty and len(volume) >= 21:
            avg_vol = float(volume.tail(21).iloc[:-1].mean())
            if avg_vol > 0:
                vol_ratio = float(volume.iloc[-1]) / avg_vol

        # 5-day return for recent momentum
        ret_5d = float((close.iloc[-1] / close.iloc[-6]) - 1) if len(close) >= 6 else 0

        score = (ret_20d * 0.6 + ret_5d * 0.4) * min(vol_ratio, 3.0)
        reason = f"ret_20d={ret_20d:.1%} vol_ratio={vol_ratio:.1f} price={cur_price:.2f}"

        return score, reason

    # ── Crypto screening ───────────────────────────────────────────────────────

    def _screen_crypto(
        self,
        pool: List[str],
        current_dynamic: Set[str],
        core_set: Set[str],
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Screen crypto pool. Returns (to_add, to_remove)."""
        cfg = self._cfg
        to_add: List[Tuple[str, str]] = []
        to_remove: List[Tuple[str, str]] = []

        all_to_check = list(set(pool) | current_dynamic)
        if not all_to_check:
            return [], []

        bars = self._fetcher.get_crypto_bars(all_to_check, TimeFrame.Hour, lookback_days=14)

        scored: List[Tuple[str, float, str]] = []
        for sym in all_to_check:
            if sym in core_set or sym in cfg.veto_symbols:
                continue
            df = bars.get(sym)
            if df is None or len(df) < 50:
                continue

            close = df["close"]
            if len(close) < 168:  # need 7 days of hourly bars
                if sym in current_dynamic:
                    to_remove.append((sym, "insufficient_data"))
                continue

            # 7-day return
            ret_7d = float((close.iloc[-1] / close.iloc[-168]) - 1)
            if ret_7d < cfg.scr_crypto_min_return_7d:
                if sym in current_dynamic:
                    to_remove.append((sym, f"ret_7d={ret_7d:.1%} below threshold"))
                continue

            score = ret_7d
            reason = f"ret_7d={ret_7d:.1%}"
            scored.append((sym, score, reason))

        scored.sort(key=lambda x: x[1], reverse=True)
        slots_available = cfg.max_dynamic_crypto_slots - (len(current_dynamic) - len(to_remove))

        for sym, score, reason in scored:
            if sym in current_dynamic:
                continue
            if slots_available <= 0:
                break
            to_add.append((sym, reason))
            slots_available -= 1

        return to_add, to_remove
