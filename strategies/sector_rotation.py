"""
Strategy 4 — Thematic Sector Rotation

Algorithm (runs weekly):
  1. Score each theme bucket by 20-day price momentum (simple return)
  2. Rank themes — overweight top 2, neutral middle, underweight/skip bottom
  3. Within each selected theme, rank individual symbols by:
       momentum_score = 0.5 × mom_20d + 0.3 × mom_5d + 0.2 × quality
       quality = Sharpe proxy = avg_daily_return / std_daily_return (20 bars)
  4. Generate BUY signals for top-N symbols in the best themes
  5. Generate SELL signals for positions in lagging themes

This is a weekly slow-burn strategy — it supplements the intraday strategies
by ensuring the portfolio is always tilted toward the best-performing themes.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

import config
from data.universe import Universe
from strategies.base import BaseStrategy, Signal
from utils.logger import log

_universe = Universe()


class SectorRotation(BaseStrategy):
    name = "sector_rotation"
    timeframe_days = 60

    MOM_LONG  = 20    # days
    MOM_SHORT = 5
    TOP_THEMES = 2    # how many themes to go long
    TOP_N_PER_THEME = 3   # top symbols per theme

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []

        # Group bars by theme
        theme_bars: Dict[str, Dict[str, pd.DataFrame]] = {}
        for sym, df in bars.items():
            theme = _universe.theme_for_symbol(sym)
            if theme not in theme_bars:
                theme_bars[theme] = {}
            theme_bars[theme][sym] = df

        # Score each theme
        theme_scores = self._score_themes(theme_bars)
        if not theme_scores:
            return signals

        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        top_themes    = [t for t, _ in sorted_themes[: self.TOP_THEMES]]
        bottom_themes = [t for t, _ in sorted_themes[self.TOP_THEMES:]]

        log.debug("SectorRotation — theme ranking: {}", sorted_themes)

        # BUY: top symbols in top themes
        for theme in top_themes:
            if theme not in theme_bars:
                continue
            ranked = self._rank_symbols(theme_bars[theme])
            for sym, score in ranked[: self.TOP_N_PER_THEME]:
                df = theme_bars[theme].get(sym)
                if df is None or not self._enough_bars(df, 25):
                    continue
                atr = self._atr(df).iloc[-1]
                cur_price = df["close"].iloc[-1]
                stop = cur_price - config.ATR_STOP_MULT * atr

                signals.append(Signal(
                    symbol=sym,
                    side="buy",
                    strategy=self.name,
                    strength=min(1.0, score * _universe.priority_for_symbol(sym)),
                    price=cur_price,
                    atr=atr,
                    stop_price=stop,
                    is_crypto=_universe.is_crypto(sym),
                    metadata={
                        "theme": theme,
                        "theme_rank": sorted_themes.index((theme, theme_scores[theme])) + 1,
                        "score": round(score, 4),
                    },
                ))

        # SELL: symbols currently held in bottom themes
        for theme in bottom_themes:
            if theme not in theme_bars:
                continue
            for sym, df in theme_bars[theme].items():
                if not self._enough_bars(df, 5):
                    continue
                cur_price = df["close"].iloc[-1]
                atr = self._atr(df).iloc[-1]
                signals.append(Signal(
                    symbol=sym,
                    side="sell",
                    strategy=self.name,
                    strength=1.0,
                    price=cur_price,
                    atr=atr,
                    stop_price=cur_price,  # immediate close
                    is_crypto=_universe.is_crypto(sym),
                    metadata={"theme": theme, "reason": "sector_rotation_exit"},
                ))

        log.info("SectorRotation: {} buy, {} sell signals",
                 sum(1 for s in signals if s.side == "buy"),
                 sum(1 for s in signals if s.side == "sell"))
        return signals

    def _score_themes(
        self, theme_bars: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for theme, sym_bars in theme_bars.items():
            theme_moms = []
            for sym, df in sym_bars.items():
                if not self._enough_bars(df, self.MOM_LONG + 5):
                    continue
                close = df["close"]
                mom   = (close.iloc[-1] / close.iloc[-self.MOM_LONG] - 1)
                priority = _universe.THEME_PRIORITY.get(theme, 1.0)
                theme_moms.append(mom * priority)
            if theme_moms:
                scores[theme] = float(np.median(theme_moms))
        return scores

    def _rank_symbols(
        self, sym_bars: Dict[str, pd.DataFrame]
    ) -> List[Tuple[str, float]]:
        rows = []
        for sym, df in sym_bars.items():
            if not self._enough_bars(df, self.MOM_LONG + 5):
                continue
            close    = df["close"]
            ret_long = close.iloc[-1] / close.iloc[-self.MOM_LONG] - 1
            ret_short = close.iloc[-1] / close.iloc[-self.MOM_SHORT] - 1
            daily_r   = close.pct_change().iloc[-self.MOM_LONG:]
            quality   = (daily_r.mean() / daily_r.std()) if daily_r.std() > 0 else 0
            score     = 0.5 * ret_long + 0.3 * ret_short + 0.2 * quality
            rows.append((sym, score))
        return sorted(rows, key=lambda x: x[1], reverse=True)

    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        bars: pd.DataFrame,
        position_side: str = "long",
    ) -> bool:
        # Exits are handled by the weekly rotation signals above
        return False
