# ── Free AI Technical Scorecard ──────────────────────────────────────────────
# Composite technical score combining RSI, MACD, Bollinger, volume, trend.
# No external APIs needed — purely from price/volume data.
#
# Scores range from 0-100 and are used to:
#   1. Filter weak signals (score < 40 = skip entry)
#   2. Adjust position size (score > 70 = increase by 20%)
#   3. Rank symbols for sector rotation

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import log


class TechnicalScorecard:
    def __init__(self) -> None:
        self._scores: Dict[str, float] = {}
        log.info('TechnicalScorecard initialised')

    def score_symbol(self, df: pd.DataFrame, symbol: str = '') -> float:
        if df is None or len(df) < 30:
            return 50.0  # neutral for insufficient data

        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(dtype=float))

            score = 0.0
            components = 0

            # ── RSI (14) ───────────────────────────────────────────────────────
            try:
                delta = close.diff()
                gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
                rs = gain / loss.replace(0, 1e-9)
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
                if rsi < 30:
                    score += 25  # oversold = bullish potential
                elif rsi > 70:
                    score += 10  # overbought = less attractive
                else:
                    score += 17  # neutral zone
                components += 1
            except Exception:
                score += 15
                components += 1

            # ── MACD momentum ───────────────────────────────────────────────────
            try:
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                macd_hist = macd.iloc[-1] - signal.iloc[-1]
                prev_hist = (macd.iloc[-2] - signal.iloc[-2])
                macd_change = macd_hist - prev_hist
                if macd_hist > 0 and macd_change > 0:
                    score += 25  # bullish MACD cross
                elif macd_hist > 0:
                    score += 18
                elif macd_hist < 0 and macd_change < 0:
                    score += 5   # bearish MACD
                else:
                    score += 12
                components += 1
            except Exception:
                score += 12
                components += 1

            # ── Bollinger Band position ─────────────────────────────────────────
            try:
                sma20 = close.rolling(20).mean()
                std20 = close.rolling(20).std()
                bb_upper = sma20 + 2 * std20
                bb_lower = sma20 - 2 * std20
                bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-9
                bb_pct = (close.iloc[-1] - bb_lower.iloc[-1]) / bb_range
                if bb_pct < 0.2:
                    score += 25  # near lower BB = potential bounce
                elif bb_pct > 0.8:
                    score += 10  # near upper BB = less upside
                else:
                    score += 17
                components += 1
            except Exception:
                score += 15
                components += 1

            # ── Volume trend ────────────────────────────────────────────────────
            try:
                if not volume.empty and len(volume) >= 20:
                    avg_vol = volume.iloc[-20:].mean()
                    cur_vol = volume.iloc[-1]
                    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
                    if vol_ratio > 1.5:
                        score += 25  # high volume confirmation
                    elif vol_ratio > 1.0:
                        score += 17
                    else:
                        score += 10
                    components += 1
            except Exception:
                score += 12
                components += 1

            # ── Trend alignment (price vs SMA50 vs SMA200) ──────────────────────
            try:
                sma50 = close.rolling(50).mean()
                sma200 = close.rolling(200).mean()
                cur = close.iloc[-1]
                if len(sma200) >= 200 and len(sma50) >= 50:
                    if cur > sma50.iloc[-1] > sma200.iloc[-1]:
                        score += 25  # strong uptrend
                    elif cur > sma50.iloc[-1]:
                        score += 18
                    elif cur < sma50.iloc[-1] < sma200.iloc[-1]:
                        score += 5   # strong downtrend
                    else:
                        score += 12
                    components += 1
            except Exception:
                score += 15
                components += 1

            # ── Momentum (5-day return) ─────────────────────────────────────────
            try:
                if len(close) >= 5:
                    mom5 = (close.iloc[-1] / close.iloc[-5] - 1) * 100
                    if mom5 > 3:
                        score += 25
                    elif mom5 > 0:
                        score += 17
                    elif mom5 < -3:
                        score += 5
                    else:
                        score += 12
                    components += 1
            except Exception:
                score += 12
                components += 1

            if components == 0:
                return 50.0

            # Normalize to 0-100
            normalized = (score / (components * 25)) * 100
            return float(np.clip(normalized, 0, 100))
        except Exception as e:
            log.warning('TechnicalScorecard.score_symbol(%s) error: %s', symbol, e)
            return 50.0

    def update_scores(self, bars_dict: Dict[str, pd.DataFrame]) -> None:
        for sym, df in bars_dict.items():
            self._scores[sym] = self.score_symbol(df, sym)
        log.debug('TechnicalScorecard: updated %d symbols', len(bars_dict))

    def get_score(self, symbol: str) -> float:
        return self._scores.get(symbol, 50.0)

    def get_multiplier(self, symbol: str) -> float:
        score = self.get_score(symbol)
        if score >= 70:
            return 1.2  # strong technicals = increase size
        elif score <= 40:
            return 0.7  # weak technicals = reduce size
        return 1.0

    def get_all_scores(self) -> Dict[str, float]:
        return dict(self._scores)

    def top_n(self, n: int = 5, min_score: float = 40.0) -> List[Tuple[str, float]]:
        filtered = [(s, sc) for s, sc in self._scores.items() if sc >= min_score]
        return sorted(filtered, key=lambda x: x[1], reverse=True)[:n]