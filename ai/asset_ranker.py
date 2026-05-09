"""
Optional AI Layer — Local Asset Ranker

Uses a lightweight scikit-learn RandomForest to rank assets by predicted
next-bar return. Features are entirely price-derived — no external data needed.

Requirements (only if ENABLE_AI_LAYER=true):
  pip install scikit-learn

The ranker adjusts signal strength scores before they reach the RiskManager.
If unavailable, signal strengths pass through unchanged (1.0 multiplier).

Feature set (per symbol, last 20 bars):
  - 5-day momentum
  - 20-day momentum
  - RSI(14) normalised
  - ATR% (ATR / price)
  - BB%  (position within Bollinger bands)
  - Volume ratio (today / 20-day avg)
  - Sector relative strength (vs SPY)
  - Trend strength (ADX normalised)
  - Short-term volatility percentile
  - Day-of-week (cyclical encoding)
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import log

_MODEL_CACHE = "ai/ranker_model.pkl"
_MIN_BARS    = 60


class LocalAssetRanker:
    def __init__(self) -> None:
        self._model = None
        self._fitted = False

    def is_available(self) -> bool:
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    def fit(self, bars_dict: Dict[str, pd.DataFrame]) -> None:
        if not self.is_available():
            log.warning("scikit-learn not installed — AI asset ranker disabled")
            return
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            X, y = [], []
            for sym, df in bars_dict.items():
                if len(df) < _MIN_BARS:
                    continue
                feats = self._features(df)
                if feats is None:
                    continue
                # Label: next-bar return (forward 1-bar)
                future_ret = df["close"].pct_change().shift(-1).iloc[-_MIN_BARS:-1]
                hist_feats = [self._features(df.iloc[:i]) for i in range(len(df) - _MIN_BARS, len(df) - 1)]
                for f, r in zip(hist_feats, future_ret):
                    if f is not None and not np.isnan(r):
                        X.append(f)
                        y.append(r)

            if len(X) < 100:
                log.warning("AssetRanker: too few training samples ({})", len(X))
                return

            model = GradientBoostingRegressor(
                n_estimators=200,     # was 100 — more trees, better generalization
                max_depth=4,          # was 3 — capture slightly more interaction
                learning_rate=0.03,   # was 0.05 — slower, more robust
                subsample=0.8,        # stochastic boosting for better generalization
                random_state=42,
            )
            model.fit(np.array(X), np.array(y))
            self._model = model
            self._fitted = True

            with open(_MODEL_CACHE, "wb") as f:
                pickle.dump(model, f)
            log.info("AssetRanker: GBM fitted on {} samples", len(X))
        except Exception as e:
            log.warning("AssetRanker fit failed: {}", e)

    def load(self) -> bool:
        if os.path.exists(_MODEL_CACHE):
            try:
                with open(_MODEL_CACHE, "rb") as f:
                    self._model = pickle.load(f)
                self._fitted = True
                log.info("AssetRanker: model loaded from cache")
                return True
            except Exception as e:
                log.warning("AssetRanker cache load failed: {}", e)
        return False

    def rank_multiplier(self, symbol: str, bars: pd.DataFrame) -> float:
        """
        Return a multiplier in [0.75, 1.4] to scale signal strength.
        1.0 = neutral; >1.0 = AI thinks this asset will outperform.
        Wider range than before (was [0.9, 1.5]) — let the AI have more impact.
        """
        if not self._fitted or self._model is None:
            return 1.0
        try:
            feats = self._features(bars)
            if feats is None:
                return 1.0
            pred = float(self._model.predict([feats])[0])
            # Map predicted return to [0.75, 1.4] — wider range for more impact
            # Scale: pred_ret of +1% → ~1.2x, -1% → ~0.8x
            multiplier = 1.0 + pred * 20  # was 25 — slightly less aggressive scaling
            return float(np.clip(multiplier, 0.75, 1.4))
        except Exception as e:
            log.warning("AssetRanker.rank_multiplier({}) error: {}", symbol, e)
            return 1.0

    @staticmethod
    def _features(df: pd.DataFrame) -> Optional[List[float]]:
        try:
            close  = df["close"]
            high   = df["high"]
            low    = df["low"]
            volume = df.get("volume", pd.Series(dtype=float))
            if len(close) < 25:
                return None

            # Momentum features
            mom5  = close.iloc[-1] / close.iloc[-5]  - 1
            mom20 = close.iloc[-1] / close.iloc[-20] - 1

            # RSI(14) normalized to 0-1
            delta = close.diff()
            gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
            rsi   = float(100 - 100 / (1 + gain.iloc[-1] / max(loss.iloc[-1], 1e-9))) / 100

            # ATR% (normalized volatility)
            h, l, c = high, low, close.shift(1)
            tr  = pd.concat([(h-l), (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
            atr_pct = float(tr.ewm(span=14, adjust=False).mean().iloc[-1]) / float(close.iloc[-1])

            # BB% (position in Bollinger Bands)
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            bb_pct = float((close.iloc[-1] - (sma - 2*std).iloc[-1]) /
                           (4 * std.iloc[-1] + 1e-9))
            bb_pct = np.clip(bb_pct, 0, 1)

            # Volume ratio
            if not volume.empty and len(volume) >= 20:
                vol_ratio = float(volume.iloc[-1] / (volume.iloc[-20:].mean() + 1e-9))
                vol_ratio = np.clip(vol_ratio, 0, 5)
            else:
                vol_ratio = 1.0

            # Trend strength: slope of 20-day linear regression on price (normalized)
            recent = close.iloc[-20:]
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent.values, 1)[0] if len(recent) >= 10 else 0
            trend_strength = float(np.clip(slope / (close.iloc[-1] + 1e-9) * 100, -3, 3))

            # Short-term volatility: 5-day realized vol vs 20-day
            ret_5  = close.pct_change().iloc[-5:].std()
            ret_20 = close.pct_change().iloc[-20:].std()
            vol_expansion = float(ret_5 / (ret_20 + 1e-9) - 1) if ret_20 > 0 else 0.0

            # Day of week (cyclical)
            dow = df.index[-1].weekday() if hasattr(df.index[-1], "weekday") else 0
            dow_sin = np.sin(2 * np.pi * dow / 5)
            dow_cos = np.cos(2 * np.pi * dow / 5)

            return [
                mom5, mom20, rsi, atr_pct, bb_pct, vol_ratio,
                trend_strength, vol_expansion, dow_sin, dow_cos,
            ]
        except Exception:
            return None
