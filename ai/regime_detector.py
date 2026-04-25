"""
Optional AI Layer — Local Regime Detector

Uses scikit-learn's GaussianHMM (Hidden Markov Model) to detect market regimes
from SPY return + volatility features.

Requirements (only if ENABLE_AI_LAYER=true):
  pip install hmmlearn scikit-learn

The model trains on the most recent 2 years of SPY data at startup,
then predicts the current regime every hour.

Regime states:
  0 → Low vol / trending up  (BULL)
  1 → High vol / directionless (TRANSITION)
  2 → High vol / trending down  (BEAR)

The predicted state overrides the rule-based RegimeFilter when available.
The bot functions 100% normally if this module is not imported or fails.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import log

_MODEL_CACHE = "ai/hmm_model.pkl"
_N_COMPONENTS = 3   # number of hidden states
_TRAIN_DAYS   = 504  # ~2 years of trading days


class LocalRegimeDetector:
    """
    HMM-based regime detector. Completely optional — if hmmlearn is not
    installed the class is imported but never instantiated.
    """

    def __init__(self) -> None:
        self._model = None
        self._fitted = False
        self._last_state: Optional[int] = None

    def is_available(self) -> bool:
        try:
            import hmmlearn  # noqa: F401
            return True
        except ImportError:
            return False

    def fit(self, spy_bars: pd.DataFrame) -> None:
        if not self.is_available():
            log.warning("hmmlearn not installed — AI regime detector disabled")
            return
        try:
            from hmmlearn.hmm import GaussianHMM

            features = self._extract_features(spy_bars)
            if features is None or len(features) < 60:
                return

            model = GaussianHMM(
                n_components=_N_COMPONENTS,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model.fit(features)
            self._model = model
            self._fitted = True

            with open(_MODEL_CACHE, "wb") as f:
                pickle.dump(model, f)
            log.info("AI RegimeDetector: HMM fitted on {} bars", len(features))
        except Exception as e:
            log.warning("AI RegimeDetector fit failed: {}", e)

    def load(self) -> bool:
        if os.path.exists(_MODEL_CACHE):
            try:
                with open(_MODEL_CACHE, "rb") as f:
                    self._model = pickle.load(f)
                self._fitted = True
                log.info("AI RegimeDetector: model loaded from cache")
                return True
            except Exception as e:
                log.warning("AI RegimeDetector: cache load failed: {}", e)
        return False

    def predict(self, spy_bars: pd.DataFrame) -> Optional[int]:
        if not self._fitted or self._model is None:
            return None
        try:
            features = self._extract_features(spy_bars)
            if features is None or len(features) < 10:
                return None
            states = self._model.predict(features)
            self._last_state = int(states[-1])

            # Re-order states by mean return so state 0 = bear, 2 = bull
            means = self._model.means_[:, 0]  # first feature = return
            order = np.argsort(means)
            state_map = {old: new for new, old in enumerate(order)}
            return state_map[self._last_state]
        except Exception as e:
            log.warning("AI RegimeDetector predict failed: {}", e)
            return None

    @property
    def last_state(self) -> Optional[int]:
        return self._last_state

    @staticmethod
    def _extract_features(spy_bars: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            close  = spy_bars["close"].iloc[-_TRAIN_DAYS:]
            ret    = close.pct_change().dropna()
            vol    = ret.rolling(10).std().dropna()
            common = ret.index.intersection(vol.index)
            if len(common) < 30:
                return None
            ret = ret.loc[common].values.reshape(-1, 1)
            vol = vol.loc[common].values.reshape(-1, 1)
            return np.hstack([ret, vol])
        except Exception:
            return None
