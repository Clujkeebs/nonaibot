# ── Free AI Sentiment Analyzer ───────────────────────────────────────────────
# Uses a lightweight sklearn model to classify market sentiment from news headlines.
# No paid APIs required — all local inference.
#
# Usage: set ENABLE_AI_LAYER=true and this module auto-trains from news data.

from __future__ import annotations

import os
import pickle
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz

import config

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

ET = pytz.timezone(config.TIMEZONE)
_MODEL_CACHE = 'ai/sentiment_model.pkl'
_NEWS_DB = 'ai/news_cache.db'
_MIN_NEWS = 20  # minimum articles needed to train

# Simple keyword-based sentiment labels for training data generation
_BULLISH_KEYWORDS = [
    'beat', 'blow', 'surge', 'rally', 'soar', 'jump', 'gain', 'rise', 'high',
    'upgrade', 'outperform', 'buy', 'strong', 'growth', 'profit', 'record',
    'breakout', 'momentum', 'bullish', 'upgraded', 'exceed',
]
_BEARISH_KEYWORDS = [
    'miss', 'fall', 'drop', 'plunge', 'tumble', 'cut', 'reduce', 'sell',
    'downgrade', 'weak', 'loss', 'warn', 'fear', 'risk', 'bearish',
    'downgraded', 'below', 'concern', 'recession', 'layoff',
]


def _text_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\\s]', ' ', text)  # keep only letters
    text = re.sub(r'\\s+', ' ', text).strip()
    return text


class SentimentAnalyzer:
    def __init__(self) -> None:
        self._model = None
        self._fitted = False
        self._symbol_sentiment: Dict[str, float] = {}
        self._last_update: Optional[datetime] = None

    def is_available(self) -> bool:
        return _HAS_SKLEARN

    def fit(self, news_df: Optional[pd.DataFrame] = None) -> None:
        if not self.is_available():
            from utils.logger import log
            log.warning('SentimentAnalyzer: scikit-learn not installed')
            return
        if news_df is None or len(news_df) < _MIN_NEWS:
            news_df = self._load_news_from_db()
        if news_df is None or len(news_df) < _MIN_NEWS:
            from utils.logger import log
            log.info('SentimentAnalyzer: insufficient news data (%d articles) - using keyword fallback',
                     len(news_df) if news_df is not None else 0)
            self._fitted = True  # Mark as fitted so we use keyword fallback
            return
        try:
            from utils.logger import log
            df = news_df.copy()
            df['text_clean'] = df['headline'].apply(_text_clean)
            df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'bullish' else 0)

            X = df['text_clean'].values
            y = df['label'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)),
                ('clf', SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)),
            ])
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            self._model = pipeline
            self._fitted = True

            with open(_MODEL_CACHE, 'wb') as f:
                pickle.dump(pipeline, f)
            log.info('SentimentAnalyzer: trained on %d articles, test accuracy=%.1f%%', len(X_train), score)
        except Exception as e:
            from utils.logger import log
            log.warning('SentimentAnalyzer fit failed: %s', e)
            self._fitted = True  # Use keyword fallback

    def load(self) -> bool:
        if os.path.exists(_MODEL_CACHE):
            try:
                with open(_MODEL_CACHE, 'rb') as f:
                    self._model = pickle.load(f)
                self._fitted = True
                from utils.logger import log
                log.info('SentimentAnalyzer: model loaded from cache')
                return True
            except Exception as e:
                from utils.logger import log
                log.warning('SentimentAnalyzer cache load failed: %s', e)
        return False

    def score_sentiment(self, text: str) -> float:
        if not self._fitted:
            return 0.0
        try:
            if self._model is not None:
                clean = _text_clean(text)
                prob = self._model.predict_proba([clean])[0][1]  # prob of bullish
                return float(prob) * 2 - 1  # [-1, 1] scale
        except Exception:
            pass
        # Keyword-based fallback sentiment scoring
        text_lower = text.lower()
        bullish_count = sum(1 for kw in _BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in _BEARISH_KEYWORDS if kw in text_lower)
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        return float(bullish_count - bearish_count) / total  # [-1, 1]

    def score_symbol(self, symbol: str, headlines: List[str]) -> float:
        if not headlines:
            return 0.0
        scores = [self.score_sentiment(h) for h in headlines]
        # Weighted average: recent headlines count more
        weights = np.linspace(0.5, 1.0, len(scores))
        weighted = np.average(scores, weights=weights)
        return float(np.clip(weighted, -1.0, 1.0))

    def sentiment_multiplier(self, symbol: str, headlines: List[str]) -> float:
        score = self.score_symbol(symbol, headlines)
        # Map sentiment score [-1, 1] to [0.5, 1.5] multiplier
        # Bearish signal is stronger signal: clip harder on downside
        # Score -1 (bearish) → 0.5x; Score +1 (bullish) → 1.5x
        if score < 0:
            # Bearish: map [-1, 0] → [0.5, 1.0] — asymmetric, penalize more
            mult = 1.0 + score * 0.5   # -1 → 0.5, 0 → 1.0
        else:
            # Bullish: map [0, 1] → [1.0, 1.5]
            mult = 1.0 + score * 0.5   # 0 → 1.0, +1 → 1.5
        return float(np.clip(mult, 0.5, 1.5))

    def update_symbol_sentiment(self, symbol: str, headlines: List[str]) -> None:
        self._symbol_sentiment[symbol] = self.score_symbol(symbol, headlines)
        self._last_update = datetime.now(ET)

    def get_sentiment(self, symbol: str) -> float:
        return self._symbol_sentiment.get(symbol, 0.0)

    def refresh(self, news_by_symbol: Dict[str, List[str]]) -> None:
        for sym, headlines in news_by_symbol.items():
            if headlines:
                self.update_symbol_sentiment(sym, headlines)
        from utils.logger import log
        log.debug('SentimentAnalyzer: updated %d symbols', len(news_by_symbol))

    def _init_db(self) -> None:
        try:
            from utils.logger import log
            os.makedirs(os.path.dirname(_NEWS_DB) or '.', exist_ok=True)
            with sqlite3.connect(_NEWS_DB) as c:
                c.execute('''
                    CREATE TABLE IF NOT EXISTS news (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        headline TEXT,
                        source TEXT,
                        url TEXT,
                        published_at TEXT,
                        fetched_at TEXT
                    )
                ''')
                c.execute('CREATE INDEX IF NOT EXISTS idx_news_symbol ON news(symbol)')
                c.execute('CREATE INDEX IF NOT EXISTS idx_news_date ON news(published_at)')
        except Exception as e:
            from utils.logger import log
            log.warning('SentimentAnalyzer._init_db error: %s', e)

    def cache_news(self, symbol: str, headlines: List[Dict]) -> None:
        self._init_db()
        try:
            from utils.logger import log
            now = datetime.now(ET).isoformat()
            with sqlite3.connect(_NEWS_DB) as c:
                for item in headlines:
                    c.execute('''
                        INSERT INTO news (symbol, headline, source, url, published_at, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        item.get('headline', ''),
                        item.get('source', ''),
                        item.get('url', ''),
                        item.get('published_at', now),
                        now,
                    ))
            log.debug('Cached %d news articles for %s', len(headlines), symbol)
        except Exception as e:
            from utils.logger import log
            log.warning('cache_news error: %s', e)

    def _load_news_from_db(self, days: int = 7, limit: int = 500) -> Optional[pd.DataFrame]:
        self._init_db()
        try:
            from utils.logger import log
            since = (datetime.now(ET) - timedelta(days=days)).isoformat()
            with sqlite3.connect(_NEWS_DB) as c:
                rows = c.execute('''
                    SELECT symbol, headline, published_at FROM news
                    WHERE published_at >= ?
                    ORDER BY published_at DESC LIMIT ?
                ''', (since, limit)).fetchall()
            if not rows:
                return None
            data = [{'symbol': r[0], 'headline': r[1], 'published_at': r[2]} for r in rows]
            df = pd.DataFrame(data)
            def label_headline(text):
                score = self.score_sentiment(text)
                return 'bullish' if score > 0 else 'bearish'
            df['sentiment'] = df['headline'].apply(label_headline)
            return df
        except Exception as e:
            from utils.logger import log
            log.warning('_load_news_from_db error: %s', e)
            return None

    def get_recent_headlines(self, symbol: str, hours: int = 24) -> List[str]:
        self._init_db()
        try:
            since = (datetime.now(ET) - timedelta(hours=hours)).isoformat()
            with sqlite3.connect(_NEWS_DB) as c:
                rows = c.execute('''
                    SELECT headline FROM news
                    WHERE symbol=? AND published_at >= ?
                    ORDER BY published_at DESC LIMIT 20
                ''', (symbol, since)).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    @property
    def last_update(self) -> Optional[datetime]:
        return self._last_update

    @property
    def all_sentiments(self) -> Dict[str, float]:
        return dict(self._symbol_sentiment)