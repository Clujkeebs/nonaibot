"""
DataFetcher — fetches OHLCV bars from Alpaca and caches them.

Cache design:
  - Key = (frozenset(symbols), timeframe, lookback_days)
  - TTL = config.cache_ttl_seconds (default 60s)
  - In-memory only (no disk cache — a restart gets fresh data, which is fine)

Returned DataFrames have columns: open, high, low, close, volume
Index: DatetimeIndex (UTC timestamps)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, FrozenSet, List, Optional, Tuple

import pandas as pd
import pytz

from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from core.broker import BrokerClient
from core.config import BotConfig

logger = logging.getLogger(__name__)

# Cache entry: (result_dict, timestamp)
_CacheKey = Tuple[FrozenSet[str], str, int]
_cache: Dict[_CacheKey, Tuple[Dict[str, pd.DataFrame], float]] = {}


class DataFetcher:
    def __init__(self, broker: BrokerClient, config: BotConfig) -> None:
        self._broker = broker
        self._cfg = config
        self._et = pytz.timezone(config.timezone)

    def get_stock_bars(
        self,
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.Day,
        lookback_days: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch daily (or intraday) bars for equity symbols. Returns {symbol: DataFrame}."""
        if not symbols:
            return {}
        if lookback_days is None:
            lookback_days = self._cfg.equity_lookback_days

        cache_key: _CacheKey = (frozenset(symbols), str(timeframe), lookback_days)
        cached = _cache.get(cache_key)
        if cached and (time.time() - cached[1]) < self._cfg.cache_ttl_seconds:
            return cached[0]

        start = datetime.now(pytz.UTC) - timedelta(days=lookback_days + 5)
        feed = "iex" if self._cfg.iex_feed else "sip"
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start,
                feed=feed,
            )
            bars = self._broker.get_stock_bars(req)
            result = self._bars_to_df(bars, symbols)
            _cache[cache_key] = (result, time.time())
            return result
        except Exception as e:
            logger.warning("get_stock_bars failed for %s: %s", symbols[:5], e)
            return cached[0] if cached else {}

    def get_crypto_bars(
        self,
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.Hour,
        lookback_days: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch hourly (or other) bars for crypto pairs. Returns {symbol: DataFrame}."""
        if not symbols:
            return {}
        if lookback_days is None:
            lookback_days = self._cfg.crypto_lookback_days

        cache_key: _CacheKey = (frozenset(symbols), str(timeframe), lookback_days)
        cached = _cache.get(cache_key)
        if cached and (time.time() - cached[1]) < self._cfg.cache_ttl_seconds:
            return cached[0]

        start = datetime.now(pytz.UTC) - timedelta(days=lookback_days + 1)
        try:
            req = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start,
            )
            bars = self._broker.get_crypto_bars(req)
            result = self._bars_to_df(bars, symbols)
            _cache[cache_key] = (result, time.time())
            return result
        except Exception as e:
            logger.warning("get_crypto_bars failed for %s: %s", symbols[:5], e)
            return cached[0] if cached else {}

    def invalidate(self) -> None:
        """Clear entire cache (call after major regime change or on restart)."""
        _cache.clear()

    @staticmethod
    def _bars_to_df(bars, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Convert alpaca-py BarSet to {symbol: pd.DataFrame}.
        Handles both single-symbol and multi-symbol responses.
        """
        result: Dict[str, pd.DataFrame] = {}
        if bars is None:
            return result

        for sym in symbols:
            try:
                sym_bars = bars[sym]
                if sym_bars is None:
                    continue
                # alpaca-py returns a list of Bar objects; convert to DataFrame
                if hasattr(sym_bars, "df"):
                    df = sym_bars.df
                elif isinstance(sym_bars, list):
                    if not sym_bars:
                        continue
                    records = [
                        {
                            "open": b.open,
                            "high": b.high,
                            "low": b.low,
                            "close": b.close,
                            "volume": b.volume,
                            "timestamp": b.timestamp,
                        }
                        for b in sym_bars
                    ]
                    df = pd.DataFrame(records).set_index("timestamp")
                else:
                    continue

                if df.empty or len(df) < 5:
                    continue

                # Normalise column names to lowercase
                df.columns = [c.lower() for c in df.columns]
                for col in ("open", "high", "low", "close"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                if "volume" in df.columns:
                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

                df = df.dropna(subset=["close"]).sort_index()
                result[sym] = df
            except (KeyError, TypeError, AttributeError):
                continue

        return result

    def data_age_minutes(self, symbol: str, timeframe: TimeFrame = TimeFrame.Day) -> float:
        """Return minutes since the latest bar for a symbol. Used in stale-data checks."""
        for key, (data, _) in _cache.items():
            if symbol in data and str(timeframe) in key[1]:
                df = data[symbol]
                if len(df) > 0:
                    try:
                        last_ts = df.index[-1]
                        if hasattr(last_ts, "tz_convert"):
                            last_ts = last_ts.tz_convert("UTC")
                        now = datetime.now(pytz.UTC)
                        # Ensure both are tz-aware
                        if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is not None:
                            age = (now - last_ts).total_seconds() / 60
                        else:
                            # Naive timestamp — assume it's UTC
                            last_ts_aware = pytz.UTC.localize(last_ts.to_pydatetime()) if hasattr(last_ts, 'to_pydatetime') else pytz.UTC.localize(last_ts)
                            age = (now - last_ts_aware).total_seconds() / 60
                        return age
                    except Exception:
                        pass
        return float("inf")
