"""
Alpaca data client — historical bars + latest quotes.

One MarketDataClient is instantiated at startup and shared across all modules.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestBarRequest,
    CryptoLatestBarRequest,
)
from alpaca.data.timeframe import TimeFrame

import config
from utils.logger import log

ET = pytz.timezone(config.TIMEZONE)


class MarketDataClient:
    """Thin wrapper around alpaca-py historical data clients."""

    def __init__(self) -> None:
        self._stocks = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )
        # Crypto client requires no credentials for historical data
        self._crypto = CryptoHistoricalDataClient()
        log.info("MarketDataClient initialised (paper={})", config.ALPACA_PAPER)

    # ── Bars ──────────────────────────────────────────────────────────────────

    def get_stock_bars(
        self,
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.Day,
        lookback_days: int = config.BARS_LOOKBACK_DAYS,
    ) -> Dict[str, pd.DataFrame]:
        """Return {symbol: OHLCV DataFrame} for each symbol."""
        start = datetime.now(ET) - timedelta(days=lookback_days)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start,
                feed="iex",          # free IEX feed; switch to "sip" on paid plan
            )
            bars = self._stocks.get_stock_bars(req)
            return self._unpack_bars(bars, symbols)
        except Exception as e:
            log.error("get_stock_bars failed for {}: {}", symbols, e)
            return {}

    def get_crypto_bars(
        self,
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.Hour,
        lookback_days: int = config.CRYPTO_BARS_LOOKBACK,
    ) -> Dict[str, pd.DataFrame]:
        start = datetime.now(ET) - timedelta(days=lookback_days)
        try:
            req = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start,
            )
            bars = self._crypto.get_crypto_bars(req)
            return self._unpack_bars(bars, symbols)
        except Exception as e:
            log.error("get_crypto_bars failed for {}: {}", symbols, e)
            return {}

    # ── Latest quotes / prices ────────────────────────────────────────────────

    def get_latest_stock_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Return {symbol: mid_price}."""
        if not symbols:
            return {}
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self._stocks.get_stock_latest_quote(req)
            result = {}
            for sym, q in quotes.items():
                bid = float(q.bid_price or 0)
                ask = float(q.ask_price or 0)
                result[sym] = (bid + ask) / 2 if bid and ask else (ask or bid)
            return result
        except Exception as e:
            log.error("get_latest_stock_prices failed: {}", e)
            return {}

    def get_latest_crypto_prices(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}
        try:
            req = CryptoLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self._crypto.get_crypto_latest_quote(req)
            result = {}
            for sym, q in quotes.items():
                bid = float(q.bid_price or 0)
                ask = float(q.ask_price or 0)
                result[sym] = (bid + ask) / 2 if bid and ask else (ask or bid)
            return result
        except Exception as e:
            log.error("get_latest_crypto_prices failed: {}", e)
            return {}

    def get_latest_bar_price(self, symbol: str, is_crypto: bool = False) -> Optional[float]:
        """Single symbol price via latest bar close (fast fallback)."""
        try:
            if is_crypto:
                req = CryptoLatestBarRequest(symbol_or_symbols=[symbol])
                bars = self._crypto.get_crypto_latest_bar(req)
            else:
                req = StockLatestBarRequest(symbol_or_symbols=[symbol])
                bars = self._stocks.get_stock_latest_bar(req)
            bar = bars.get(symbol)
            return float(bar.close) if bar else None
        except Exception as e:
            log.warning("get_latest_bar_price({}) failed: {}", symbol, e)
            return None

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_bars(bars_response, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Convert alpaca BarSet to {symbol: DataFrame}."""
        result: Dict[str, pd.DataFrame] = {}
        try:
            df_all = bars_response.df
            if df_all.empty:
                return result
            if isinstance(df_all.index, pd.MultiIndex):
                for sym in symbols:
                    try:
                        df = df_all.xs(sym, level="symbol").copy()
                        df.index = pd.to_datetime(df.index, utc=True)
                        df.sort_index(inplace=True)
                        df.columns = [c.lower() for c in df.columns]
                        if len(df) >= 20:   # need enough bars for indicators
                            result[sym] = df
                    except KeyError:
                        pass
            else:
                sym = symbols[0] if len(symbols) == 1 else None
                if sym:
                    df_all.index = pd.to_datetime(df_all.index, utc=True)
                    df_all.columns = [c.lower() for c in df_all.columns]
                    result[sym] = df_all
        except Exception as e:
            log.error("_unpack_bars error: {}", e)
        return result
