"""
Technical indicators — pure functions, no side effects.

All functions accept pandas Series/DataFrames and return pandas Series.
NaN is propagated: callers should check for NaN before using results.

WHY these indicators:
  EMA: smoother than SMA, reacts faster to recent price changes
  RSI: momentum oscillator, identifies overbought/oversold conditions
  ATR: measures volatility — used for position sizing and stop distances
  ADX: trend strength (not direction) — tells us if trend is real or chop
  Bollinger Bands: volatility envelope — lower band = potential oversold
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder smoothing).
    Returns 0-100. Values <30 = oversold, >70 = overbought (classic).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder smoothing).
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index.
    Returns 0-100. >20 = trending, >40 = strong trend.
    WHY: tells us if the market is trending — we only use trend strategy when ADX is high.
    """
    tr = _true_range(high, low, close)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, float("nan")))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, float("nan")))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan")))
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(
    series: pd.Series, period: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns (upper, middle, lower).
    WHY: price touching lower band in a range market = potential mean-reversion entry.
    """
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    return upper, middle, lower


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Current volume / N-period average volume.
    Ratio > 1.5 = above-average volume (confirms moves).
    """
    avg = volume.rolling(period).mean()
    return volume / avg.replace(0, float("nan"))


def ema_slope(ema_series: pd.Series, lookback: int = 3) -> float:
    """
    Slope of the EMA over the last `lookback` bars, expressed as % per bar.
    Positive = rising trend, negative = falling.
    """
    if len(ema_series) < lookback + 1:
        return 0.0
    vals = ema_series.iloc[-lookback - 1:]
    first = vals.iloc[0]
    if first == 0 or pd.isna(first):
        return 0.0
    slope = (vals.iloc[-1] - first) / (first * lookback)
    return float(slope)


def percent_b(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    """
    %B — position of price within Bollinger Bands.
    0 = at lower band, 1 = at upper band, <0 = below lower band.
    """
    band_width = upper - lower
    return (close - lower) / band_width.replace(0, float("nan"))
