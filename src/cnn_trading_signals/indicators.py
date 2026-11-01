"""Causal, vectorised technical indicators.

Every function takes OHLC price series and a look-back ``period`` and returns a
series aligned with the input whose value at bar ``t`` depends only on bars
``<= t``. Values are normalised to a roughly ``[-1, 1]`` range so that the
indicator image is on a comparable scale across indicators and instruments.

The indicator set follows the idea in Sezer & Ozbayoglu (2018), "Algorithmic
financial trading with deep convolutional neural networks: Time series to image
conversion approach", with volume-free substitutes so it also works for FX,
where exchanges do not publish volume.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Series = pd.Series


def _wilder(x: Series, period: int) -> Series:
    """Wilder's smoothing (an EMA with alpha = 1 / period)."""
    return x.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _ema(x: Series, period: int) -> Series:
    return x.ewm(span=period, adjust=False, min_periods=period).mean()


def _sma(x: Series, period: int) -> Series:
    return x.rolling(period, min_periods=period).mean()


def _wma(x: Series, period: int) -> Series:
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    return x.rolling(period, min_periods=period).apply(lambda w: float(np.dot(w, weights)), raw=True)


def _distance(close: Series, average: Series) -> Series:
    """Signed distance of price from an average, in tenths of a percent."""
    return (close - average) / close * 10.0


def _true_range(high: Series, low: Series, close: Series) -> Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def rsi(high: Series, low: Series, close: Series, period: int) -> Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), period)
    loss = _wilder(-delta.clip(upper=0.0), period)
    rs = gain / loss
    value = 100.0 - 100.0 / (1.0 + rs)
    value = value.where(loss != 0, 100.0).where(~(gain.eq(0) & loss.eq(0)), 50.0)
    return value.where(gain.notna()) / 50.0 - 1.0


def williams_r(high: Series, low: Series, close: Series, period: int) -> Series:
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    rng = (hh - ll).replace(0.0, np.nan)
    value = -100.0 * (hh - close) / rng  # in [-100, 0]
    return value / 50.0 + 1.0


def wma(high: Series, low: Series, close: Series, period: int) -> Series:
    return _distance(close, _wma(close, period))


def ema(high: Series, low: Series, close: Series, period: int) -> Series:
    return _distance(close, _ema(close, period))


def sma(high: Series, low: Series, close: Series, period: int) -> Series:
    return _distance(close, _sma(close, period))


def hma(high: Series, low: Series, close: Series, period: int) -> Series:
    half = max(period // 2, 1)
    root = max(int(np.sqrt(period)), 1)
    raw = 2.0 * _wma(close, half) - _wma(close, period)
    return _distance(close, _wma(raw, root))


def tema(high: Series, low: Series, close: Series, period: int) -> Series:
    e1 = _ema(close, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    return _distance(close, 3.0 * e1 - 3.0 * e2 + e3)


def cci(high: Series, low: Series, close: Series, period: int) -> Series:
    typical = (high + low + close) / 3.0
    mean = typical.rolling(period, min_periods=period).mean()
    mad = typical.rolling(period, min_periods=period).apply(lambda w: float(np.mean(np.abs(w - w.mean()))), raw=True)
    value = (typical - mean) / (0.015 * mad.replace(0.0, np.nan))
    return value / 300.0


def cmo(high: Series, low: Series, close: Series, period: int) -> Series:
    delta = close.diff()
    up = delta.clip(lower=0.0).rolling(period, min_periods=period).sum()
    down = (-delta.clip(upper=0.0)).rolling(period, min_periods=period).sum()
    total = (up + down).replace(0.0, np.nan)
    return (up - down) / total  # CMO / 100


def macd(high: Series, low: Series, close: Series, period: int) -> Series:
    """MACD with fast = period and slow = 2 * period, as a percent of price."""
    line = _ema(close, period) - _ema(close, 2 * period)
    return np.tanh(line / close * 100.0)


def ppo(high: Series, low: Series, close: Series, period: int) -> Series:
    slow = _ema(close, 2 * period)
    return 100.0 * (_ema(close, period) - slow) / slow / 10.0


def roc(high: Series, low: Series, close: Series, period: int) -> Series:
    return (close / close.shift(period) - 1.0) * 100.0 / 20.0


def dx(high: Series, low: Series, close: Series, period: int) -> Series:
    """Directional movement index (DX), the un-smoothed core of ADX."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _wilder(_true_range(high, low, close), period)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr
    minus_di = 100.0 * _wilder(minus_dm, period) / atr
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    value = 100.0 * (plus_di - minus_di).abs() / denom
    return value / 50.0 - 1.0


def bollinger_b(high: Series, low: Series, close: Series, period: int) -> Series:
    """Bollinger %B (2 standard deviations) mapped so the bands sit at -1 and +1."""
    mid = _sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return (close - mid) / (2.0 * std.replace(0.0, np.nan))


def atr_percent(high: Series, low: Series, close: Series, period: int) -> Series:
    """Average true range as a percent of price (a volatility channel)."""
    return _wilder(_true_range(high, low, close), period) / close * 100.0


Indicator = Callable[[Series, Series, Series, int], Series]

#: Image rows, in order. Each indicator is evaluated for every period column.
INDICATORS: dict[str, Indicator] = {
    "RSI": rsi,
    "WILLR": williams_r,
    "WMA": wma,
    "EMA": ema,
    "SMA": sma,
    "HMA": hma,
    "TEMA": tema,
    "CCI": cci,
    "CMO": cmo,
    "MACD": macd,
    "PPO": ppo,
    "ROC": roc,
    "DX": dx,
    "BBP": bollinger_b,
    "ATRP": atr_percent,
}
