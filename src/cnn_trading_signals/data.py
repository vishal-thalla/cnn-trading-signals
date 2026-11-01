"""Market data loading.

Three sources are supported, all returning a DataFrame indexed by a UTC
``DatetimeIndex`` with ``open, high, low, close`` columns (``volume`` optional):

* ``yfinance`` - free historical bars from Yahoo Finance (cached as CSV).
* ``csv``      - any local file with a datetime column and OHLC columns.
* ``synthetic`` - a seeded regime-switching random walk for tests and demos.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"price data is missing columns: {missing}")
    index = pd.DatetimeIndex(df.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    df.index = index.rename("datetime")
    cols = OHLC + (["volume"] if "volume" in df.columns else [])
    df = df[cols].astype(float).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=OHLC)
    # Yahoo occasionally ships bars where high/low do not bracket open/close.
    df["high"] = df[OHLC].max(axis=1)
    df["low"] = df[OHLC].min(axis=1)
    return df[(df[OHLC] > 0).all(axis=1)]


def load_csv(path: str | Path, datetime_column: str = "datetime") -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.lower() for c in raw.columns]
    column = datetime_column.lower()
    if column not in raw.columns:
        raise ValueError(f"{path}: no '{datetime_column}' column")
    raw.index = pd.to_datetime(raw.pop(column), utc=True)
    return _normalise(raw)


def load_yfinance(
    symbol: str,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | Path | None = "data/cache",
) -> pd.DataFrame:
    """Download bars from Yahoo Finance, reusing a CSV cache when present."""
    cache: Path | None = None
    if cache_dir is not None:
        key = hashlib.sha1(f"{symbol}|{interval}|{start}|{end}".encode()).hexdigest()[:10]
        safe = "".join(ch if ch.isalnum() else "_" for ch in symbol)
        cache = Path(cache_dir) / f"{safe}_{interval}_{key}.csv"
        if cache.exists():
            return load_csv(cache)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("yfinance is not installed: pip install 'cnn-trading-signals[data]'") from exc

    raw = yf.download(symbol, interval=interval, start=start, end=end, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {symbol} ({interval})")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = _normalise(raw)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index_label="datetime")
    return df


def synthetic_bars(
    n: int = 3000,
    seed: int = 7,
    start: str = "2010-01-01",
    freq: str = "B",
    start_price: float = 100.0,
) -> pd.DataFrame:
    """A seeded random walk that alternates between trending and mean-reverting regimes.

    It is meant for tests and a credentials-free demo, not as evidence that a
    strategy works: any edge found here says nothing about real markets.
    """
    rng = np.random.default_rng(seed)
    regime_len = rng.integers(40, 160, size=n // 40 + 2)
    regimes = np.repeat(rng.choice([-1, 0, 1], size=regime_len.size), regime_len)[:n]
    drift = regimes * 0.0006
    vol = 0.006 + 0.004 * (regimes == 0)
    log_ret = drift + vol * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(log_ret))
    open_ = np.r_[start_price, close[:-1]] * np.exp(0.001 * rng.standard_normal(n))
    spread = np.abs(rng.standard_normal(n)) * vol * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="datetime")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


def load_bars(source: dict) -> pd.DataFrame:
    """Dispatch on ``source['type']`` (as written in an experiment config)."""
    params = dict(source)
    kind = params.pop("type")
    if kind == "yfinance":
        return load_yfinance(**params)
    if kind == "csv":
        return load_csv(**params)
    if kind == "synthetic":
        return synthetic_bars(**params)
    raise ValueError(f"unknown data source type: {kind!r}")
