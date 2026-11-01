"""Convert OHLC bars into indicator images.

Row ``i`` of an image is indicator ``i`` from :data:`~cnn_trading_signals.indicators.INDICATORS`;
column ``j`` is look-back period ``periods[j]``. With the defaults (15 indicators,
periods 6..20) each bar becomes a 15 x 15 single-channel image.

Because every indicator is causal, the image at bar ``t`` uses only bars ``<= t``.
``tests/test_imaging.py`` checks this by perturbing future bars.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .indicators import INDICATORS

DEFAULT_PERIODS: tuple[int, ...] = tuple(range(6, 21))

#: Bars dropped at the start so slow indicators (TEMA, MACD at 2x period) have settled.
DEFAULT_WARMUP = 120

#: Normalised indicator values are clipped to this range before modelling.
CLIP = 3.0


def min_warmup(periods: Sequence[int]) -> int:
    """Bars needed before every indicator is defined (TEMA chains three EMAs)."""
    return 3 * max(periods)


def indicator_images(
    bars: pd.DataFrame,
    periods: Sequence[int] = DEFAULT_PERIODS,
    indicators: Sequence[str] | None = None,
) -> np.ndarray:
    """Return a float32 array of shape ``(len(bars), n_indicators, n_periods)``.

    ``bars`` needs ``high``, ``low`` and ``close`` columns. Rows whose look-back
    is not yet full contain NaN; pass the result through :func:`finalize`.
    """
    names = list(indicators) if indicators is not None else list(INDICATORS)
    high, low, close = (bars[c].astype(float) for c in ("high", "low", "close"))
    images = np.empty((len(bars), len(names), len(periods)), dtype=np.float32)
    for i, name in enumerate(names):
        fn = INDICATORS[name]
        for j, period in enumerate(periods):
            images[:, i, j] = fn(high, low, close, int(period)).to_numpy(dtype=float)
    images[np.isinf(images)] = np.nan
    np.clip(images, -CLIP, CLIP, out=images)
    return images


def finalize(images: np.ndarray, warmup: int = DEFAULT_WARMUP) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, usable)`` ready for modelling.

    Rows before ``warmup`` are marked unusable. After the warm-up an undefined
    cell can only come from a degenerate window (for example zero range over the
    look-back), so it is set to 0, the neutral value of every indicator. Rows are
    never dropped mid-series, which keeps the backtest's bar sequence contiguous.
    """
    usable = np.arange(images.shape[0]) >= warmup
    return np.nan_to_num(images, nan=0.0), usable
