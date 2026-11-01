"""Turning-point labels.

A bar is labelled ``BUY`` when its close is the lowest close of the centred
window around it, ``SELL`` when it is the highest, and ``HOLD`` otherwise. This
matches the sliding-window labelling from the original trading-bot-v2 research
code (itself from Sezer & Ozbayoglu, 2018).

The label at bar ``t`` looks ``horizon = window // 2`` bars into the future, so
it is only *known* at bar ``t + horizon``. Walk-forward splits must purge that
many bars before each test fold; see :mod:`cnn_trading_signals.splits`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BUY, HOLD, SELL = 0, 1, 2
CLASS_NAMES = ("BUY", "HOLD", "SELL")


def label_horizon(window: int) -> int:
    """Number of future bars a label of this window size depends on."""
    return window // 2


def turning_point_labels(close: pd.Series | np.ndarray, window: int) -> np.ndarray:
    """Return an int array of BUY/HOLD/SELL labels, ``-1`` where undefined.

    ``window`` must be odd so the middle bar is well defined. Ties resolve to the
    earliest bar in the window, as in the original implementation.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError(f"window must be an odd integer >= 3, got {window}")
    values = np.asarray(close, dtype=float)
    labels = np.full(values.shape[0], -1, dtype=np.int64)
    if values.shape[0] < window:
        return labels

    half = label_horizon(window)
    windows = sliding_window_view(values, window)
    middle = np.arange(windows.shape[0]) + half
    is_top = windows.argmax(axis=1) == half
    is_bottom = windows.argmin(axis=1) == half

    labels[middle] = HOLD
    labels[middle[is_bottom]] = BUY
    labels[middle[is_top & ~is_bottom]] = SELL
    return labels
