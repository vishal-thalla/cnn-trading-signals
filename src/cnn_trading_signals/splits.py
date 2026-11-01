"""Purged walk-forward splits.

Labels look ``horizon`` bars ahead, so a training sample at bar ``t`` leaks
information about bars up to ``t + horizon``. Each fold therefore drops the last
``horizon + embargo`` training rows before the test window starts (purging, as
described in Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Fold(train={self.train[0]}..{self.train[-1]} n={self.train.size}, "
            f"test={self.test[0]}..{self.test[-1]} n={self.test.size})"
        )


def walk_forward(
    n_samples: int,
    n_folds: int,
    min_train: int,
    horizon: int,
    embargo: int = 0,
    max_train: int | None = None,
) -> Iterator[Fold]:
    """Yield chronological folds over ``range(n_samples)``.

    The data after the first ``min_train`` rows is cut into ``n_folds`` equal,
    contiguous test windows. Training uses every earlier row (expanding window,
    or the latest ``max_train`` rows when set) except the purged gap.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    gap = horizon + embargo
    test_span = n_samples - min_train
    if test_span < n_folds:
        raise ValueError(f"not enough samples: {n_samples} rows, min_train={min_train}, n_folds={n_folds}")

    edges = np.linspace(min_train, n_samples, n_folds + 1).astype(int)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        train_stop = start - gap
        train_start = 0 if max_train is None else max(0, train_stop - max_train)
        if train_stop - train_start < 1:
            raise ValueError("purge gap leaves no training data; lower horizon/embargo or raise min_train")
        yield Fold(train=np.arange(train_start, train_stop), test=np.arange(start, stop))
