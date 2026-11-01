import numpy as np
import pytest

from cnn_trading_signals.data import synthetic_bars
from cnn_trading_signals.imaging import DEFAULT_WARMUP, finalize, indicator_images
from cnn_trading_signals.labeling import BUY, HOLD, SELL, label_horizon, turning_point_labels
from cnn_trading_signals.splits import walk_forward


def test_turning_points_on_a_zigzag():
    close = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
    labels = turning_point_labels(close, 5)
    assert labels[4] == BUY
    assert labels[8] == SELL
    assert labels[0] == labels[1] == labels[-1] == labels[-2] == -1  # undefined at the edges
    assert set(labels[2:-2]) <= {BUY, HOLD, SELL}
    assert (labels[2:-2] != HOLD).sum() == 2


def test_label_depends_only_on_its_window():
    rng = np.random.default_rng(1)
    close = 100 + rng.standard_normal(300).cumsum()
    window = 11
    h = label_horizon(window)
    base = turning_point_labels(close, window)
    changed = close.copy()
    changed[200:] += 50  # move the future
    after = turning_point_labels(changed, window)
    np.testing.assert_array_equal(base[: 200 - h], after[: 200 - h])


@pytest.mark.parametrize("window", [2, 4, 1])
def test_window_must_be_odd(window):
    with pytest.raises(ValueError):
        turning_point_labels(np.arange(20.0), window)


def test_images_shape_and_no_lookahead():
    bars = synthetic_bars(n=500, seed=11)
    images = indicator_images(bars)
    assert images.shape == (500, 15, 15)

    shocked = bars.copy()
    shocked.iloc[400:, :] *= 1.5  # perturb the future
    images_shocked = indicator_images(shocked)
    np.testing.assert_allclose(images[:400], images_shocked[:400], equal_nan=True)


def test_finalize_marks_warmup_and_fills_degenerate_cells():
    bars = synthetic_bars(n=300, seed=2)
    bars.iloc[200:230, :] = 100.0  # a flat stretch: zero ranges, undefined oscillators
    images, usable = finalize(indicator_images(bars))
    assert not usable[:DEFAULT_WARMUP].any() and usable[DEFAULT_WARMUP:].all()
    assert not np.isnan(images).any()


def test_walk_forward_purges_label_horizon():
    folds = list(walk_forward(1000, n_folds=4, min_train=400, horizon=5, embargo=2))
    assert len(folds) == 4
    tests = np.concatenate([f.test for f in folds])
    np.testing.assert_array_equal(tests, np.arange(400, 1000))  # contiguous, no overlap
    for f in folds:
        assert f.train[-1] == f.test[0] - 5 - 2 - 1
        assert f.train[0] == 0


def test_walk_forward_rolling_window():
    folds = list(walk_forward(1000, n_folds=3, min_train=400, horizon=3, max_train=200))
    assert all(f.train.size == 200 for f in folds)
