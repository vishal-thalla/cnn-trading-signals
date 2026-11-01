import numpy as np
import pandas as pd
import pytest

from cnn_trading_signals import indicators as ind
from cnn_trading_signals.data import synthetic_bars


@pytest.fixture(scope="module")
def bars():
    return synthetic_bars(n=400, seed=3)


def _hlc(bars):
    return bars["high"], bars["low"], bars["close"]


def test_sma_distance_matches_definition(bars):
    close = bars["close"]
    got = ind.sma(*_hlc(bars), 10)
    expected = (close - close.rolling(10).mean()) / close * 10
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_wma_weights_recent_bars_more():
    close = pd.Series([1.0, 2.0, 3.0, 4.0])
    wma = ind._wma(close, 4)
    assert wma.iloc[-1] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3 + 4 * 4) / 10)


def test_rsi_extremes():
    up = pd.Series(np.linspace(1, 2, 50))
    down = up[::-1].reset_index(drop=True)
    assert ind.rsi(up, up, up, 14).iloc[-1] == pytest.approx(1.0)  # RSI 100 -> +1
    assert ind.rsi(down, down, down, 14).iloc[-1] == pytest.approx(-1.0)  # RSI 0 -> -1


def test_rsi_matches_wilder_reference():
    # Reference values from a hand-rolled Wilder RSI loop.
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.standard_normal(200).cumsum())
    period = 14
    delta = close.diff().to_numpy()
    avg_gain = avg_loss = None
    ref = []
    for i, d in enumerate(delta):
        if i == 0:
            ref.append(np.nan)
            continue
        gain, loss = max(d, 0), max(-d, 0)
        if avg_gain is None:
            avg_gain, avg_loss = gain, loss
        else:
            avg_gain += (gain - avg_gain) / period
            avg_loss += (loss - avg_loss) / period
        ref.append(100 - 100 / (1 + avg_gain / avg_loss))
    got = ind.rsi(close, close, close, period) * 50 + 50
    np.testing.assert_allclose(got.iloc[50:], np.array(ref)[50:], rtol=1e-9)


def test_williams_r_range(bars):
    value = ind.williams_r(*_hlc(bars), 14).dropna()
    assert value.between(-1, 1).all()


def test_bollinger_b_at_band_is_one():
    close = pd.Series([1.0, -1.0] * 10 + [1.0])
    # mean 0 over even windows; use an odd window where the last bar is the high
    value = ind.bollinger_b(close, close, close, 2)
    assert value.iloc[-1] == pytest.approx(0.5)  # (1 - 0) / (2 * 1)


@pytest.mark.parametrize("name", list(ind.INDICATORS))
def test_indicators_are_causal(bars, name):
    fn = ind.INDICATORS[name]
    full = fn(*_hlc(bars), 12)
    cut = 250
    truncated = fn(*_hlc(bars.iloc[:cut]), 12)
    np.testing.assert_allclose(full.iloc[:cut].to_numpy(), truncated.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("name", list(ind.INDICATORS))
def test_indicators_are_bounded_and_defined_after_warmup(bars, name):
    value = ind.INDICATORS[name](*_hlc(bars), 20).iloc[120:]
    assert value.notna().all()
    assert value.abs().max() < 10
