import numpy as np
import pandas as pd
import pytest

from cnn_trading_signals.backtest import (
    CostModel,
    round_trips,
    signals_from_probabilities,
    strategy_returns,
    target_positions,
)
from cnn_trading_signals.labeling import BUY, HOLD, SELL
from cnn_trading_signals.metrics import performance


def _bars(opens, closes):
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="D", tz="UTC")
    o, c = np.asarray(opens, float), np.asarray(closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c}, index=idx)


def test_target_positions_long_only_and_long_short():
    signals = np.array([HOLD, BUY, HOLD, SELL, HOLD, SELL, BUY])
    np.testing.assert_array_equal(target_positions(signals), [0, 1, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(target_positions(signals, allow_short=True), [0, 1, 1, -1, -1, -1, 1])


def test_confidence_threshold_turns_weak_signals_into_hold():
    proba = np.array([[0.5, 0.3, 0.2], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    np.testing.assert_array_equal(signals_from_probabilities(proba, 0.6), [HOLD, BUY, SELL])


def test_fills_at_next_open_without_costs():
    # Decide to buy at the close of bar 0 (100); fill at bar 1's open (110).
    bars = _bars(opens=[100, 110, 120], closes=[100, 115, 126])
    rets = strategy_returns(bars, np.array([1.0, 1.0, 1.0]), CostModel(0, 0))
    assert rets.iloc[0] == 0
    assert rets.iloc[1] == pytest.approx(115 / 110 - 1)  # no credit for the 100 -> 110 gap
    assert rets.iloc[2] == pytest.approx(126 / 115 - 1)  # gap and intrabar while held
    assert (1 + rets).prod() == pytest.approx(126 / 110)


def test_costs_are_charged_per_unit_turnover():
    bars = _bars(opens=[100] * 4, closes=[100] * 4)
    costs = CostModel(spread_bps=2, commission_bps=1)  # 2 bp per unit
    rets = strategy_returns(bars, np.array([1.0, -1.0, 0.0, 0.0]), costs)
    np.testing.assert_allclose(rets.to_numpy(), [0, -2e-4, -4e-4, -2e-4])


def test_short_profits_when_price_falls():
    bars = _bars(opens=[100, 100, 90], closes=[100, 90, 81])
    rets = strategy_returns(bars, np.array([-1.0, -1.0, -1.0]), CostModel(0, 0))
    assert rets.iloc[1] == pytest.approx(0.1)
    assert rets.iloc[2] == pytest.approx(0.1)


def test_round_trips_record_next_open_prices():
    bars = _bars(opens=[10, 11, 12, 13, 14], closes=[10.5, 11.5, 12.5, 13.5, 14.5])
    trades = round_trips(bars, np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    assert len(trades) == 1
    assert (trades[0].entry_price, trades[0].exit_price) == (11, 13)
    assert trades[0].gross_return == pytest.approx(13 / 11 - 1)


def test_performance_metrics_on_known_series():
    rets = pd.Series([0.1, -0.5, 0.2])
    perf = performance(rets, bars_per_year=3)
    assert perf["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert perf["max_drawdown"] == pytest.approx(-0.5)
    assert perf["cagr"] == pytest.approx(perf["total_return"])  # exactly one year
