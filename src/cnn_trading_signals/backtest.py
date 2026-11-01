"""A small, explicit backtester for BUY/HOLD/SELL signals.

Execution model
---------------
* A signal is computed from the bar that just **closed** (``t``).
* The resulting position change fills at the **next bar's open** (``t + 1``),
  so a strategy can never trade at a price it used to decide.
* Each unit of turnover (going 0 -> 1 is 1, going -1 -> 1 is 2) pays
  ``spread_bps / 2 + commission_bps`` basis points.

Per-bar strategy return at bar ``t`` (the position is re-set at the open)::

    1 + r_t = (1 + pos_{t-1} * (open_t / close_{t-1} - 1))   # overnight gap, old position
            * (1 + pos_t     * (close_t / open_t - 1))       # intrabar move, new position
            - cost * |pos_t - pos_{t-1}|
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labeling import BUY, HOLD, SELL


@dataclass(frozen=True)
class CostModel:
    spread_bps: float = 1.0
    commission_bps: float = 0.0

    @property
    def per_unit_turnover(self) -> float:
        return (self.spread_bps / 2.0 + self.commission_bps) / 1e4


def signals_from_probabilities(proba: np.ndarray, min_confidence: float = 0.0) -> np.ndarray:
    """Arg-max class per row; BUY/SELL below ``min_confidence`` become HOLD."""
    signals = proba.argmax(axis=1)
    weak = proba.max(axis=1) < min_confidence
    signals[weak & (signals != HOLD)] = HOLD
    return signals


def target_positions(signals: np.ndarray, allow_short: bool = False) -> np.ndarray:
    """Turn a BUY/HOLD/SELL stream into the position held after each bar's close.

    Long-only: BUY opens a long, SELL closes it. Long/short: BUY goes long and
    SELL goes short (reversing an open position). HOLD keeps the current position.
    """
    positions = np.zeros(len(signals), dtype=float)
    current = 0.0
    for i, signal in enumerate(signals):
        if signal == BUY:
            current = 1.0
        elif signal == SELL:
            current = -1.0 if allow_short else 0.0
        positions[i] = current
    return positions


def strategy_returns(bars: pd.DataFrame, target: np.ndarray, costs: CostModel) -> pd.Series:
    """Per-bar returns for ``target`` positions decided at each close (see module docs)."""
    if len(bars) != len(target):
        raise ValueError("bars and target positions must have the same length")
    open_ = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)

    held = np.r_[0.0, target[:-1]]  # position live during bar t (filled at its open)
    prev_held = np.r_[0.0, held[:-1]]
    prev_close = np.r_[open_[0], close[:-1]]

    gap = open_ / prev_close - 1.0
    intrabar = close / open_ - 1.0
    turnover = np.abs(held - prev_held)
    returns = (1.0 + prev_held * gap) * (1.0 + held * intrabar) - 1.0 - costs.per_unit_turnover * turnover
    return pd.Series(returns, index=bars.index, name="strategy")


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_price: float
    exit_price: float

    @property
    def gross_return(self) -> float:
        return self.side * (self.exit_price / self.entry_price - 1.0)


def round_trips(bars: pd.DataFrame, target: np.ndarray) -> list[Trade]:
    """Round-trip trades at next-open fills; an open trade is closed at the last close."""
    held = np.r_[0.0, target[:-1]]
    opens = bars["open"].to_numpy(dtype=float)
    trades: list[Trade] = []
    side, entry_i = 0, 0
    for i in range(len(held)):
        if held[i] != side:
            if side != 0:
                trades.append(Trade(bars.index[entry_i], bars.index[i], side, opens[entry_i], opens[i]))
            side, entry_i = int(held[i]), i
    if side != 0:
        trades.append(Trade(bars.index[entry_i], bars.index[-1], side, opens[entry_i], float(bars["close"].iloc[-1])))
    return trades
