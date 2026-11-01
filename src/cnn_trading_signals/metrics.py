"""Performance and classification metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .backtest import Trade
from .labeling import CLASS_NAMES

BARS_PER_YEAR = {
    "1d": 252,
    "1wk": 52,
    "1h": 252 * 24,  # FX trades ~24h on weekdays
    "4h": 252 * 6,
    "30m": 252 * 48,
    "15m": 252 * 96,
}


def performance(returns: pd.Series, bars_per_year: float, trades: list[Trade] | None = None) -> dict:
    """Return-series statistics. ``returns`` are simple per-bar returns."""
    r = returns.fillna(0.0).to_numpy(dtype=float)
    equity = np.cumprod(1.0 + r)
    total = float(equity[-1] - 1.0) if r.size else 0.0
    years = r.size / bars_per_year
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else float("nan")
    std = r.std(ddof=1) if r.size > 1 else 0.0
    downside = r[r < 0]
    down_std = np.sqrt(np.mean(downside**2)) if downside.size else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min()) if r.size else 0.0

    out = {
        "total_return": total,
        "cagr": cagr,
        "ann_volatility": float(std * np.sqrt(bars_per_year)),
        "sharpe": float(r.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0.0,
        "sortino": float(r.mean() / down_std * np.sqrt(bars_per_year)) if down_std > 0 else 0.0,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 and np.isfinite(cagr) else 0.0,
    }
    if trades is not None:
        wins = [t.gross_return > 0 for t in trades]
        out["trades"] = len(trades)
        out["win_rate"] = float(np.mean(wins)) if trades else 0.0
        out["avg_trade_return"] = float(np.mean([t.gross_return for t in trades])) if trades else 0.0
    return out


def exposure(target: np.ndarray) -> float:
    """Fraction of bars with a non-zero position."""
    return float(np.mean(target != 0)) if target.size else 0.0


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
