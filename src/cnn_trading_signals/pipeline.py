"""End-to-end experiment: data -> images + labels -> walk-forward models -> backtests."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import CostModel, round_trips, signals_from_probabilities, strategy_returns, target_positions
from .config import ExperimentConfig
from .data import load_bars
from .imaging import finalize, indicator_images, min_warmup
from .labeling import CLASS_NAMES, label_horizon, turning_point_labels
from .metrics import BARS_PER_YEAR, classification_report, exposure, performance
from .models import build_model
from .splits import walk_forward

log = logging.getLogger(__name__)

MODES = {"long_only": False, "long_short": True}


@dataclass
class Dataset:
    bars: pd.DataFrame  # rows usable for modelling
    images: np.ndarray
    labels: np.ndarray


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    test_bars: pd.DataFrame
    labels: np.ndarray
    probabilities: dict[str, np.ndarray] = field(default_factory=dict)
    returns: dict[str, pd.Series] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def build_dataset(bars: pd.DataFrame, cfg: ExperimentConfig) -> Dataset:
    if cfg.warmup < min_warmup(cfg.periods):
        raise ValueError(f"warmup={cfg.warmup} is shorter than the {min_warmup(cfg.periods)} bars the periods need")
    images, usable = finalize(indicator_images(bars, cfg.periods), cfg.warmup)
    labels = turning_point_labels(bars["close"], cfg.label_window)
    keep = usable & (labels >= 0)  # labels are undefined only at the two ends
    return Dataset(bars=bars[keep], images=images[keep], labels=labels[keep])


def random_benchmark(
    labels: np.ndarray,
    folds: list,
    test_bars: pd.DataFrame,
    costs: CostModel,
    bars_per_year: float,
    trials: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sharpe ratios of ``trials`` random strategies with each fold's training class mix.

    A single random run is too noisy to compare against; the distribution says how
    often chance alone does as well as a model on the same bars, costs and folds.
    """
    priors = [np.bincount(labels[f.train], minlength=len(CLASS_NAMES)) / f.train.size for f in folds]
    sharpes: dict[str, list[float]] = {mode: [] for mode in MODES}
    for k in range(trials):
        rng = np.random.default_rng(10_000 + seed * 1_000 + k)
        signals = np.concatenate(
            [rng.choice(len(CLASS_NAMES), size=f.test.size, p=p) for f, p in zip(folds, priors, strict=True)]
        )
        for mode, allow_short in MODES.items():
            rets = strategy_returns(test_bars, target_positions(signals, allow_short), costs)
            sharpes[mode].append(performance(rets, bars_per_year)["sharpe"])
    return {mode: np.asarray(v) for mode, v in sharpes.items()}


def run_experiment(cfg: ExperimentConfig, bars: pd.DataFrame | None = None) -> ExperimentResult:
    bars = load_bars(cfg.data) if bars is None else bars
    ds = build_dataset(bars, cfg)
    n = len(ds.labels)
    horizon = label_horizon(cfg.label_window)
    folds = list(
        walk_forward(
            n,
            n_folds=cfg.folds,
            min_train=int(n * cfg.min_train_fraction),
            horizon=horizon,
            embargo=cfg.embargo,
            max_train=cfg.max_train,
        )
    )
    test_idx = np.concatenate([f.test for f in folds])
    test_bars = ds.bars.iloc[test_idx]
    y_test = ds.labels[test_idx]
    log.info(
        "%s: %d usable bars, %d folds, out-of-sample %s .. %s",
        cfg.name,
        n,
        len(folds),
        test_bars.index[0].date(),
        test_bars.index[-1].date(),
    )

    result = ExperimentResult(config=cfg, test_bars=test_bars, labels=y_test)
    bpy = BARS_PER_YEAR.get(cfg.interval, 252)
    costs = CostModel(cfg.spread_bps, cfg.commission_bps)

    summary: dict = {
        "experiment": cfg.name,
        "bars_total": int(len(bars)),
        "bars_usable": int(n),
        "out_of_sample": {
            "start": str(test_bars.index[0]),
            "end": str(test_bars.index[-1]),
            "bars": int(len(test_idx)),
        },
        "label_window": cfg.label_window,
        "label_balance_oos": {c: float(np.mean(y_test == i)) for i, c in enumerate(CLASS_NAMES)},
        "costs_bps_per_side": costs.per_unit_turnover * 1e4,
        "folds": [
            {
                "train": [str(ds.bars.index[f.train[0]].date()), str(ds.bars.index[f.train[-1]].date())],
                "test": [str(ds.bars.index[f.test[0]].date()), str(ds.bars.index[f.test[-1]].date())],
            }
            for f in folds
        ],
        "models": {},
    }

    # Buy & hold over the same out-of-sample bars, paying one entry.
    hold_target = np.ones(len(test_bars))
    bh = strategy_returns(test_bars, hold_target, costs)
    result.returns["buy_and_hold"] = bh
    summary["buy_and_hold"] = performance(bh, bpy)

    random_sharpes = random_benchmark(ds.labels, folds, test_bars, costs, bpy, cfg.random_trials, cfg.seed)
    summary["random_benchmark"] = {
        mode: {
            "trials": int(v.size),
            "sharpe_mean": float(v.mean()),
            "sharpe_p05": float(np.percentile(v, 5)),
            "sharpe_p95": float(np.percentile(v, 95)),
        }
        for mode, v in random_sharpes.items()
        if v.size
    }

    for spec in cfg.models:
        started = time.perf_counter()
        proba = np.zeros((len(test_idx), len(CLASS_NAMES)))
        offset = 0
        for fold_no, fold in enumerate(folds):
            model = build_model(spec.name, seed=cfg.seed + fold_no, **spec.params)
            model.fit(ds.images[fold.train], ds.labels[fold.train])
            proba[offset : offset + fold.test.size] = model.predict_proba(ds.images[fold.test])
            offset += fold.test.size
        result.probabilities[spec.name] = proba

        signals = signals_from_probabilities(proba, cfg.min_confidence)
        entry = {"classification": classification_report(y_test, signals), "fit_seconds": time.perf_counter() - started}
        for mode, allow_short in MODES.items():
            target = target_positions(signals, allow_short=allow_short)
            rets = strategy_returns(test_bars, target, costs)
            result.returns[f"{spec.name}:{mode}"] = rets
            perf = performance(rets, bpy, round_trips(test_bars, target))
            perf["exposure"] = exposure(target)
            if random_sharpes[mode].size:
                perf["sharpe_percentile_vs_random"] = float(np.mean(random_sharpes[mode] < perf["sharpe"]))
            entry[mode] = perf
        summary["models"][spec.name] = entry
        log.info(
            "%s/%s: macro-F1 %.3f, long-only Sharpe %.2f (%.1fs)",
            cfg.name,
            spec.name,
            entry["classification"]["macro_f1"],
            entry["long_only"]["sharpe"],
            entry["fit_seconds"],
        )

    result.summary = summary
    return result
