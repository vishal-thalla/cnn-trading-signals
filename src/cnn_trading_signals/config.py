"""Experiment configuration (loaded from YAML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .imaging import DEFAULT_PERIODS, DEFAULT_WARMUP


@dataclass
class ModelSpec:
    name: str
    params: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    data: dict
    interval: str = "1d"
    label_window: int = 11
    periods: list[int] = field(default_factory=lambda: list(DEFAULT_PERIODS))
    warmup: int = DEFAULT_WARMUP
    folds: int = 6
    min_train_fraction: float = 0.5
    max_train: int | None = None
    embargo: int = 0
    models: list[ModelSpec] = field(default_factory=lambda: [ModelSpec("cnn"), ModelSpec("logreg")])
    spread_bps: float = 1.0
    commission_bps: float = 0.0
    min_confidence: float = 0.0
    random_trials: int = 200
    seed: int = 0

    @classmethod
    def from_dict(cls, raw: dict) -> ExperimentConfig:
        raw = dict(raw)
        raw["models"] = [ModelSpec(**m) if isinstance(m, dict) else ModelSpec(m) for m in raw.get("models", [])] or [
            ModelSpec("cnn"),
            ModelSpec("logreg"),
        ]
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))
