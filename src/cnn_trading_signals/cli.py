"""Command line interface: ``cts run CONFIG`` and ``cts images CONFIG``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ExperimentConfig


def _load(path: str, seed: int | None) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(path)
    if seed is not None:
        cfg.seed = seed
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_experiment
    from .report import markdown_table, write_report

    cfg = _load(args.config, args.seed)
    result = run_experiment(cfg)
    out = write_report(result, Path(args.out) / cfg.name)
    print(markdown_table(result.summary))
    print(f"artefacts written to {out}")
    return 0


def cmd_images(args: argparse.Namespace) -> int:
    from .data import load_bars
    from .pipeline import build_dataset
    from .report import plot_class_images

    cfg = _load(args.config, None)
    ds = build_dataset(load_bars(cfg.data), cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_class_images(ds.images, ds.labels, out)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cts", description="CNN trading-signal research pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a walk-forward experiment and write a report")
    run.add_argument("config")
    run.add_argument("--out", default="reports")
    run.add_argument("--seed", type=int, default=None, help="override the config seed")
    run.set_defaults(func=cmd_run)

    images = sub.add_parser("images", help="plot the average indicator image per class")
    images.add_argument("config")
    images.add_argument("--out", default="docs/class_images.png")
    images.set_defaults(func=cmd_images)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
