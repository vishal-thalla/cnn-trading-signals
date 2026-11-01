"""Write experiment artefacts: JSON summary, Markdown table, CSVs and charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .imaging import DEFAULT_PERIODS  # noqa: E402
from .indicators import INDICATORS  # noqa: E402
from .labeling import CLASS_NAMES  # noqa: E402
from .pipeline import ExperimentResult  # noqa: E402


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _pct_rank(x: float | None) -> str:
    return "-" if x is None else ordinal(round(x * 100))


def markdown_table(summary: dict) -> str:
    oos = summary["out_of_sample"]
    lines = [
        f"### {summary['experiment']}",
        "",
        f"Out-of-sample: {oos['start'][:10]} to {oos['end'][:10]} ({oos['bars']} bars, "
        f"{len(summary['folds'])} walk-forward folds), cost {summary['costs_bps_per_side']:.2f} bp per unit turnover.",
        "",
        "| Strategy | Macro-F1 | Total return | CAGR | Sharpe | Sharpe pct. vs random | Max DD | Trades | Win rate "
        "| Exposure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    bh = summary["buy_and_hold"]
    lines.append(
        f"| Buy & hold | - | {_pct(bh['total_return'])} | {_pct(bh['cagr'])} | {bh['sharpe']:.2f} "
        f"| - | {_pct(bh['max_drawdown'])} | 1 | - | 100% |"
    )
    for name, entry in summary["models"].items():
        f1 = entry["classification"]["macro_f1"]
        for mode in ("long_only", "long_short"):
            p = entry[mode]
            lines.append(
                f"| {name} ({mode.replace('_', '-')}) | {f1:.3f} | {_pct(p['total_return'])} | {_pct(p['cagr'])} "
                f"| {p['sharpe']:.2f} | {_pct_rank(p.get('sharpe_percentile_vs_random'))} "
                f"| {_pct(p['max_drawdown'])} | {p['trades']} | {p['win_rate'] * 100:.0f}% "
                f"| {p['exposure'] * 100:.0f}% |"
            )
    for mode, band in summary.get("random_benchmark", {}).items():
        lines.append("")
        lines.append(
            f"Random {mode.replace('_', '-')} strategies ({band['trials']} trials): mean Sharpe "
            f"{band['sharpe_mean']:.2f}, 5th-95th percentile {band['sharpe_p05']:.2f} to {band['sharpe_p95']:.2f}."
        )
    return "\n".join(lines) + "\n"


def plot_equity(result: ExperimentResult, path: Path, mode: str = "long_only") -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    for key, rets in result.returns.items():
        if key != "buy_and_hold" and not key.endswith(mode):
            continue
        label = "buy & hold" if key == "buy_and_hold" else key.split(":")[0]
        style = {"color": "0.35", "lw": 1.6, "ls": "--"} if key == "buy_and_hold" else {"lw": 1.4}
        ax.plot(rets.index, (1 + rets).cumprod(), label=label, **style)
    ax.axhline(1.0, color="0.7", lw=0.8)
    ax.set_title(f"{result.config.name}: out-of-sample equity ({mode.replace('_', '-')}, after costs)")
    ax.set_ylabel("growth of 1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_class_images(images: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Average indicator image per class, each cell z-scored across all bars (what the CNN sees)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), dpi=130, sharey=True)
    z = (images - images.mean(axis=0)) / (images.std(axis=0) + 1e-9)
    vmax = 1.5
    for cls, ax in enumerate(axes):
        sel = z[labels == cls]
        mean = sel.mean(axis=0) if len(sel) else np.zeros(images.shape[1:])
        im = ax.imshow(mean, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{CLASS_NAMES[cls]} (n={len(sel)})")
        ax.set_xticks(range(0, len(DEFAULT_PERIODS), 2), [str(p) for p in DEFAULT_PERIODS[::2]], fontsize=7)
        ax.set_xlabel("look-back period")
    axes[0].set_yticks(range(len(INDICATORS)), list(INDICATORS), fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean z-score")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(result: ExperimentResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result.summary, indent=2))
    (out / "results.md").write_text(markdown_table(result.summary))

    preds = pd.DataFrame(index=result.test_bars.index)
    preds["close"] = result.test_bars["close"]
    preds["label"] = [CLASS_NAMES[i] for i in result.labels]
    for name, proba in result.probabilities.items():
        for i, cls in enumerate(CLASS_NAMES):
            preds[f"{name}_p_{cls.lower()}"] = proba[:, i]
    preds.to_csv(out / "predictions.csv", float_format="%.6f")
    pd.DataFrame({k: (1 + v).cumprod() for k, v in result.returns.items()}).to_csv(
        out / "equity.csv", float_format="%.6f"
    )

    plot_equity(result, out / "equity_long_only.png", "long_only")
    plot_equity(result, out / "equity_long_short.png", "long_short")
    return out
