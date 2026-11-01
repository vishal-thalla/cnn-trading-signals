import json

import numpy as np
import pytest

from cnn_trading_signals.cli import main
from cnn_trading_signals.config import ExperimentConfig
from cnn_trading_signals.models import build_model
from cnn_trading_signals.pipeline import run_experiment


def _config(models):
    return ExperimentConfig.from_dict(
        {
            "name": "synthetic-test",
            "data": {"type": "synthetic", "n": 900, "seed": 5},
            "label_window": 11,
            "folds": 3,
            "min_train_fraction": 0.5,
            "models": models,
            "random_trials": 10,
        }
    )


def test_pipeline_runs_end_to_end_with_baselines():
    result = run_experiment(_config(["logreg", "prior", "random"]))
    s = result.summary
    assert set(s["models"]) == {"logreg", "prior", "random"}
    assert s["out_of_sample"]["bars"] == len(result.test_bars)
    # The prior model always predicts HOLD, so it never trades.
    assert s["models"]["prior"]["long_only"]["trades"] == 0
    for name, proba in result.probabilities.items():
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6, err_msg=name)
    json.dumps(s)  # serialisable


def test_cnn_learns_a_separable_toy_problem():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=900)
    X = rng.normal(0, 1, size=(900, 15, 15)).astype(np.float32)
    X[:, 0, :] += (y[:, None] - 1) * 2.0  # class signal in the first image row
    model = build_model("cnn", seed=0, epochs=15).fit(X[:700], y[:700])
    acc = (model.predict_proba(X[700:]).argmax(axis=1) == y[700:]).mean()
    assert acc > 0.9


def test_cli_run_writes_artifacts(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("name: cli-test\ndata: {type: synthetic, n: 700, seed: 1}\nfolds: 2\nmodels: [logreg]\n")
    assert main(["run", str(cfg), "--out", str(tmp_path / "reports")]) == 0
    out = tmp_path / "reports" / "cli-test"
    for name in ("summary.json", "results.md", "predictions.csv", "equity.csv", "equity_long_only.png"):
        assert (out / name).exists(), name


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown config keys"):
        ExperimentConfig.from_dict({"name": "x", "data": {"type": "synthetic"}, "typo": 1})


def test_random_benchmark_reports_a_distribution_and_percentiles():
    cfg = _config(["logreg"])
    cfg.random_trials = 25
    s = run_experiment(cfg).summary
    band = s["random_benchmark"]["long_only"]
    assert band["trials"] == 25
    assert band["sharpe_p05"] <= band["sharpe_mean"] <= band["sharpe_p95"]
    assert 0.0 <= s["models"]["logreg"]["long_only"]["sharpe_percentile_vs_random"] <= 1.0


@pytest.mark.parametrize(
    ("n", "text"), [(1, "1st"), (2, "2nd"), (3, "3rd"), (11, "11th"), (12, "12th"), (53, "53rd"), (100, "100th")]
)
def test_ordinal_suffixes(n, text):
    from cnn_trading_signals.report import ordinal

    assert ordinal(n) == text
