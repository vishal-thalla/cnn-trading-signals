"""Classifiers that map indicator images to BUY/HOLD/SELL probabilities.

All models share one interface: ``fit(X, y)`` and ``predict_proba(X)`` where
``X`` has shape ``(n, rows, cols)`` and the output has one column per class.
Feature standardisation is fit on the training fold only.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .labeling import CLASS_NAMES

N_CLASSES = len(CLASS_NAMES)


class Model(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def _full_proba(classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Expand probabilities to all classes when a fold lacked some class."""
    out = np.zeros((proba.shape[0], N_CLASSES))
    out[:, classes.astype(int)] = proba
    return out


class SklearnModel:
    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnModel:
        self.estimator.fit(X.reshape(len(X), -1), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.estimator.predict_proba(X.reshape(len(X), -1))
        return _full_proba(self.estimator.classes_, proba)


def logistic_regression(seed: int = 0, C: float = 0.1) -> SklearnModel:
    est = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    return SklearnModel("logreg", est)


def gradient_boosting(seed: int = 0, max_iter: int = 200, learning_rate: float = 0.05) -> SklearnModel:
    est = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )
    return SklearnModel("hgb", est)


class PriorModel:
    """Predicts the training class frequencies for every row (a no-skill baseline)."""

    name = "prior"

    def fit(self, X: np.ndarray, y: np.ndarray) -> PriorModel:
        counts = np.bincount(y, minlength=N_CLASSES).astype(float)
        self.prior_ = counts / counts.sum()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.prior_, (len(X), 1))


class RandomModel:
    """Random BUY/HOLD/SELL calls drawn at the training class frequencies."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomModel:
        counts = np.bincount(y, minlength=N_CLASSES).astype(float)
        self.prior_ = counts / counts.sum()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        picks = self.rng.choice(N_CLASSES, size=len(X), p=self.prior_)
        return np.eye(N_CLASSES)[picks]


class CNNModel:
    """The two-conv-layer CNN from the original project, in PyTorch.

    Conv(32, 3x3) -> Conv(64, 3x3) -> MaxPool(2) -> Dropout(0.25) -> Dense(128)
    -> Dropout(0.5) -> softmax, trained with class-weighted cross-entropy. The
    last ``val_fraction`` of the (chronological) training fold is held out for
    early stopping.
    """

    name = "cnn"

    def __init__(
        self,
        seed: int = 0,
        epochs: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 8,
        val_fraction: float = 0.15,
        device: str = "cpu",
    ):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = device

    def _network(self, rows: int, cols: int):
        from torch import nn

        flat = 64 * ((rows - 4) // 2) * ((cols - 4) // 2)
        return nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, N_CLASSES),
        )

    def _tensor(self, X: np.ndarray):
        import torch

        Xs = (X - self.mean_) / self.std_
        return torch.as_tensor(Xs[:, None, :, :], dtype=torch.float32, device=self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> CNNModel:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("PyTorch is not installed: pip install 'cnn-trading-signals[cnn]'") from exc
        from torch import nn

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        n_val = max(int(len(X) * self.val_fraction), 1)
        X_tr, y_tr, X_val, y_val = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
        self.mean_ = X_tr.mean(axis=0, keepdims=True)
        self.std_ = X_tr.std(axis=0, keepdims=True) + 1e-6

        present = np.unique(y_tr)
        weights = np.ones(N_CLASSES)
        weights[present] = compute_class_weight("balanced", classes=present, y=y_tr)
        loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=self.device))

        self.net_ = self._network(X.shape[1], X.shape[2]).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt, yt = self._tensor(X_tr), torch.as_tensor(y_tr, device=self.device)
        xv, yv = self._tensor(X_val), torch.as_tensor(y_val, device=self.device)

        best_loss, best_state, stale = np.inf, None, 0
        self.history_: list[tuple[float, float]] = []
        for _ in range(self.epochs):
            self.net_.train()
            order = rng.permutation(len(xt))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = torch.as_tensor(order[start : start + self.batch_size], device=self.device)
                opt.zero_grad()
                loss = loss_fn(self.net_(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.net_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net_(xv), yv))
            self.history_.append((total / len(xt), val_loss))
            if val_loss < best_loss - 1e-4:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(self._tensor(X))
        return torch.softmax(logits, dim=1).cpu().numpy()


def build_model(name: str, seed: int = 0, **params) -> Model:
    factories = {
        "cnn": CNNModel,
        "logreg": logistic_regression,
        "hgb": gradient_boosting,
        "prior": lambda seed=0: PriorModel(),
        "random": RandomModel,
    }
    if name not in factories:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(factories)}")
    return factories[name](seed=seed, **params)
