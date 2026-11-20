"""Classical machine learning baselines.

The classical models serve as a point of comparison for the VQC.
Outperforming the classical baseline is **not** a goal of this project;
the comparison illustrates the differences in model behaviour.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

from quantumml.training.metrics import compute_metrics


def train_classical_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "mlp",
    seed: int = 21,
) -> dict:
    """Train a classical baseline and return metrics.

    Parameters
    ----------
    model_type : {"mlp", "logistic"}
    seed : int

    Returns
    -------
    dict with keys: model, train_metrics, test_metrics, training_time
    """
    t0 = time.perf_counter()

    if model_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(8, 8),
            activation="tanh",
            solver="lbfgs",
            alpha=0.01,
            max_iter=700,
            random_state=seed,
        )
    elif model_type == "logistic":
        model = LogisticRegression(
            max_iter=1000, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "model_type": model_type,
        "train_metrics": compute_metrics(y_train, train_probs),
        "test_metrics": compute_metrics(y_test, test_probs),
        "test_probs": test_probs,
        "training_time": training_time,
    }
