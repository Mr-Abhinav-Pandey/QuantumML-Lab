"""Classification metrics for binary classification.

All functions accept arrays of true labels and predicted
probabilities / labels.
"""

from __future__ import annotations

import numpy as np


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Binary cross‑entropy loss with ε‑clipping.

    L = −(1/N) Σ [ yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ) ]
    """
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc, bce
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    y_int = y_true.astype(int)

    # Basic counts
    tp = int(np.sum((y_pred == 1) & (y_int == 1)))
    tn = int(np.sum((y_pred == 0) & (y_int == 0)))
    fp = int(np.sum((y_pred == 1) & (y_int == 0)))
    fn = int(np.sum((y_pred == 0) & (y_int == 1)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    recall = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # ROC-AUC (simple trapezoidal)
    roc_auc = _compute_roc_auc(y_int, y_prob)

    bce = binary_cross_entropy(y_true, y_prob)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "bce": float(bce),
    }


def _compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC-AUC using the Wilcoxon–Mann–Whitney statistic.

    Returns 0.5 when classes are not separable or only one class is
    present (avoids division‑by‑zero).
    """
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Wilcoxon statistic
    auc = 0.0
    for p in pos:
        auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(auc / (len(pos) * len(neg)))
