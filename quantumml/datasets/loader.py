"""Dataset loading with consistent preprocessing.

Features are scaled to [0.05, 0.95] so that angle‑encoding
RY(π·x) produces rotations in a useful range (avoids exact 0 and π).
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATASET_NAMES = ("Two moons", "Concentric circles", "Linearly separable")


def load_dataset(
    name: str = "Two moons",
    samples: int = 180,
    noise: float = 0.12,
    seed: int = 7,
    test_size: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate and split a 2‑D binary classification dataset.

    Parameters
    ----------
    name : str
        One of ``DATASET_NAMES``.
    samples : int
        Total sample count.
    noise : float
        Noise level.
    seed : int
        Random seed for reproducibility.
    test_size : float
        Fraction of data for testing.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if name == "Two moons":
        X, y = make_moons(n_samples=samples, noise=noise, random_state=seed)
    elif name == "Concentric circles":
        X, y = make_circles(n_samples=samples, noise=noise, factor=0.45,
                            random_state=seed)
    elif name == "Linearly separable":
        X, y = make_classification(
            n_samples=samples, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1,
            class_sep=0.9, flip_y=noise, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASET_NAMES}")

    scaler = MinMaxScaler(feature_range=(0.05, 0.95))
    X = scaler.fit_transform(X)

    return train_test_split(
        X, y.astype(float), test_size=test_size,
        random_state=seed, stratify=y,
    )
