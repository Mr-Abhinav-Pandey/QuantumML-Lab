from __future__ import annotations
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_dataset(name: str, samples: int = 180, noise: float = 0.12, seed: int = 7):
    if name == "Two moons":
        X, y = make_moons(n_samples=samples, noise=noise, random_state=seed)
    elif name == "Concentric circles":
        X, y = make_circles(n_samples=samples, noise=noise, factor=0.45, random_state=seed)
    else:
        X, y = make_classification(n_samples=samples, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   class_sep=0.9, flip_y=noise, random_state=seed)
    scaler = MinMaxScaler(feature_range=(0.05, 0.95))
    X = scaler.fit_transform(X)
    return train_test_split(X, y.astype(float), test_size=0.25, random_state=seed, stratify=y)
