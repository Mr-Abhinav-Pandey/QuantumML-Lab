from __future__ import annotations
import numpy as np
from .circuit import predict_expectations, parameter_shift_gradient, binary_cross_entropy


def train_vqc(X, y, layers=2, epochs=35, learning_rate=0.08, seed=21, callback=None):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.25, size=(layers, 4))
    history = []
    for epoch in range(epochs):
        expectations = predict_expectations(X, theta)
        probabilities = (1 - expectations) / 2
        loss = binary_cross_entropy(y, probabilities)
        accuracy = float(np.mean((probabilities >= 0.5) == y))
        history.append({"epoch": epoch + 1, "loss": loss, "accuracy": accuracy})
        gradient = parameter_shift_gradient(X, y, theta)
        theta -= learning_rate * gradient
        if callback:
            callback(epoch + 1, loss, accuracy)
    return theta, history


def predict_labels(X, theta, noise_std=0.0):
    probs = (1 - predict_expectations(X, theta, noise_std=noise_std)) / 2
    return (probs >= 0.5).astype(int), probs
