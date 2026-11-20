"""VQC trainer — ties together the simulator, ansatz, gradients, and optimiser."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from quantumml.simulator.statevector import StatevectorSimulator
from quantumml.circuits.encoding import angle_encoding
from quantumml.circuits.ansatz import VariationalAnsatz
from quantumml.gradients.parameter_shift import parameter_shift_gradient
from quantumml.training.metrics import binary_cross_entropy, compute_metrics
from quantumml.training.optimizers import GradientDescent


class VQCTrainer:
    """High‑level trainer for a variational quantum classifier.

    Parameters
    ----------
    n_qubits : int
    n_layers : int
    entanglement : str
        Entanglement pattern for the ansatz.
    optimizer
        An object with a ``.step(params, grad)`` method.
    shots : int or None
        If *None*, use exact expectation values.  Otherwise, estimate
        ⟨Z₀⟩ from finite‑shot sampling.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        entanglement: str = "linear",
        optimizer=None,
        shots: Optional[int] = None,
        measurement_seed: Optional[int] = None,
    ) -> None:
        self.ansatz = VariationalAnsatz(n_qubits, n_layers, entanglement)
        self.optimizer = optimizer or GradientDescent(lr=0.1)
        self.shots = shots
        self.measurement_seed = measurement_seed

    # ── forward pass ─────────────────────────────────────────────────

    def expectation(self, x: np.ndarray, params: np.ndarray) -> float:
        """Compute ⟨Z₀⟩ for a single data point *x*."""
        sim = StatevectorSimulator(self.ansatz.n_qubits)
        angle_encoding(sim, x)
        self.ansatz.apply(sim, params)
        if self.shots is not None:
            return sim.expectation_value_z_shots(
                0, self.shots, seed=self.measurement_seed
            )
        return sim.expectation_value_z(0)

    def predict_proba(
        self, X: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """Return p(class=1) = (1 − ⟨Z₀⟩) / 2 for every row in *X*."""
        exps = np.array([self.expectation(x, params) for x in X])
        return np.clip((1 - exps) / 2, 0, 1)

    def predict(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Return predicted labels (0/1) for every row in *X*."""
        return (self.predict_proba(X, params) >= 0.5).astype(int)

    # ── cost function ────────────────────────────────────────────────

    def cost(
        self, params: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Binary cross‑entropy cost averaged over *X*, *y*."""
        proba = self.predict_proba(X, params)
        return binary_cross_entropy(y, proba)

    # ── training loop ────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: np.ndarray,
        epochs: int = 50,
        callback: Optional[Callable] = None,
    ) -> dict:
        """Train the VQC and return results dict.

        Parameters
        ----------
        X_train, y_train : ndarray
            Training data and binary labels.
        params : ndarray
            Initial parameters (will not be modified in place).
        epochs : int
            Number of training epochs.
        callback : callable, optional
            Called as ``callback(epoch, loss, accuracy)`` after each
            epoch.

        Returns
        -------
        dict with keys: ``final_params``, ``history``, ``training_time``
        """
        params = params.copy()
        history: list[dict] = []
        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            # Build a closure for the parameter‑shift rule
            def _cost_fn(p: np.ndarray) -> float:
                return self.cost(p, X_train, y_train)

            grad = parameter_shift_gradient(_cost_fn, params)
            params = self.optimizer.step(params, grad)

            # Evaluate
            proba = self.predict_proba(X_train, params)
            loss = binary_cross_entropy(y_train, proba)
            acc = float(np.mean((proba >= 0.5) == y_train))

            history.append({
                "epoch": epoch,
                "loss": loss,
                "accuracy": acc,
            })

            if callback is not None:
                callback(epoch, loss, acc)

        training_time = time.perf_counter() - t0

        return {
            "final_params": params,
            "history": history,
            "training_time": training_time,
        }

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> dict[str, float]:
        """Compute full classification metrics on a dataset."""
        proba = self.predict_proba(X, params)
        return compute_metrics(y, proba)
