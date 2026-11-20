"""Optimisers for parameter updates.

Each optimiser implements a consistent interface:

    opt = Optimizer(lr=...)
    opt.step(params, grad)  →  updated_params
"""

from __future__ import annotations

import numpy as np


class GradientDescent:
    """Vanilla gradient descent:  θ ← θ − η·∇L."""

    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return params - self.lr * grad

    def reset(self) -> None:
        pass


class Adam:
    """Adam optimiser (Kingma & Ba, 2015).

    θ ← θ − η · m̂ / (√v̂ + ε)

    where m̂ and v̂ are bias‑corrected first‑ and second‑moment
    estimates of the gradient.
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t: int = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)

        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad ** 2

        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        self._m = None
        self._v = None
        self._t = 0
