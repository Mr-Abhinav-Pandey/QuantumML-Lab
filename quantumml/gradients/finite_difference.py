"""Central finite‑difference gradient (for validation only).

This is **not** used for training.  It serves as a numerical reference
to verify that the parameter‑shift implementation is correct.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def finite_difference_gradient(
    cost_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Central finite‑difference gradient of *cost_fn*.

    Parameters
    ----------
    cost_fn : callable
        A function  params → scalar.
    params : ndarray
        Current parameter values (arbitrary shape).
    epsilon : float
        Step size for finite differences.

    Returns
    -------
    grad : ndarray
        Gradient array with the same shape as *params*.
    """
    params = np.asarray(params, dtype=float)
    grad = np.zeros_like(params)
    flat = params.ravel()

    for i in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[i] += epsilon
        minus[i] -= epsilon
        grad.flat[i] = (
            cost_fn(plus.reshape(params.shape))
            - cost_fn(minus.reshape(params.shape))
        ) / (2 * epsilon)

    return grad
