"""Parameter‑shift rule for rotation gates.

For a circuit whose only generators are Pauli operators (which is the
case for R_X, R_Y, R_Z):

    ∂f/∂θ_k = [ f(θ_k + π/2) − f(θ_k − π/2) ] / 2

This is an *exact* analytic formula, not a finite‑difference
approximation.  It requires two circuit evaluations per parameter.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def parameter_shift_gradient(
    cost_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Compute the gradient of *cost_fn* using the parameter‑shift rule.

    Parameters
    ----------
    cost_fn : callable
        A function  params → scalar  that evaluates the cost.
    params : ndarray
        Current parameter values (arbitrary shape).
    shift : float
        Shift amount.  Default π/2 is exact for Pauli‑rotation gates.

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
        plus[i] += shift
        minus[i] -= shift
        f_plus = cost_fn(plus.reshape(params.shape))
        f_minus = cost_fn(minus.reshape(params.shape))
        grad.flat[i] = (f_plus - f_minus) / (2 * np.sin(shift))

    return grad
