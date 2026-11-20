"""Quantum gate matrices.

All gates return 2×2 complex NumPy arrays.  Rotation gates use the
convention  R_σ(θ) = exp(−i θ σ/2)  where σ is the corresponding
Pauli matrix.

Qubit‑indexing convention used throughout the project:
    qubit 0 is the most‑significant bit (MSB).
    |q₀ q₁ … qₙ₋₁⟩  →  computational‑basis index
        = q₀·2^(n−1) + q₁·2^(n−2) + … + qₙ₋₁·2⁰
"""

from __future__ import annotations

import numpy as np


# ── fixed gates ──────────────────────────────────────────────────────

def identity_gate() -> np.ndarray:
    """2×2 identity gate."""
    return np.eye(2, dtype=complex)


def pauli_x() -> np.ndarray:
    """Pauli‑X (NOT) gate."""
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)


def pauli_y() -> np.ndarray:
    """Pauli‑Y gate."""
    return np.array([[0, -1j],
                     [1j,  0]], dtype=complex)


def pauli_z() -> np.ndarray:
    """Pauli‑Z gate."""
    return np.array([[ 1,  0],
                     [ 0, -1]], dtype=complex)


def hadamard() -> np.ndarray:
    """Hadamard gate  H = (X + Z)/√2."""
    return np.array([[1,  1],
                     [1, -1]], dtype=complex) / np.sqrt(2)


# ── parametric rotation gates ────────────────────────────────────────

def rx(theta: float) -> np.ndarray:
    """R_X(θ) = exp(−i θ X/2).

    Matrix:
        [[   cos(θ/2), −i sin(θ/2)],
         [−i sin(θ/2),    cos(θ/2)]]
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s],
                     [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    """R_Y(θ) = exp(−i θ Y/2).

    Matrix:
        [[cos(θ/2), −sin(θ/2)],
         [sin(θ/2),  cos(θ/2)]]
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[ c, -s],
                     [ s,  c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    """R_Z(θ) = exp(−i θ Z/2).

    Matrix:
        [[exp(−iθ/2),          0],
         [         0, exp(+iθ/2)]]
    """
    return np.array([[np.exp(-0.5j * theta), 0],
                     [0, np.exp(0.5j * theta)]], dtype=complex)
