"""Small, educational statevector simulator for a two-qubit VQC."""
from __future__ import annotations
import numpy as np

I2 = np.eye(2, dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex)


def apply_single(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int = 2) -> np.ndarray:
    """Apply a 2x2 gate; qubit 0 is the left/most-significant tensor factor."""
    tensor = state.reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    return np.moveaxis(updated, 0, qubit).reshape(-1)


def apply_cnot(state: np.ndarray, control: int = 0, target: int = 1) -> np.ndarray:
    out = state.copy()
    for index in range(len(state)):
        bits = [(index >> (1 - q)) & 1 for q in range(2)]
        if bits[control] == 1 and bits[target] == 0:
            other = index ^ (1 << (1 - target))
            out[index], out[other] = state[other], state[index]
    return out


def encode_features(x: np.ndarray) -> np.ndarray:
    """Angle encode two normalized features into |00> using RY rotations."""
    state = np.array([1, 0, 0, 0], dtype=complex)
    state = apply_single(state, ry(float(np.pi * x[0])), 0)
    state = apply_single(state, ry(float(np.pi * x[1])), 1)
    return state


def forward_one(x: np.ndarray, theta: np.ndarray, noise_std: float = 0.0) -> tuple[float, np.ndarray]:
    """Return <Z0> and final state. theta shape: (layers, 4)."""
    state = encode_features(x)
    for layer in theta:
        state = apply_single(state, ry(float(layer[0])), 0)
        state = apply_single(state, rz(float(layer[1])), 0)
        state = apply_single(state, ry(float(layer[2])), 1)
        state = apply_single(state, rz(float(layer[3])), 1)
        state = apply_cnot(state, 0, 1)
    probabilities = np.abs(state) ** 2
    z0 = sum((1 if ((i >> 1) & 1) == 0 else -1) * probabilities[i] for i in range(4))
    if noise_std > 0:
        z0 = float(np.clip(z0 + np.random.normal(0, noise_std), -1, 1))
    return float(np.clip(z0, -1, 1)), state


def predict_expectations(X: np.ndarray, theta: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
    return np.array([forward_one(x, theta, noise_std)[0] for x in X])


def measurement_probabilities(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    _, state = forward_one(x, theta)
    return np.abs(state) ** 2


def parameter_shift_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Gradient of binary cross entropy using the parameter-shift rule.

    p(class 1)=(1-<Z0>)/2. Each trainable RY/RZ angle uses the exact
    two-evaluation parameter-shift derivative for a Pauli rotation.
    """
    grad = np.zeros_like(theta)
    for index in np.ndindex(theta.shape):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2
        exp_plus = predict_expectations(X, plus)
        exp_minus = predict_expectations(X, minus)
        d_exp = 0.5 * (exp_plus - exp_minus)
        probs = np.clip((1 - predict_expectations(X, theta)) / 2, epsilon, 1 - epsilon)
        dloss_dp = (probs - y) / (probs * (1 - probs))
        dp_dexp = -0.5
        grad[index] = np.mean(dloss_dp * dp_dexp * d_exp)
    return grad


def binary_cross_entropy(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-8, 1 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
