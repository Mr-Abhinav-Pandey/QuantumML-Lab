import numpy as np
from quantumml.circuit import apply_single, ry, apply_cnot, forward_one, measurement_probabilities


def test_ry_preserves_norm():
    state = np.array([1, 0, 0, 0], dtype=complex)
    result = apply_single(state, ry(0.73), 0)
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_cnot_preserves_norm():
    state = np.array([0, 1, 0, 0], dtype=complex)
    result = apply_cnot(state)
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_probabilities_sum_to_one():
    theta = np.zeros((2, 4))
    probs = measurement_probabilities(np.array([0.2, 0.7]), theta)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)


def test_expectation_in_range():
    value, _ = forward_one(np.array([0.3, 0.8]), np.zeros((1, 4)))
    assert -1.0 <= value <= 1.0
