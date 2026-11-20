"""N‑qubit statevector simulator.

The simulator stores the full 2ⁿ‑dimensional complex state vector and
applies gates via efficient tensor‑contraction (single‑qubit) or
index‑permutation (CNOT).  No large dense 2ⁿ×2ⁿ matrices are formed
for individual gate operations.

Qubit indexing: qubit 0 is the most‑significant bit (MSB).
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from . import gates as _g


class StatevectorSimulator:
    """Exact statevector simulator for *n* qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be ≥ 1).  Memory scales as 2ⁿ complex128.

    Raises
    ------
    ValueError
        If *n_qubits* < 1 or > 20 (safety limit ≈ 16 MiB state).
    """

    _MAX_QUBITS = 20  # 2^20 complex128 ≈ 16 MiB

    def __init__(self, n_qubits: int) -> None:
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}")
        if n_qubits > self._MAX_QUBITS:
            raise ValueError(
                f"n_qubits={n_qubits} exceeds safety limit of {self._MAX_QUBITS} "
                f"(would need {2**n_qubits * 16 / 1e6:.0f} MB)"
            )
        self._n_qubits: int = n_qubits
        self._dim: int = 2 ** n_qubits
        self._state: np.ndarray = np.zeros(self._dim, dtype=complex)
        self._state[0] = 1.0  # |00…0⟩

    # ── properties ───────────────────────────────────────────────────

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def state(self) -> np.ndarray:
        """Return a *copy* of the current state vector."""
        return self._state.copy()

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset to |00…0⟩."""
        self._state[:] = 0.0
        self._state[0] = 1.0

    def copy(self) -> "StatevectorSimulator":
        """Deep‑copy the simulator (independent state)."""
        new = StatevectorSimulator.__new__(StatevectorSimulator)
        new._n_qubits = self._n_qubits
        new._dim = self._dim
        new._state = self._state.copy()
        return new

    def set_state(self, state: np.ndarray) -> None:
        """Replace the internal state (must be normalised)."""
        state = np.asarray(state, dtype=complex).ravel()
        if state.shape[0] != self._dim:
            raise ValueError(
                f"State dimension {state.shape[0]} != 2^{self._n_qubits} = {self._dim}"
            )
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0, atol=1e-10):
            raise ValueError(f"State is not normalised (norm = {norm})")
        self._state = state.copy()

    # ── qubit‑index validation ───────────────────────────────────────

    def _check_qubit(self, qubit: int, name: str = "qubit") -> None:
        if not (0 <= qubit < self._n_qubits):
            raise ValueError(
                f"{name}={qubit} out of range for {self._n_qubits}‑qubit system"
            )

    # ── single‑qubit gate ───────────────────────────────────────────

    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a 2×2 unitary to the specified qubit.

        Uses tensor contraction — no 2ⁿ×2ⁿ matrix is formed.
        """
        self._check_qubit(qubit)
        gate = np.asarray(gate, dtype=complex)
        if gate.shape != (2, 2):
            raise ValueError(f"Gate must be 2×2, got {gate.shape}")

        tensor = self._state.reshape((2,) * self._n_qubits)
        moved = np.moveaxis(tensor, qubit, 0)
        updated = np.tensordot(gate, moved, axes=([1], [0]))
        self._state = np.moveaxis(updated, 0, qubit).reshape(-1)

    # ── two‑qubit CNOT ──────────────────────────────────────────────

    def apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT (controlled‑X) gate.

        Uses vectorised index permutation — no 4×4 or 2ⁿ×2ⁿ matrix.
        """
        self._check_qubit(control, "control")
        self._check_qubit(target, "target")
        if control == target:
            raise ValueError("control and target must differ")

        n = self._n_qubits
        ctrl_shift = n - 1 - control
        tgt_shift = n - 1 - target

        indices = np.arange(self._dim)
        ctrl_bits = (indices >> ctrl_shift) & 1
        new_indices = np.where(ctrl_bits == 1,
                               indices ^ (1 << tgt_shift),
                               indices)

        new_state = np.empty_like(self._state)
        new_state[new_indices] = self._state
        self._state = new_state

    # ── convenience gate methods ────────────────────────────────────

    def h(self, qubit: int) -> None:
        """Hadamard gate on *qubit*."""
        self.apply_single_qubit_gate(_g.hadamard(), qubit)

    def x(self, qubit: int) -> None:
        """Pauli‑X gate on *qubit*."""
        self.apply_single_qubit_gate(_g.pauli_x(), qubit)

    def y(self, qubit: int) -> None:
        """Pauli‑Y gate on *qubit*."""
        self.apply_single_qubit_gate(_g.pauli_y(), qubit)

    def z(self, qubit: int) -> None:
        """Pauli‑Z gate on *qubit*."""
        self.apply_single_qubit_gate(_g.pauli_z(), qubit)

    def rx_gate(self, theta: float, qubit: int) -> None:
        self.apply_single_qubit_gate(_g.rx(theta), qubit)

    def ry_gate(self, theta: float, qubit: int) -> None:
        self.apply_single_qubit_gate(_g.ry(theta), qubit)

    def rz_gate(self, theta: float, qubit: int) -> None:
        self.apply_single_qubit_gate(_g.rz(theta), qubit)

    # ── measurement ──────────────────────────────────────────────────

    def get_probabilities(self) -> np.ndarray:
        """Return Born‑rule probabilities for all 2ⁿ basis states."""
        return np.abs(self._state) ** 2

    def expectation_value_z(self, qubit: int) -> float:
        """⟨Z_qubit⟩ — expectation value of Pauli‑Z on one qubit."""
        self._check_qubit(qubit)
        probs = self.get_probabilities()
        n = self._n_qubits
        shift = n - 1 - qubit
        indices = np.arange(self._dim)
        bits = (indices >> shift) & 1
        signs = 1 - 2 * bits  # +1 for |0⟩, −1 for |1⟩
        return float(np.sum(signs * probs))

    def sample(self, shots: int, seed: Optional[int] = None) -> np.ndarray:
        """Finite‑shot sampling from the probability distribution.

        Returns
        -------
        counts : ndarray of shape (2**n,)
            Histogram of measured basis‑state outcomes.
        """
        rng = np.random.default_rng(seed)
        probs = self.get_probabilities()
        outcomes = rng.choice(self._dim, size=shots, p=probs)
        counts = np.bincount(outcomes, minlength=self._dim)
        return counts

    def expectation_value_z_shots(
        self, qubit: int, shots: int, seed: Optional[int] = None
    ) -> float:
        """Estimate ⟨Z_qubit⟩ from finite‑shot sampling."""
        self._check_qubit(qubit)
        counts = self.sample(shots, seed=seed)
        n = self._n_qubits
        shift = n - 1 - qubit
        indices = np.arange(self._dim)
        bits = (indices >> shift) & 1
        signs = 1 - 2 * bits
        return float(np.sum(signs * counts)) / shots

    # ── diagnostics ──────────────────────────────────────────────────

    def norm(self) -> float:
        return float(np.linalg.norm(self._state))

    def is_normalised(self, atol: float = 1e-10) -> bool:
        return np.isclose(self.norm(), 1.0, atol=atol)

    def __repr__(self) -> str:
        return (
            f"StatevectorSimulator(n_qubits={self._n_qubits}, "
            f"dim={self._dim}, norm={self.norm():.6f})"
        )
