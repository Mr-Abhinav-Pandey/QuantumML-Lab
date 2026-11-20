"""Variational ansatz for parameterised quantum circuits.

Each variational layer applies:
    1.  R_Y(θ_{l,q,0}) on every qubit q
    2.  R_Z(θ_{l,q,1}) on every qubit q
    3.  Entanglement layer (CNOT pattern)

Parameter tensor shape: (n_layers, n_qubits, 2).

The ansatz is separated from data encoding: call ``angle_encoding``
*before* applying the ansatz.
"""

from __future__ import annotations

import numpy as np

from quantumml.simulator.statevector import StatevectorSimulator
from quantumml.simulator.gates import ry, rz
from quantumml.circuits.entanglement import apply_entanglement, ENTANGLEMENT_TYPES


class VariationalAnsatz:
    """Configurable hardware‑efficient variational ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers (≥ 1).
    entanglement : str
        Entanglement pattern — one of ``ENTANGLEMENT_TYPES``.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        entanglement: str = "linear",
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be ≥ 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {n_layers}")
        if entanglement not in ENTANGLEMENT_TYPES:
            raise ValueError(
                f"entanglement must be one of {ENTANGLEMENT_TYPES}, "
                f"got '{entanglement}'"
            )
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return self.n_layers * self.n_qubits * 2

    @property
    def parameter_shape(self) -> tuple[int, int, int]:
        """Canonical shape of the parameter tensor: (layers, qubits, 2)."""
        return (self.n_layers, self.n_qubits, 2)

    def apply(
        self,
        sim: StatevectorSimulator,
        params: np.ndarray,
    ) -> None:
        """Apply the variational ansatz to the simulator state.

        Parameters
        ----------
        sim : StatevectorSimulator
            Simulator (modified in place).
        params : ndarray
            Parameter array — either shape ``parameter_shape`` or flat
            array of length ``num_parameters``.
        """
        params = np.asarray(params, dtype=float)
        expected = self.parameter_shape
        if params.ndim == 1:
            if params.size != self.num_parameters:
                raise ValueError(
                    f"Flat params length {params.size} != "
                    f"expected {self.num_parameters}"
                )
            params = params.reshape(expected)
        elif params.shape != expected:
            raise ValueError(
                f"params shape {params.shape} != expected {expected}"
            )

        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                sim.apply_single_qubit_gate(ry(float(params[l, q, 0])), q)
                sim.apply_single_qubit_gate(rz(float(params[l, q, 1])), q)
            if self.n_qubits >= 2:
                apply_entanglement(sim, self.entanglement)

    def describe(self) -> dict:
        """Return a JSON‑serialisable description of the ansatz."""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "entanglement": self.entanglement,
            "num_parameters": self.num_parameters,
            "gates_per_layer": f"{self.n_qubits}×RY + {self.n_qubits}×RZ + entanglement",
        }

    def __repr__(self) -> str:
        return (
            f"VariationalAnsatz(n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, entanglement='{self.entanglement}', "
            f"params={self.num_parameters})"
        )
