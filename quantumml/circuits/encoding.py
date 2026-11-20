"""Data‑encoding strategies for variational quantum circuits.

Currently implements angle encoding: each classical feature x_i is
encoded as an R_Y(π · x_i) rotation on qubit i.  Features beyond
n_qubits are ignored.
"""

from __future__ import annotations

import numpy as np

from quantumml.simulator.statevector import StatevectorSimulator
from quantumml.simulator.gates import ry


def angle_encoding(sim: StatevectorSimulator, x: np.ndarray) -> None:
    """Encode classical features into the quantum state using RY(π·xᵢ).

    Parameters
    ----------
    sim : StatevectorSimulator
        Simulator instance (modified in place).
    x : array‑like of shape (n_features,)
        Classical feature vector.  Values should be in [0, 1] for
        meaningful angle range.  At most sim.n_qubits features are
        encoded; extras are silently ignored.
    """
    x = np.asarray(x, dtype=float)
    n_encode = min(len(x), sim.n_qubits)
    for i in range(n_encode):
        sim.apply_single_qubit_gate(ry(float(np.pi * x[i])), i)
