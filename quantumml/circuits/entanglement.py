"""Entanglement patterns for variational quantum circuits.

Three architectures are supported:

* **linear**:    CNOT chain  0→1, 1→2, …, (n−2)→(n−1)
* **circular**:  linear + wrap‑around CNOT  (n−1)→0
* **all_to_all**: CNOT between every ordered pair  i→j  for i < j

For a single qubit, all patterns are no‑ops.
"""

from __future__ import annotations

from quantumml.simulator.statevector import StatevectorSimulator

ENTANGLEMENT_TYPES = ("linear", "circular", "all_to_all")


def _linear(sim: StatevectorSimulator) -> None:
    for i in range(sim.n_qubits - 1):
        sim.apply_cnot(i, i + 1)


def _circular(sim: StatevectorSimulator) -> None:
    _linear(sim)
    if sim.n_qubits >= 2:
        sim.apply_cnot(sim.n_qubits - 1, 0)


def _all_to_all(sim: StatevectorSimulator) -> None:
    n = sim.n_qubits
    for i in range(n):
        for j in range(i + 1, n):
            sim.apply_cnot(i, j)


_DISPATCH = {
    "linear": _linear,
    "circular": _circular,
    "all_to_all": _all_to_all,
}


def apply_entanglement(sim: StatevectorSimulator, pattern: str) -> None:
    """Apply an entanglement layer to the simulator.

    Parameters
    ----------
    sim : StatevectorSimulator
    pattern : {"linear", "circular", "all_to_all"}

    Raises
    ------
    ValueError
        If *pattern* is not recognised.
    """
    if pattern not in _DISPATCH:
        raise ValueError(
            f"Unknown entanglement pattern '{pattern}'. "
            f"Choose from {ENTANGLEMENT_TYPES}"
        )
    _DISPATCH[pattern](sim)
