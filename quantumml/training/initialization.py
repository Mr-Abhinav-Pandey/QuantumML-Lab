"""Parameter initialisation strategies.

Three strategies are provided:

1. **random** — Uniform(0, 2π).
2. **small_angle** — Normal(0, σ) with σ = 0.01 by default.
3. **identity_block** — Paired layers whose rotation angles approximately
   cancel so the initial circuit acts close to the identity.

All strategies are reproducible given the same seed.
"""

from __future__ import annotations

import numpy as np

INIT_STRATEGIES = ("random", "small_angle", "identity_block")


def initialize_parameters(
    shape: tuple[int, ...],
    strategy: str = "random",
    seed: int = 42,
    *,
    sigma: float = 0.01,
) -> np.ndarray:
    """Create an initial parameter array.

    Parameters
    ----------
    shape : tuple
        Canonical shape (n_layers, n_qubits, 2).
    strategy : str
        One of ``INIT_STRATEGIES``.
    seed : int
        Random seed for reproducibility.
    sigma : float
        Scale parameter used by *small_angle* and *identity_block*.

    Returns
    -------
    params : ndarray of given shape

    Mathematical details
    --------------------
    **random**:
        θ ~ Uniform(0, 2π).  Standard random initialisation.  Under
        Haar‑random conditions and large system sizes, this may lead
        to barren‑plateau behaviour (McClean et al., 2018).

    **small_angle**:
        θ ~ Normal(0, σ²).  All rotations start near the identity
        gate: R_Y(≈0) ≈ I, R_Z(≈0) ≈ I.  This keeps the initial
        circuit close to the data‑encoding unitary, which may preserve
        trainability for small circuits.

    **identity_block**:
        Layers are paired.  For each pair (2k, 2k+1):
            layer 2k   gets  θ ~ Normal(0, σ²)
            layer 2k+1 gets −θ  (element‑wise negation)

        When RY(θ)RZ(φ) is followed by RY(−θ)RZ(−φ) with CNOT layers
        between them, the net rotation on each qubit is approximately
        zero for small θ, φ.  CNOT·CNOT = I on the same qubit pairs,
        so the entanglement also partially cancels.

        For small σ the total unitary is 𝟙 + O(σ²).  This is an
        *approximation* to exact identity‑block initialisation; it
        does not achieve exact identity because RY and RZ do not
        commute with CNOT when acting on different qubits.

        If n_layers is odd, the last unpaired layer uses small‑angle
        initialisation.

    Raises
    ------
    ValueError
        If *strategy* is not recognised.
    """
    if strategy not in INIT_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {INIT_STRATEGIES}"
        )

    rng = np.random.default_rng(seed)

    if strategy == "random":
        return rng.uniform(0, 2 * np.pi, size=shape)

    if strategy == "small_angle":
        return rng.normal(0, sigma, size=shape)

    # identity_block
    if len(shape) < 1:
        raise ValueError("identity_block requires shape with ≥1 dimension for layers")

    n_layers = shape[0]
    params = np.zeros(shape, dtype=float)
    remaining_shape = shape[1:]

    for k in range(0, n_layers - 1, 2):
        block = rng.normal(0, sigma, size=remaining_shape)
        params[k] = block
        params[k + 1] = -block  # approximate inverse

    # If odd number of layers, last layer gets small‑angle init
    if n_layers % 2 == 1:
        params[n_layers - 1] = rng.normal(0, sigma, size=remaining_shape)

    return params
