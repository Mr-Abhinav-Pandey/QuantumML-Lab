"""Gradient‑analysis experiments (A, B, C).

Experiment A — Circuit depth and gradient concentration
Experiment B — Entanglement architecture comparison
Experiment C — Parameter initialization comparison

Each function returns a results dict **and** saves it to disk via
``ExperimentRunner``.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from quantumml.simulator.statevector import StatevectorSimulator
from quantumml.circuits.encoding import angle_encoding
from quantumml.circuits.ansatz import VariationalAnsatz
from quantumml.gradients.parameter_shift import parameter_shift_gradient
from quantumml.training.initialization import initialize_parameters
from quantumml.experiments.analysis import GradientAnalysis
from quantumml.experiments.runner import ExperimentRunner
from quantumml.utils import stable_seed


# ── helper: single‑seed gradient collection ──────────────────────────

def _collect_gradients_single_seed(
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    init_strategy: str,
    seed: int,
    x_sample: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the cost‑function gradient for one random initialisation.

    Cost function: C(θ) = ⟨Z₀⟩  (local observable on qubit 0).
    Data point: either *x_sample* or a fixed reference input.

    Returns flat gradient array of shape (num_parameters,).
    """
    ansatz = VariationalAnsatz(n_qubits, n_layers, entanglement)
    params = initialize_parameters(ansatz.parameter_shape, init_strategy, seed=seed)

    if x_sample is None:
        x_sample = np.full(min(2, n_qubits), 0.5)

    def cost_fn(p: np.ndarray) -> float:
        sim = StatevectorSimulator(n_qubits)
        angle_encoding(sim, x_sample)
        ansatz.apply(sim, p)
        return sim.expectation_value_z(0)

    grad = parameter_shift_gradient(cost_fn, params)
    return grad.ravel()


# ── Experiment A ─────────────────────────────────────────────────────

def run_depth_experiment(
    qubit_counts: list[int] = [2, 4, 6, 8],
    depth_counts: list[int] = [1, 2, 4, 6, 8],
    n_seeds: int = 10,
    entanglement: str = "linear",
    init_strategy: str = "random",
    base_seed: int = 0,
    progress_callback: Optional[Callable] = None,
    save: bool = True,
) -> dict:
    """Experiment A — gradient statistics vs circuit depth and qubit count.

    Parameters
    ----------
    qubit_counts : list of int
    depth_counts : list of int
    n_seeds : int
        Independent random initialisations per configuration.
    entanglement, init_strategy : str
    base_seed : int
    progress_callback : callable, optional
        Called as ``callback(step, total, msg)`` for progress updates.
    save : bool
        Whether to persist results to disk.

    Returns
    -------
    dict with keys:
        configs — list of (n_qubits, depth) tuples
        results — dict mapping (n_qubits, depth) → GradientAnalysis.summary()
        raw_gradients — dict mapping (n_qubits, depth) → (n_seeds, n_params) array
    """
    config = {
        "experiment": "A_depth_gradient",
        "qubit_counts": qubit_counts,
        "depth_counts": depth_counts,
        "n_seeds": n_seeds,
        "entanglement": entanglement,
        "init_strategy": init_strategy,
        "base_seed": base_seed,
    }

    runner = ExperimentRunner("A_depth_gradient", config, "gradient_results")
    total_steps = len(qubit_counts) * len(depth_counts) * n_seeds
    step = 0

    all_results = {}
    all_raw = {}

    with runner:
        for nq in qubit_counts:
            for depth in depth_counts:
                grads = []
                for s in range(n_seeds):
                    seed = base_seed + s * 1000 + nq * 100 + depth
                    g = _collect_gradients_single_seed(
                        nq, depth, entanglement, init_strategy, seed
                    )
                    grads.append(g)
                    step += 1
                    if progress_callback:
                        progress_callback(
                            step, total_steps,
                            f"qubits={nq} depth={depth} seed={s+1}/{n_seeds}"
                        )

                grads_arr = np.array(grads)
                analysis = GradientAnalysis(grads_arr)
                key = f"{nq}q_{depth}L"
                all_results[key] = analysis.summary()
                all_raw[key] = grads_arr.tolist()

    output = {
        "configs": [(nq, d) for nq in qubit_counts for d in depth_counts],
        "results": all_results,
        "raw_gradients": all_raw,
    }

    if save:
        runner.save_results(output)

    output["experiment_id"] = runner.experiment_id
    return output


# ── Experiment B ─────────────────────────────────────────────────────

def run_entanglement_experiment(
    n_qubits: int = 4,
    n_layers: int = 4,
    entanglements: list[str] = ["linear", "circular", "all_to_all"],
    n_seeds: int = 10,
    init_strategy: str = "random",
    base_seed: int = 0,
    progress_callback: Optional[Callable] = None,
    save: bool = True,
) -> dict:
    """Experiment B — gradient statistics vs entanglement architecture.

    All other variables (qubits, depth, initialisation) are fixed.
    """
    config = {
        "experiment": "B_entanglement",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "entanglements": entanglements,
        "n_seeds": n_seeds,
        "init_strategy": init_strategy,
        "base_seed": base_seed,
    }

    runner = ExperimentRunner("B_entanglement", config, "gradient_results")
    total_steps = len(entanglements) * n_seeds
    step = 0

    all_results = {}
    all_raw = {}

    with runner:
        for ent in entanglements:
            grads = []
            for s in range(n_seeds):
                seed = stable_seed("B_entanglement", base_seed, ent, s)
                g = _collect_gradients_single_seed(
                    n_qubits, n_layers, ent, init_strategy, seed
                )
                grads.append(g)
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps, f"{ent} seed={s+1}/{n_seeds}")

            grads_arr = np.array(grads)
            analysis = GradientAnalysis(grads_arr)
            all_results[ent] = analysis.summary()
            all_raw[ent] = grads_arr.tolist()

    output = {
        "entanglements": entanglements,
        "results": all_results,
        "raw_gradients": all_raw,
    }

    if save:
        runner.save_results(output)

    output["experiment_id"] = runner.experiment_id
    return output


# ── Experiment C ─────────────────────────────────────────────────────

def run_initialization_experiment(
    n_qubits: int = 4,
    n_layers: int = 4,
    strategies: list[str] = ["random", "small_angle", "identity_block"],
    entanglement: str = "linear",
    n_seeds: int = 10,
    base_seed: int = 0,
    progress_callback: Optional[Callable] = None,
    save: bool = True,
) -> dict:
    """Experiment C — gradient statistics vs initialisation strategy.

    All other variables (qubits, depth, entanglement) are fixed.
    """
    config = {
        "experiment": "C_initialization",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "strategies": strategies,
        "entanglement": entanglement,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
    }

    runner = ExperimentRunner("C_initialization", config, "gradient_results")
    total_steps = len(strategies) * n_seeds
    step = 0

    all_results = {}
    all_raw = {}

    with runner:
        for strat in strategies:
            grads = []
            for s in range(n_seeds):
                seed = base_seed + s * 1000
                g = _collect_gradients_single_seed(
                    n_qubits, n_layers, entanglement, strat, seed
                )
                grads.append(g)
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps, f"{strat} seed={s+1}/{n_seeds}")

            grads_arr = np.array(grads)
            analysis = GradientAnalysis(grads_arr)
            all_results[strat] = analysis.summary()
            all_raw[strat] = grads_arr.tolist()

    output = {
        "strategies": strategies,
        "results": all_results,
        "raw_gradients": all_raw,
    }

    if save:
        runner.save_results(output)

    output["experiment_id"] = runner.experiment_id
    return output
