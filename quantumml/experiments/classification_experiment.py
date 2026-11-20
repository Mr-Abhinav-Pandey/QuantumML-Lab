"""Experiment D — Binary classification with VQC and classical baseline.

Trains a VQC on the two‑moons dataset, compares against a classical
MLP, and records comprehensive metrics across multiple seeds.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from quantumml.datasets.loader import load_dataset
from quantumml.circuits.ansatz import VariationalAnsatz
from quantumml.training.initialization import initialize_parameters
from quantumml.training.trainer import VQCTrainer
from quantumml.training.optimizers import GradientDescent, Adam
from quantumml.training.metrics import compute_metrics
from quantumml.models.classical import train_classical_baseline
from quantumml.experiments.runner import ExperimentRunner


def run_classification_experiment(
    n_qubits: int = 2,
    n_layers: int = 2,
    entanglement: str = "linear",
    init_strategy: str = "small_angle",
    optimizer_name: str = "gd",
    learning_rate: float = 0.1,
    epochs: int = 35,
    dataset_name: str = "Two moons",
    samples: int = 180,
    noise: float = 0.12,
    seeds: list[int] = [21],
    classical_model: str = "mlp",
    progress_callback: Optional[Callable] = None,
    save: bool = True,
) -> dict:
    """Run Experiment D — binary classification comparison.

    Parameters
    ----------
    seeds : list of int
        Seeds for independent VQC training runs (use multiple for
        variation analysis).
    progress_callback : callable, optional
        Called as ``callback(step, total, msg)``.

    Returns
    -------
    dict with per‑seed quantum results, classical baseline, and summary.
    """
    config = {
        "experiment": "D_classification",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "entanglement": entanglement,
        "init_strategy": init_strategy,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "dataset": dataset_name,
        "samples": samples,
        "noise": noise,
        "seeds": seeds,
        "classical_model": classical_model,
    }

    runner = ExperimentRunner("D_classification", config, "training_results")
    total_steps = len(seeds) + 1  # VQC seeds + classical
    step = 0

    with runner:
        # ── load data once ────────────────────────────────────────────
        X_train, X_test, y_train, y_test = load_dataset(
            dataset_name, samples, noise, seed=7
        )

        # ── classical baseline ────────────────────────────────────────
        classical_result = train_classical_baseline(
            X_train, y_train, X_test, y_test,
            model_type=classical_model, seed=42,
        )
        step += 1
        if progress_callback:
            progress_callback(step, total_steps, "Classical baseline trained")

        # ── VQC runs ──────────────────────────────────────────────────
        quantum_runs = []
        for run_seed in seeds:
            ansatz = VariationalAnsatz(n_qubits, n_layers, entanglement)
            params = initialize_parameters(
                ansatz.parameter_shape, init_strategy, seed=run_seed
            )

            if optimizer_name == "adam":
                opt = Adam(lr=learning_rate)
            else:
                opt = GradientDescent(lr=learning_rate)

            trainer = VQCTrainer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                entanglement=entanglement,
                optimizer=opt,
            )

            train_result = trainer.train(
                X_train, y_train, params, epochs=epochs,
            )

            final_params = train_result["final_params"]
            test_metrics = trainer.evaluate(X_test, y_test, final_params)
            train_metrics = trainer.evaluate(X_train, y_train, final_params)
            proba = trainer.predict_proba(X_test, final_params)

            quantum_runs.append({
                "seed": run_seed,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "history": train_result["history"],
                "training_time": train_result["training_time"],
                "final_params": final_params.tolist(),
                "test_probs": proba.tolist(),
            })

            step += 1
            if progress_callback:
                progress_callback(
                    step, total_steps,
                    f"VQC seed={run_seed} acc={test_metrics['accuracy']:.3f}"
                )

        # ── summarise across seeds ────────────────────────────────────
        test_accs = [r["test_metrics"]["accuracy"] for r in quantum_runs]
        summary = {
            "quantum_mean_accuracy": float(np.mean(test_accs)),
            "quantum_std_accuracy": float(np.std(test_accs)),
            "quantum_min_accuracy": float(np.min(test_accs)),
            "quantum_max_accuracy": float(np.max(test_accs)),
            "classical_accuracy": classical_result["test_metrics"]["accuracy"],
            "n_runs": len(seeds),
        }

    output = {
        "quantum_runs": quantum_runs,
        "classical": {
            "model_type": classical_result["model_type"],
            "train_metrics": classical_result["train_metrics"],
            "test_metrics": classical_result["test_metrics"],
            "training_time": classical_result["training_time"],
        },
        "summary": summary,
        "data_info": {
            "dataset": dataset_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
    }

    if save:
        runner.save_results(output)

    output["experiment_id"] = runner.experiment_id
    return output
