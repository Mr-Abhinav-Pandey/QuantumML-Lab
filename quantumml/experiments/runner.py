"""Experiment runner — base infrastructure for reproducible experiments.

Every experiment generates a metadata record that is saved to
``research/original_experiments/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from quantumml import __version__
from quantumml.utils import (
    generate_experiment_id,
    utc_now_iso,
    save_json,
    original_experiments_dir,
)


class ExperimentRunner:
    """Context‑manager that tracks timing and saves metadata."""

    def __init__(
        self,
        experiment_name: str,
        config: dict,
        sub_dir: str = "gradient_results",
    ) -> None:
        self.experiment_name = experiment_name
        self.config = config
        self.sub_dir = sub_dir
        self.experiment_id = generate_experiment_id()
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self._t0: float = 0.0

    def __enter__(self):
        self.started_at = utc_now_iso()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished_at = utc_now_iso()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def save_results(self, results: dict) -> Path:
        """Persist the experiment record to disk.

        Returns the path of the saved JSON file.
        """
        record = {
            "experiment_id": self.experiment_id,
            "source_label": "ORIGINAL_EXPERIMENT",
            "experiment_name": self.experiment_name,
            "software_version": __version__,
            "started_at_utc": self.started_at,
            "finished_at_utc": self.finished_at or utc_now_iso(),
            "execution_time_seconds": round(self.elapsed, 3),
            "configuration": self.config,
            "results": _make_serializable(results),
            "scientific_scope": (
                "Project‑generated simulation result.  Not a published "
                "result and not a physical quantum hardware measurement."
            ),
        }
        out_dir = original_experiments_dir() / self.sub_dir
        path = out_dir / f"{self.experiment_id}.json"
        save_json(record, path)
        return path


def _make_serializable(obj):
    """Recursively convert numpy types to Python built‑ins."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
