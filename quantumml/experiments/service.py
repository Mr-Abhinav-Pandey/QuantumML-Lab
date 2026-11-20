"""Streamlit-independent facade for scientific experiment execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quantumml.experiments.classification_experiment import (
    run_classification_experiment,
)
from quantumml.experiments.gradient_experiment import (
    run_depth_experiment,
    run_entanglement_experiment,
    run_initialization_experiment,
)


class ExperimentService:
    """Expose experiment execution without depending on a UI framework."""

    _RUNNERS = {
        "depth": run_depth_experiment,
        "entanglement": run_entanglement_experiment,
        "initialization": run_initialization_experiment,
        "classification": run_classification_experiment,
    }

    @classmethod
    def run(
        cls,
        experiment: str,
        *,
        progress_callback: Callable[..., Any] | None = None,
        **configuration: Any,
    ) -> dict:
        """Run a named experiment using a UI-agnostic configuration."""
        try:
            runner = cls._RUNNERS[experiment]
        except KeyError as exc:
            valid = ", ".join(sorted(cls._RUNNERS))
            raise ValueError(
                f"Unknown experiment '{experiment}'. Choose from {valid}"
            ) from exc

        return runner(progress_callback=progress_callback, **configuration)
