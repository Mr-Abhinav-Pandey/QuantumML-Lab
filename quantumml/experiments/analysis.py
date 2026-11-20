"""Statistical analysis of gradient distributions.

The ``GradientAnalysis`` class consumes raw gradient arrays collected
from multiple random initialisations and computes summary statistics
used to investigate barren‑plateau behaviour.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class GradientAnalysis:
    """Analyse gradient statistics from repeated experiments.

    Parameters
    ----------
    gradients : ndarray of shape (n_seeds, n_params)
        Each row is the full gradient vector from one random
        initialisation.
    small_gradient_threshold : float
        Magnitude below which a gradient component is classified as
        "near‑zero".  This is an *operational definition*, not a
        universal physical boundary.
    """

    def __init__(
        self,
        gradients: np.ndarray,
        small_gradient_threshold: float = 1e-3,
    ) -> None:
        self.gradients = np.asarray(gradients, dtype=float)
        if self.gradients.ndim == 1:
            self.gradients = self.gradients.reshape(1, -1)
        self.threshold = small_gradient_threshold

    @property
    def n_seeds(self) -> int:
        return self.gradients.shape[0]

    @property
    def n_params(self) -> int:
        return self.gradients.shape[1]

    # ── per‑parameter statistics (across seeds) ──────────────────────

    def gradient_mean(self) -> np.ndarray:
        """Mean gradient per parameter (across seeds)."""
        return np.mean(self.gradients, axis=0)

    def gradient_variance(self) -> np.ndarray:
        """Variance per parameter (across seeds)."""
        return np.var(self.gradients, axis=0)

    def gradient_std(self) -> np.ndarray:
        return np.std(self.gradients, axis=0)

    def mean_absolute_gradient(self) -> np.ndarray:
        return np.mean(np.abs(self.gradients), axis=0)

    def median_absolute_gradient(self) -> np.ndarray:
        return np.median(np.abs(self.gradients), axis=0)

    # ── aggregate statistics (single scalars) ────────────────────────

    def overall_mean_variance(self) -> float:
        """Average gradient variance across all parameters."""
        return float(np.mean(self.gradient_variance()))

    def overall_mean_abs_gradient(self) -> float:
        return float(np.mean(np.abs(self.gradients)))

    def fraction_near_zero(self) -> float:
        """Fraction of all gradient values below the threshold.

        This metric quantifies how many gradient components are
        "practically vanishing" under the operational threshold.
        """
        return float(np.mean(np.abs(self.gradients) < self.threshold))

    # ── summary dict ─────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a JSON‑serialisable summary of gradient statistics."""
        all_flat = self.gradients.ravel()
        return {
            "n_seeds": self.n_seeds,
            "n_params": self.n_params,
            "gradient_mean": float(np.mean(all_flat)),
            "gradient_std": float(np.std(all_flat)),
            "mean_abs_gradient": float(np.mean(np.abs(all_flat))),
            "median_abs_gradient": float(np.median(np.abs(all_flat))),
            "gradient_variance_per_param_mean": self.overall_mean_variance(),
            "gradient_variance_per_param_values": self.gradient_variance().tolist(),
            "fraction_near_zero": self.fraction_near_zero(),
            "small_gradient_threshold": self.threshold,
            "min_abs_gradient": float(np.min(np.abs(all_flat))),
            "max_abs_gradient": float(np.max(np.abs(all_flat))),
            "iqr": float(np.subtract(*np.percentile(all_flat, [75, 25]))),
        }


def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute mean and symmetric confidence interval.

    Uses the t‑distribution for small samples.

    Returns
    -------
    (mean, lower, upper)
    """
    from scipy import stats

    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se
    return mean, mean - margin, mean + margin
