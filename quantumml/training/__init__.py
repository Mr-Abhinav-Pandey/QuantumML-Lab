"""Training components for variational quantum classifiers."""

from .initialization import initialize_parameters, INIT_STRATEGIES
from .optimizers import GradientDescent, Adam
from .metrics import compute_metrics, binary_cross_entropy
from .trainer import VQCTrainer

__all__ = [
    "initialize_parameters",
    "INIT_STRATEGIES",
    "GradientDescent",
    "Adam",
    "compute_metrics",
    "binary_cross_entropy",
    "VQCTrainer",
]
