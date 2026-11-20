"""Gradient computation modules."""

from .parameter_shift import parameter_shift_gradient
from .finite_difference import finite_difference_gradient

__all__ = ["parameter_shift_gradient", "finite_difference_gradient"]
