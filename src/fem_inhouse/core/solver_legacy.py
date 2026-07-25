"""Compatibility imports for the historical numerical-kernel module."""

from fem_inhouse.core.nonlinear import _verify, run_fem
from fem_inhouse.solver import linear_solver_backend

_SOLVER_NAME = linear_solver_backend()

__all__ = ["_SOLVER_NAME", "_verify", "run_fem"]
