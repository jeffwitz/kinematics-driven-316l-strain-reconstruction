#!/usr/bin/env python3
"""Compatibility entry point for the historical ``fem_pixel`` module."""

from fem_inhouse.core.constitutive import von_mises as _vm
from fem_inhouse.core.solver_legacy import run_fem

__all__ = ["_vm", "run_fem"]


if __name__ == "__main__":
    from fem_inhouse.core.solver_legacy import _verify

    _verify()
