"""Spectral two-dimensional plane-stress mechanics primitives.

The package is deliberately layered: grid and boundary data do not depend on
the constitutive backend, while the material protocol is imported from the
solver-neutral core contract.
"""

from fem_inhouse.spectral2d.anderson import AndersonAccelerator, AndersonDiagnostics
from fem_inhouse.spectral2d.boundary import (
    AppliedDisplacementExtension2D,
    HarmonicDirichletExtension2D,
    TransfiniteBoundaryInterpolation2D,
)
from fem_inhouse.spectral2d.config import Spectral2DConfig
from fem_inhouse.spectral2d.diagnostics import Spectral2DDiagnostics
from fem_inhouse.spectral2d.green import (
    B0Green2D,
    GreenDiagnostics,
    ReferenceOperatorSymbols,
    TwoMuGreen2D,
    project_isotropic_plane_stress_tangent,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import QUAD1_2D, TRI2_2D, DiscreteKinematics2D
from fem_inhouse.spectral2d.nonlinear import (
    SpectralIncrementConvergenceError,
    solve_dirichlet_plane_stress_spectral,
)
from fem_inhouse.spectral2d.result import Spectral2DResult
from fem_inhouse.spectral2d.transforms import FullDirichletDSTIPlan2D, TransformPlan2D

__all__ = [
    "QUAD1_2D",
    "TRI2_2D",
    "AndersonAccelerator",
    "AndersonDiagnostics",
    "AppliedDisplacementExtension2D",
    "B0Green2D",
    "DiscreteKinematics2D",
    "FullDirichletDSTIPlan2D",
    "GreenDiagnostics",
    "HarmonicDirichletExtension2D",
    "ReferenceOperatorSymbols",
    "Spectral2DConfig",
    "Spectral2DDiagnostics",
    "Spectral2DResult",
    "SpectralIncrementConvergenceError",
    "StructuredGrid2D",
    "TransfiniteBoundaryInterpolation2D",
    "TransformPlan2D",
    "TwoMuGreen2D",
    "project_isotropic_plane_stress_tangent",
    "solve_dirichlet_plane_stress_spectral",
]
