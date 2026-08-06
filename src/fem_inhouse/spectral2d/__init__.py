"""Spectral two-dimensional plane-stress mechanics primitives.

The package is deliberately layered: grid and boundary data do not depend on
the constitutive backend, while the material protocol is imported from the
solver-neutral core contract.
"""

# pyFFTW bundles its own FFTW runtime.  Import it before SciPy/BLAS-backed
# modules are loaded when it is available; otherwise its first plan can fail
# with a planner-NULL error after another native numerical library has already
# initialized the process.  The dependency remains optional.
try:
    import importlib

    importlib.import_module("pyfftw")
except ImportError:
    pass

from fem_inhouse.spectral2d.anderson import (
    AndersonDiagnostics,
    DisplacementAndersonAccelerator,
)
from fem_inhouse.spectral2d.boundary import (
    AppliedDisplacementExtension2D,
    HarmonicDirichletExtension2D,
    TransfiniteBoundaryInterpolation2D,
)
from fem_inhouse.spectral2d.config import Spectral2DConfig
from fem_inhouse.spectral2d.diagnostics import (
    JacobianActionDiagnostics,
    LinearSolveDiagnostics,
    PreconditionerActionDiagnostics,
    Spectral2DDiagnostics,
)
from fem_inhouse.spectral2d.ebi import (
    EBIPlaneStressElementBatch,
    EBIPlaneStressTrial,
    hookean_plane_stress_relative_error,
)
from fem_inhouse.spectral2d.green import (
    B0Green2D,
    GreenDiagnostics,
    ReferenceOperatorSymbols,
    TwoMuGreen2D,
    project_isotropic_plane_stress_tangent,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import (
    CellCenteredOnePoint2D,
    DiscreteKinematics2D,
    EBITwoTriangleKinematics2D,
    TwoSubcellDiagnostic2D,
)
from fem_inhouse.spectral2d.newton_ebi import (
    EBISpectralSolverConfig,
    pack_interior,
    pack_interior_into,
    solve_ebi_dirichlet_plane_stress,
    unpack_interior,
    unpack_interior_into,
)
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateJacobianWorkspace,
    apply_tangent_into,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.nonlinear import (
    SpectralIncrementConvergenceError,
    solve_dirichlet_plane_stress_spectral,
)
from fem_inhouse.spectral2d.result import Spectral2DResult
from fem_inhouse.spectral2d.step_control import (
    AdaptiveLoadPath,
    AdaptiveLoadStepController,
    AdaptiveStepConfig,
    LoadPathStep,
    LoadStepDecision,
    LoadStepObservation,
)
from fem_inhouse.spectral2d.step_doubling import (
    LoadStepAttempt,
    ObservableError,
    StepDoublingErrorConfig,
    StepDoublingFailureError,
    StepDoublingResult,
    StepErrorEstimate,
    StepObservables,
    estimate_step_error,
    estimate_step_error_by_doubling,
    next_step_factor,
    step_error_to_record,
)
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import (
    BufferedTransformPlan2D,
    FFTWPlannerEffort,
    FullDirichletDSTIPlan2D,
    SpectralTransformConfig,
    TransformBackend,
    TransformDiagnostics,
    TransformPlan2D,
)

__all__ = [
    "AdaptiveLoadPath",
    "AdaptiveLoadStepController",
    "AdaptiveStepConfig",
    "AndersonDiagnostics",
    "AppliedDisplacementExtension2D",
    "B0Green2D",
    "BufferedTransformPlan2D",
    "CellCenteredOnePoint2D",
    "DiscreteKinematics2D",
    "DisplacementAndersonAccelerator",
    "EBIPlaneStressElementBatch",
    "EBIPlaneStressTrial",
    "EBISpectralSolverConfig",
    "EBITwoTriangleKinematics2D",
    "FFTWPlannerEffort",
    "FullDirichletDSTIPlan2D",
    "GreenDiagnostics",
    "HarmonicDirichletExtension2D",
    "JacobianActionDiagnostics",
    "LinearSolveDiagnostics",
    "LoadPathStep",
    "LoadStepAttempt",
    "LoadStepDecision",
    "LoadStepObservation",
    "ObservableError",
    "PreconditionerActionDiagnostics",
    "ReferenceOperatorSymbols",
    "Spectral2DConfig",
    "Spectral2DDiagnostics",
    "Spectral2DResult",
    "SpectralIncrementConvergenceError",
    "SpectralTransformConfig",
    "StepDoublingErrorConfig",
    "StepDoublingFailureError",
    "StepDoublingResult",
    "StepErrorEstimate",
    "StepObservables",
    "StructuredGrid2D",
    "TransfiniteBoundaryInterpolation2D",
    "TransformBackend",
    "TransformDiagnostics",
    "TransformPlan2D",
    "TwoMuGreen2D",
    "TwoStateJacobianWorkspace",
    "TwoSubcellDiagnostic2D",
    "apply_tangent_into",
    "create_full_dirichlet_dsti_plan",
    "estimate_step_error",
    "estimate_step_error_by_doubling",
    "hookean_plane_stress_relative_error",
    "next_step_factor",
    "pack_interior",
    "pack_interior_into",
    "project_isotropic_plane_stress_tangent",
    "solve_dirichlet_plane_stress_spectral",
    "solve_ebi_dirichlet_plane_stress",
    "solve_two_state_dirichlet_plane_stress",
    "step_error_to_record",
    "unpack_interior",
    "unpack_interior_into",
]
