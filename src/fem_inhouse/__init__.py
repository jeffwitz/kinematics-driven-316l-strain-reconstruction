"""Kinematics-driven 316L strain-reconstruction tools."""

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig
from fem_inhouse.core.tensor_reconstruction import FullTensorState
from fem_inhouse.material import LudwikLaw, abaqus_plastic_table
from fem_inhouse.results import FEMResult, FrameResult, SolverDiagnostics
from fem_inhouse.solver import linear_solver_backend, require_pypardiso, run_case_study

__all__ = [
    "CaseStudyConfig",
    "FEMResult",
    "FrameResult",
    "FullTensorState",
    "LudwikLaw",
    "MaterialConfig",
    "MeshConfig",
    "SolverConfig",
    "SolverDiagnostics",
    "abaqus_plastic_table",
    "linear_solver_backend",
    "require_pypardiso",
    "run_case_study",
]

__version__ = "0.1.0"
