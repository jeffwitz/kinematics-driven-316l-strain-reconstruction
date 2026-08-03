"""Typed result containers for finite-element case-study solves."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.tensor_reconstruction import (
    FullTensorState,
    reconstruct_python_plane_stress_state,
)

FloatArray = NDArray


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Fields recorded at one pseudo-time."""

    stress_mpa: FloatArray
    total_strain: FloatArray
    equivalent_plastic_strain: FloatArray
    displacement_mm: FloatArray


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    """Convergence and timing data for one completed nonlinear solve."""

    backend: str
    elapsed_seconds: float
    initialization_seconds: float
    elastic_assembly_seconds: float
    constitutive_seconds: float
    tangent_assembly_seconds: float
    linear_solve_seconds: float
    output_seconds: float
    attempted_increments: int
    converged_increments: int
    cutbacks: int
    total_newton_iterations: int
    maximum_newton_iterations: int
    final_residual_norm: float
    final_relative_residual: float
    final_convergence_criterion: str
    newton_line_search_enabled: bool = False
    line_search_evaluations: int = 0
    line_search_reductions: int = 0
    line_search_failures: int = 0
    minimum_accepted_line_search_factor: float = 1.0
    boundary_history_predictor: str = "elastic"
    secant_predictor_uses: int = 0
    secant_predictor_fallbacks: int = 0
    tensor_reconstruction_source: str = "unspecified"
    #: Element formulation actually used, and what it cost in material points.
    element_formulation: str = "cps4"
    gauss_points_per_element: int = 4
    constitutive_material_point_count: int = 0
    #: Work done against the hourglass stabilisation. NUMERICAL, not physical:
    #: it exists only because one integration point cannot see the hourglass
    #: modes, and it is reported apart from the constitutive energy for that
    #: reason. Zero for the fully integrated element.
    hourglass_energy: float = 0.0
    #: Mechanical internal work integrated over accepted increments.
    internal_work: float = 0.0
    #: Hourglass energy over the internal work actually done. Below 1 percent
    #: the influence is small; between 1 and 5 the result deserves a look; above
    #: 5 it may be contaminated. These are analysis rules, not universal truths,
    #: and a small ratio hiding a concentration inside a plasticity band is the
    #: case they do not cover -- read the per-element field for that.
    hourglass_energy_ratio: float = 0.0
    linear_system_matrix_type: str = "unspecified"
    maximum_relative_constitutive_tangent_asymmetry: float = 0.0
    maximum_gauss_point_plane_stress_residual_mpa: float = 0.0
    maximum_local_plane_stress_iterations: int = 0
    mean_local_plane_stress_iterations: float = 0.0
    local_plane_stress_failures: int = 0
    maximum_cbb_condition_number: float = 0.0
    nonlocal_plasticity_enabled: bool = False
    nonlocal_convergence_norm: str = "not_applicable"
    nonlocal_length_scale_mm: float = 0.0
    nonlocal_coupling_modulus_mpa: float = 0.0
    nonlocal_relaxation: float = 0.0
    nonlocal_relaxation_strategy: str = "fixed"
    nonlocal_minimum_relaxation: float = 0.0
    nonlocal_maximum_relaxation: float = 0.0
    nonlocal_aitken_residual_growth_factor: float = 0.0
    nonlocal_fixed_point_history: tuple[dict[str, object], ...] = ()
    nonlocal_iterations_per_newton: tuple[int, ...] = ()
    nonlocal_iterations_per_increment: tuple[int, ...] = ()
    total_nonlocal_iterations: int = 0
    maximum_nonlocal_iterations: int = 0
    mean_nonlocal_iterations: float = 0.0
    final_nonlocal_relative_residual: float = 0.0
    maximum_helmholtz_residual_relative: float = 0.0
    maximum_absolute_nonlocal_mean_drift: float = 0.0
    helmholtz_seconds: float = 0.0
    nonlocal_mfront_seconds: float = 0.0
    nonlocal_coupling_failures: int = 0
    mfront_integration_without_tangent_seconds: float = 0.0
    mfront_integration_with_tangent_seconds: float = 0.0
    kelvin_conversion_seconds: float = 0.0
    tensor_reconstruction_seconds: float = 0.0
    internal_force_seconds: float = 0.0
    element_matrix_seconds: float = 0.0
    sparse_assembly_seconds: float = 0.0
    free_system_extraction_seconds: float = 0.0
    pardiso_seconds: float = 0.0
    pardiso_analysis_seconds: float = 0.0
    pardiso_factorization_seconds: float = 0.0
    pardiso_solve_seconds: float = 0.0
    pardiso_analysis_calls: int = 0
    pardiso_factorization_calls: int = 0
    pardiso_solve_calls: int = 0
    nonlocal_mfront_without_tangent_seconds: float = 0.0
    nonlocal_mfront_with_tangent_seconds: float = 0.0
    mfront_integration_without_tangent_calls: int = 0
    mfront_integration_with_tangent_calls: int = 0
    tensor_reconstruction_calls: int = 0


@dataclass(frozen=True, slots=True)
class FEMResult:
    """Final fields returned by the supported structured CPS4 solve."""

    displacement_mm: FloatArray
    stress_mpa: FloatArray
    total_strain: FloatArray
    plastic_strain: FloatArray
    equivalent_plastic_strain: FloatArray
    reaction_force: FloatArray
    stress_tensor_mpa: FloatArray | None = None
    total_strain_tensor: FloatArray | None = None
    elastic_strain_tensor: FloatArray | None = None
    plastic_strain_tensor: FloatArray | None = None
    plane_stress_residual_mpa: FloatArray | None = None
    plane_stress_residual_vector_mpa: FloatArray | None = None
    nonlocal_equivalent_plastic_strain: FloatArray | None = None
    equivalent_plastic_strain_mismatch: FloatArray | None = None
    nonlocal_hardening_mpa: FloatArray | None = None
    yield_surface_radius_mpa: FloatArray | None = None
    nonlocal_residual: FloatArray | None = None
    boundary_misfit_mm: FloatArray | None = None
    #: Numerical hourglass energy at element level. Present for CPS4R runs,
    #: absent for the fully integrated reference formulation.
    hourglass_energy_by_element: FloatArray | None = None
    frames: dict[float, FrameResult] = field(default_factory=dict)
    diagnostics: SolverDiagnostics | None = None

    def arrays(self) -> tuple[FloatArray, ...]:
        """Return every final-state array for common validation operations."""

        historical = (
            self.displacement_mm,
            self.stress_mpa,
            self.total_strain,
            self.plastic_strain,
            self.equivalent_plastic_strain,
            self.reaction_force,
        )
        reconstructed = (
            self.stress_tensor_mpa,
            self.total_strain_tensor,
            self.elastic_strain_tensor,
            self.plastic_strain_tensor,
            self.plane_stress_residual_mpa,
            self.plane_stress_residual_vector_mpa,
            self.nonlocal_equivalent_plastic_strain,
            self.equivalent_plastic_strain_mismatch,
            self.nonlocal_hardening_mpa,
            self.yield_surface_radius_mpa,
            self.nonlocal_residual,
            self.hourglass_energy_by_element,
        )
        return historical + tuple(field for field in reconstructed if field is not None)


def load_full_tensor_state(
    directory: str | Path,
    *,
    poisson_ratio: float | None = None,
    completion_strategy: str | None = None,
) -> FullTensorState:
    """Load new tensor files or reconstruct a legacy ``S/E/PE`` result set."""

    root = Path(directory)
    tensor_files = {
        "stress_tensor_mpa": root / "S_3D.npy",
        "total_strain_tensor": root / "E_3D.npy",
        "elastic_strain_tensor": root / "EE_3D.npy",
        "plastic_strain_tensor": root / "PE_3D.npy",
        "plane_stress_residual_mpa": root / "S33_RESIDUAL_MPA.npy",
        "plane_stress_residual_vector_mpa": root / "PLANE_STRESS_RESIDUAL_MPA.npy",
    }
    present = {name: path.is_file() for name, path in tensor_files.items()}
    legacy_vector_missing = not present["plane_stress_residual_vector_mpa"] and all(
        available
        for name, available in present.items()
        if name != "plane_stress_residual_vector_mpa"
    )
    if any(present.values()) and not all(present.values()) and not legacy_vector_missing:
        missing = sorted(name for name, available in present.items() if not available)
        raise RuntimeError(f"saved result contains an incomplete full tensor state: {missing}")
    if all(present.values()) or legacy_vector_missing:
        arrays = {name: np.load(path) for name, path in tensor_files.items() if path.is_file()}
        stress_tensor = arrays["stress_tensor_mpa"]
        residual = arrays["plane_stress_residual_mpa"]
        if stress_tensor.shape[-2:] != (3, 3):
            raise ValueError("S_3D.npy must have trailing dimensions (3, 3)")
        for name in (
            "total_strain_tensor",
            "elastic_strain_tensor",
            "plastic_strain_tensor",
        ):
            if arrays[name].shape != stress_tensor.shape:
                raise ValueError(f"{tensor_files[name].name} shape does not match S_3D.npy")
        if residual.shape != stress_tensor.shape[:-2]:
            raise ValueError("S33_RESIDUAL_MPA.npy shape does not match S_3D.npy")
        residual_vector = arrays.get("plane_stress_residual_vector_mpa")
        if residual_vector is None:
            residual_vector = np.zeros((*residual.shape, 3), dtype=float)
            residual_vector[..., 0] = residual
        if residual_vector.shape != (*residual.shape, 3):
            raise ValueError("PLANE_STRESS_RESIDUAL_MPA.npy shape is invalid")
        if not np.array_equal(residual, residual_vector[..., 0]):
            raise ValueError("S33_RESIDUAL_MPA.npy must equal PLANE_STRESS_RESIDUAL_MPA[..., 0]")
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("saved full tensor state contains non-finite values")
        if not np.array_equal(residual, stress_tensor[..., 2, 2]):
            raise ValueError("saved plane-stress residual does not equal S_3D[..., 2, 2]")
        return FullTensorState(
            stress_tensor_mpa=stress_tensor,
            total_strain_tensor=arrays["total_strain_tensor"],
            elastic_strain_tensor=arrays["elastic_strain_tensor"],
            plastic_strain_tensor=arrays["plastic_strain_tensor"],
            plane_stress_residual_mpa=residual,
            plane_stress_residual_vector_mpa=residual_vector,
        )

    if completion_strategy != "j2_isotropic_analytical":
        raise ValueError(
            "legacy S/E/PE reconstruction requires the explicit "
            "completion_strategy='j2_isotropic_analytical'"
        )
    if poisson_ratio is None:
        raise ValueError(
            "poisson_ratio is required to reconstruct full tensors from legacy S/E/PE files"
        )
    legacy_files = {name: root / f"{name}.npy" for name in ("S", "E", "PE")}
    missing_legacy = sorted(name for name, path in legacy_files.items() if not path.is_file())
    if missing_legacy:
        raise FileNotFoundError(f"legacy result fields are missing: {missing_legacy}")
    return reconstruct_python_plane_stress_state(
        np.load(legacy_files["E"]),
        np.load(legacy_files["PE"]),
        np.load(legacy_files["S"]),
        poisson_ratio,
    )
