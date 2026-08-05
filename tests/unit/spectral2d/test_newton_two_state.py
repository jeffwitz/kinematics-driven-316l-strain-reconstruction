import numpy as np
import pytest

from fem_inhouse.core.plane_stress_material import ConstitutiveTrial, InPlaneConstitutiveTrial
from fem_inhouse.spectral2d import (
    EBISpectralSolverConfig,
    EBITwoTriangleKinematics2D,
    StructuredGrid2D,
)
from fem_inhouse.spectral2d.newton_two_state import (
    TraditionalTwoStateTriangleBatch,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig
from scripts.qualify_ebi_state_sharing import moment, side_resultants


class NonlinearStateBatch:
    def __init__(self, point_count: int) -> None:
        self.point_count = point_count
        self.calls = 0
        self.commits = 0
        self.reverts = 0
        self._has_trial = False
        self.committed_stress = None
        self._stiffness = np.broadcast_to(
            np.array([[4.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 1.5]]),
            (point_count, 3, 3),
        ).copy()

    def evaluate_in_plane(self, strain, *, time_increment, consistent_tangent=True):
        self.calls += 1
        self._has_trial = True
        values = np.asarray(strain, dtype=float).reshape(-1, 3)
        stress = np.einsum("pij,pj->pi", self._stiffness, values) + 0.2 * values**2
        self._last_stress = stress.copy()
        tangent = self._stiffness + np.einsum("pi,ij->pij", 0.4 * values, np.eye(3))
        return InPlaneConstitutiveTrial(stress_in_plane_mpa=stress, tangent_in_plane_mpa=tangent)

    def complete_trial(self, trial):
        stress = np.asarray(trial.stress_in_plane_mpa)
        full_stress = np.zeros((self.point_count, 3, 3))
        full_stress[:, 0, 0] = stress[:, 0]
        full_stress[:, 1, 1] = stress[:, 1]
        full_stress[:, 0, 1] = full_stress[:, 1, 0] = stress[:, 2]
        strain = np.zeros_like(full_stress)
        return ConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=trial.tangent_in_plane_mpa,
            full_stress_tensor_mpa=full_stress,
            full_strain_tensor=strain,
            elastic_strain_tensor=strain,
            plastic_strain_tensor=strain,
            plane_stress_residual_mpa=np.zeros((self.point_count, 3)),
            observables={"accumulated_slip": np.zeros(self.point_count)},
        )

    def commit(self):
        if not self._has_trial:
            raise RuntimeError("commit requires an active trial")
        self.commits += 1
        self.committed_stress = self._last_stress.copy()
        self._has_trial = False

    def revert(self):
        self.reverts += 1
        self._has_trial = False


def test_two_state_nonlinear_directional_tangent() -> None:
    grid = StructuredGrid2D(2, 2, 1.0, 1.0)
    material = NonlinearStateBatch(8)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    kinematics = EBITwoTriangleKinematics2D(grid)
    displacement = np.zeros((*grid.node_shape, 2))
    displacement[1, 1] = (0.1, -0.06)
    direction = np.zeros_like(displacement)
    direction[1, 1] = (0.7, -0.4)
    trial = elements.evaluate_samples(kinematics.strain_samples(displacement), time_increment=0.1)
    analytical = elements.tangent_action(direction, kinematics=kinematics, trial=trial)
    step = 1.0e-7
    perturbed = elements.evaluate_samples(
        kinematics.strain_samples(displacement + step * direction), time_increment=0.1
    )
    base = kinematics.divergence_from_sample_stress(trial.sample_stress_mpa)
    updated = kinematics.divergence_from_sample_stress(perturbed.sample_stress_mpa)
    error = np.linalg.norm((updated - base) / step - analytical) / max(
        np.linalg.norm(analytical), 1.0
    )
    assert error < 1.0e-7
    assert material.calls == 2


def test_two_state_trial_revert_commit_is_transactional() -> None:
    grid = StructuredGrid2D(2, 2, 1.0, 1.0)
    material = NonlinearStateBatch(8)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    trial_a = elements.evaluate_samples(np.zeros((2, 2, 2, 3)), time_increment=0.1)
    elements.complete_trial(trial_a)
    elements.revert()
    with pytest.raises(RuntimeError):
        elements.commit()
    trial_b = elements.evaluate_samples(np.full((2, 2, 2, 3), 0.1), time_increment=0.1)
    elements.complete_trial(trial_b)
    elements.commit()
    assert material.reverts == 1
    assert material.commits == 1
    np.testing.assert_allclose(
        material.committed_stress, trial_b.material_trial.stress_in_plane_mpa
    )


def test_side_resultants_and_moment_use_nodal_coordinates() -> None:
    grid = StructuredGrid2D(2, 2, 2.0, 3.0)
    reaction = np.zeros((*grid.node_shape, 2))
    reaction[-1, :, 1] = 2.0
    reaction[:, 0, 0] = 1.0
    sides = side_resultants(reaction)
    np.testing.assert_allclose(sides[1], (1.0, 6.0))
    np.testing.assert_allclose(sides[2], (3.0, 2.0))
    assert moment(reaction, grid) == 12.0


def test_two_state_solver_backend_equivalence() -> None:
    pytest.importorskip("pyfftw")
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    x, y = grid.coordinates
    boundary = np.zeros((3, *grid.node_shape, 2))
    boundary[1:, ..., 0] = 0.04 * x[:, None]
    boundary[1:, ..., 1] = 0.03 * y[None, :]
    results = {}
    for backend in ("scipy", "fftw"):
        results[backend] = solve_two_state_dirichlet_plane_stress(
            grid=grid,
            material=NonlinearStateBatch(32),
            boundary_displacement_history=boundary,
            config=EBISpectralSolverConfig(
                relative_equilibrium_tolerance=1.0e-10,
                transform=SpectralTransformConfig(
                    backend=backend, fftw_planner_effort="estimate", fftw_use_wisdom=False
                ),
            ),
        )
        assert results[backend].diagnostics.dimensionless_equilibrium_history[-1] < 1.0e-10
    np.testing.assert_allclose(
        results["fftw"].displacement, results["scipy"].displacement, rtol=1.0e-11, atol=1.0e-12
    )
    np.testing.assert_allclose(
        results["fftw"].stress_in_plane_mpa,
        results["scipy"].stress_in_plane_mpa,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
