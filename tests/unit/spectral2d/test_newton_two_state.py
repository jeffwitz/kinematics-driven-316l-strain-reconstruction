import numpy as np

from fem_inhouse.core.plane_stress_material import ConstitutiveTrial, InPlaneConstitutiveTrial
from fem_inhouse.spectral2d import EBITwoTriangleKinematics2D, StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import TraditionalTwoStateTriangleBatch
from scripts.qualify_ebi_state_sharing import moment, side_resultants


class NonlinearStateBatch:
    def __init__(self, point_count: int) -> None:
        self.point_count = point_count
        self.calls = 0
        self.commits = 0
        self.reverts = 0
        self._stiffness = np.broadcast_to(
            np.array([[4.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 1.5]]),
            (point_count, 3, 3),
        ).copy()

    def evaluate_in_plane(self, strain, *, time_increment, consistent_tangent=True):
        self.calls += 1
        values = np.asarray(strain, dtype=float).reshape(-1, 3)
        stress = np.einsum("pij,pj->pi", self._stiffness, values) + 0.2 * values**2
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
        self.commits += 1

    def revert(self):
        self.reverts += 1


def test_two_state_plastic_nonlinear_directional_tangent() -> None:
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
    trial = elements.evaluate_samples(np.zeros((2, 2, 2, 3)), time_increment=0.1)
    elements.complete_trial(trial)
    elements.revert()
    elements.commit()
    assert material.reverts == 1
    assert material.commits == 1


def test_side_resultants_and_moment_use_nodal_coordinates() -> None:
    grid = StructuredGrid2D(2, 2, 2.0, 3.0)
    reaction = np.zeros((*grid.node_shape, 2))
    reaction[-1, :, 1] = 2.0
    reaction[:, 0, 0] = 1.0
    sides = side_resultants(reaction)
    np.testing.assert_allclose(sides[1], (1.0, 6.0))
    np.testing.assert_allclose(sides[2], (3.0, 2.0))
    assert moment(reaction, grid) == 12.0
