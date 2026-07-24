import numpy as np
import pytest

from fem_inhouse.core import solver_legacy as kernel


def test_shape_derivatives_and_element_matrix() -> None:
    derivatives = kernel._dN(0.0, 0.0)
    np.testing.assert_allclose(derivatives.sum(axis=1), 0.0)

    mesh = kernel._Mesh(0.002, 0.002, 0.001, 1.0)
    elasticity = kernel._Cps(205_000.0, 0.3)
    matrix_b, determinant = kernel._B_detJ(mesh.coords[mesh.conn[0]], 0.0, 0.0)
    element_matrix, matrices_b, determinants = kernel._precomp(mesh, elasticity)

    assert determinant > 0
    assert matrix_b.shape == (3, 8)
    assert matrices_b.shape == (4, 3, 8)
    assert np.all(determinants > 0)
    np.testing.assert_allclose(element_matrix, element_matrix.T, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(element_matrix)
    assert np.count_nonzero(np.abs(eigenvalues) < 1e-7) == 3


def test_assembly_and_internal_force_contracts() -> None:
    mesh = kernel._Mesh(0.002, 0.002, 0.001, 1.0)
    elasticity = kernel._Cps(205_000.0, 0.3)
    element_matrix, matrices_b, determinants = kernel._precomp(mesh, elasticity)
    location = np.empty((mesh.n_elems, 8), dtype=int)
    location[:, 0::2] = 2 * mesh.conn
    location[:, 1::2] = 2 * mesh.conn + 1

    stiffness = kernel._assemble(mesh, element_matrix, location)
    element_matrices = np.broadcast_to(
        element_matrix,
        (mesh.n_elems, *element_matrix.shape),
    )
    rows = location[:, :, None].repeat(8, axis=2).ravel()
    columns = location[:, None, :].repeat(8, axis=1).ravel()
    stiffness_precomputed = kernel._assemble(
        mesh,
        element_matrices,
        location,
        (rows, columns),
    )
    force = kernel._Fint(
        mesh,
        np.zeros((mesh.n_elems, kernel.N_GP, 3)),
        matrices_b,
        determinants,
        location,
    )

    assert stiffness.shape == (mesh.n_dof, mesh.n_dof)
    np.testing.assert_allclose(stiffness.toarray(), stiffness_precomputed.toarray())
    np.testing.assert_allclose(force, 0.0)


def test_hardening_modes_and_von_mises() -> None:
    ludwik, ludwik_derivative = kernel.make_hardening(0.25, "ludwik")
    tabular, tabular_derivative = kernel.make_hardening(
        0.25,
        "tabular",
        ep_max=0.2,
        n_pts=1_000,
    )
    strain = np.array([0.0, 0.01, 0.25])

    np.testing.assert_allclose(ludwik(strain), np.maximum(strain, 0.0) ** 0.25)
    assert ludwik_derivative(strain)[0] == 0.0
    assert tabular(strain)[-1] == pytest.approx(0.2**0.25)
    assert tabular_derivative(np.array([0.0]))[0] == pytest.approx(1e-6**0.25 / 1e-6)
    assert tabular_derivative(strain)[-1] == 0.0
    equivalent = kernel._vm(np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 10.0]]))
    np.testing.assert_allclose(equivalent, [100.0, np.sqrt(300.0)])

    with pytest.raises(ValueError, match="unknown hardening mode"):
        kernel.make_hardening(0.25, "unknown")


def test_return_mapping_elastic_and_consistent_tangent() -> None:
    elasticity = kernel._Cps(205_000.0, 0.3)
    metric_product = elasticity @ kernel._M
    cm11, cm12, cm33 = metric_product[0, 0], metric_product[0, 1], metric_product[2, 2]
    hardening, hardening_derivative = kernel.make_hardening(0.245, "ludwik")
    yield_stress = np.array([250.0])
    coefficient = np.array([500.0])
    accumulated_strain = np.array([0.0])

    elastic_trial = np.array([[100.0, 20.0, 5.0]])
    elastic_stress, plastic_increment, equivalent_increment = kernel._rm(
        elastic_trial,
        accumulated_strain,
        yield_stress,
        coefficient,
        hardening,
        cm11,
        cm12,
        cm33,
    )
    np.testing.assert_array_equal(elastic_stress, elastic_trial)
    np.testing.assert_array_equal(plastic_increment, 0.0)
    np.testing.assert_array_equal(equivalent_increment, 0.0)

    total_strain = np.array([0.005, -0.001, 0.002])

    def mapped_stress(strain):
        trial = (elasticity @ strain)[None, :]
        return kernel._rm(
            trial,
            accumulated_strain,
            yield_stress,
            coefficient,
            hardening,
            cm11,
            cm12,
            cm33,
        )

    stress, _, increment = mapped_stress(total_strain)
    assert increment[0] > 0
    tangent = kernel._cep(
        stress,
        increment,
        accumulated_strain,
        yield_stress,
        coefficient,
        hardening,
        hardening_derivative,
        elasticity,
        cm11,
        cm12,
        cm33,
    )[0]
    finite_difference = np.empty((3, 3))
    step = 1e-8
    for component in range(3):
        perturbed = total_strain.copy()
        perturbed[component] += step
        perturbed_stress, _, _ = mapped_stress(perturbed)
        finite_difference[:, component] = (perturbed_stress[0] - stress[0]) / step

    relative_error = np.linalg.norm(tangent - finite_difference) / np.linalg.norm(finite_difference)
    assert relative_error < 1e-5
