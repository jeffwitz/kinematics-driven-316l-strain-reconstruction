import numpy as np
import pytest

from fem_inhouse.core import (
    PLANE_STRESS_VON_MISES_METRIC,
    StructuredMesh,
    assemble_stiffness,
    consistent_tangent,
    element_tangent_stiffness,
    make_hardening,
    plane_stress_elasticity,
    precompute_element,
    return_mapping,
    shape_function_derivatives,
    strain_displacement_matrix,
    von_mises,
)
from fem_inhouse.core.assembly import FixedCSRAssembler, internal_force
from fem_inhouse.core.element import GAUSS_POINT_COUNT


def test_shape_derivatives_and_element_matrix() -> None:
    derivatives = shape_function_derivatives(0.0, 0.0)
    np.testing.assert_allclose(derivatives.sum(axis=1), 0.0)

    mesh = StructuredMesh(0.002, 0.002, 0.001, 1.0)
    elasticity = plane_stress_elasticity(205_000.0, 0.3)
    matrix_b, determinant = strain_displacement_matrix(
        mesh.coords[mesh.conn[0]],
        0.0,
        0.0,
    )
    operators = precompute_element(mesh, elasticity)
    element_matrix = operators.elastic_stiffness
    matrices_b = operators.strain_displacement
    determinants = operators.jacobian_determinants

    assert determinant > 0
    assert matrix_b.shape == (3, 8)
    assert matrices_b.shape == (4, 3, 8)
    assert np.all(determinants > 0)
    np.testing.assert_allclose(element_matrix, element_matrix.T, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(element_matrix)
    assert np.count_nonzero(np.abs(eigenvalues) < 1e-7) == 3


def test_assembly_and_internal_force_contracts() -> None:
    mesh = StructuredMesh(0.002, 0.002, 0.001, 1.0)
    elasticity = plane_stress_elasticity(205_000.0, 0.3)
    operators = precompute_element(mesh, elasticity)
    element_matrix = operators.elastic_stiffness
    matrices_b = operators.strain_displacement
    determinants = operators.jacobian_determinants
    location = mesh.location_matrix()

    stiffness = assemble_stiffness(mesh, element_matrix, location)
    element_matrices = np.broadcast_to(
        element_matrix,
        (mesh.n_elems, *element_matrix.shape),
    )
    rows = location[:, :, None].repeat(8, axis=2).ravel()
    columns = location[:, None, :].repeat(8, axis=1).ravel()
    stiffness_precomputed = assemble_stiffness(
        mesh,
        element_matrices,
        location,
        (rows, columns),
    )
    force = internal_force(
        mesh,
        np.zeros((mesh.n_elems, GAUSS_POINT_COUNT, 3)),
        matrices_b,
        determinants,
        location,
    )

    assert stiffness.shape == (mesh.n_dof, mesh.n_dof)
    np.testing.assert_allclose(stiffness.toarray(), stiffness_precomputed.toarray())
    np.testing.assert_allclose(force, 0.0)


def test_fixed_csr_assembler_preserves_structure_and_values() -> None:
    mesh = StructuredMesh(0.004, 0.003, 0.001, 1.0)
    elasticity = plane_stress_elasticity(205_000.0, 0.3)
    element_matrix = precompute_element(mesh, elasticity).elastic_stiffness
    location = mesh.location_matrix()
    free = mesh.dofs_free
    assembler = FixedCSRAssembler.from_location_matrix(
        location,
        free,
        chunk_size=2,
    )

    first = assembler.assemble(element_matrix)
    reference_first = assemble_stiffness(
        mesh,
        element_matrix,
        location,
    )[free][:, free].tocsr()
    first_values = first.toarray().copy()
    matrix_identity = id(first)
    indptr = first.indptr.copy()
    indices = first.indices.copy()

    element_matrices = np.broadcast_to(
        1.1 * element_matrix,
        (mesh.n_elems, 8, 8),
    ).copy()
    second = assembler.assemble(element_matrices)
    reference_second = assemble_stiffness(
        mesh,
        element_matrices,
        location,
    )[free][:, free].tocsr()

    assert id(second) == matrix_identity
    assert np.array_equal(second.indptr, indptr)
    assert np.array_equal(second.indices, indices)
    np.testing.assert_allclose(first_values, reference_first.toarray(), atol=1e-9)
    np.testing.assert_allclose(
        second.toarray(),
        reference_second.toarray(),
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("location", "dofs", "message"),
    [
        (np.zeros((2, 7), dtype=int), np.array([0]), "location_matrix"),
        (np.zeros((2, 8), dtype=int), np.array([], dtype=int), "selected_dofs"),
        (np.zeros((2, 8), dtype=int), np.array([0, 0]), "unique"),
    ],
)
def test_fixed_csr_assembler_rejects_invalid_contracts(
    location,
    dofs,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        FixedCSRAssembler.from_location_matrix(location, dofs)


def test_chunked_plastic_tangent_matches_dense_gauss_tensor() -> None:
    mesh = StructuredMesh(0.003, 0.001, 0.001, 1.0)
    elasticity = plane_stress_elasticity(205_000.0, 0.3)
    operators = precompute_element(mesh, elasticity)
    plastic_indices = np.array([0, 2, 5, 11])
    plastic_tangents = np.stack(
        (
            0.80 * elasticity,
            0.85 * elasticity,
            0.90 * elasticity,
            0.95 * elasticity,
        )
    )
    dense_tangents = np.broadcast_to(
        elasticity,
        (mesh.n_elems * GAUSS_POINT_COUNT, 3, 3),
    ).copy()
    dense_tangents[plastic_indices] = plastic_tangents
    dense_tangents = dense_tangents.reshape(
        mesh.n_elems,
        GAUSS_POINT_COUNT,
        3,
        3,
    )
    dense_product = np.einsum(
        "egij,gjk->egik",
        dense_tangents,
        operators.strain_displacement,
    )
    expected = np.einsum(
        "g,g,gik,egil->ekl",
        np.ones(GAUSS_POINT_COUNT),
        operators.jacobian_determinants,
        operators.strain_displacement,
        dense_product,
    )

    chunked = element_tangent_stiffness(
        operators.elastic_stiffness,
        elasticity,
        plastic_tangents,
        plastic_indices,
        operators.strain_displacement,
        operators.jacobian_determinants,
        element_count=mesh.n_elems,
        chunk_size=2,
    )

    np.testing.assert_allclose(chunked, expected, rtol=1e-13, atol=1e-9)


def test_chunked_plastic_tangent_rejects_invalid_contracts() -> None:
    elasticity = np.eye(3)
    matrices_b = np.zeros((GAUSS_POINT_COUNT, 3, 8))
    determinants = np.ones(GAUSS_POINT_COUNT)
    with pytest.raises(ValueError, match="elastic_element_stiffness"):
        element_tangent_stiffness(
            np.zeros((3, 3)),
            elasticity,
            np.empty((0, 3, 3)),
            np.array([], dtype=int),
            matrices_b,
            determinants,
            element_count=1,
        )
    with pytest.raises(ValueError, match="out-of-range"):
        element_tangent_stiffness(
            np.zeros((8, 8)),
            elasticity,
            np.zeros((1, 3, 3)),
            np.array([4]),
            matrices_b,
            determinants,
            element_count=1,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("elasticity", np.zeros((2, 2)), "elasticity"),
        ("plastic_tangents", np.zeros((1, 3, 3)), "plastic_tangents"),
        ("strain_displacement", np.zeros((3, 3, 8)), "strain_displacement"),
        ("jacobian_determinants", np.ones(3), "jacobian_determinants"),
        ("element_count", 0, "element_count"),
        ("chunk_size", 0, "chunk_size"),
    ],
)
def test_chunked_plastic_tangent_rejects_each_invalid_shape(
    keyword,
    value,
    message,
) -> None:
    arguments = {
        "elastic_element_stiffness": np.zeros((8, 8)),
        "elasticity": np.eye(3),
        "plastic_tangents": np.empty((0, 3, 3)),
        "plastic_flat_indices": np.array([], dtype=int),
        "strain_displacement": np.zeros((GAUSS_POINT_COUNT, 3, 8)),
        "jacobian_determinants": np.ones(GAUSS_POINT_COUNT),
        "element_count": 1,
        "chunk_size": 2,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        element_tangent_stiffness(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        (0.0, 0.002, 0.001, 1.0),
        (0.002, 0.002, 0.0, 1.0),
        (0.002, 0.002, 0.001, 0.0),
        (0.0025, 0.002, 0.001, 1.0),
    ],
)
def test_mesh_contract_rejects_invalid_geometry(arguments) -> None:
    with pytest.raises(ValueError):
        StructuredMesh(*arguments)


def test_element_and_assembly_contract_failures() -> None:
    with pytest.raises(ValueError, match="young_modulus"):
        plane_stress_elasticity(0.0, 0.3)
    with pytest.raises(ValueError, match="poisson_ratio"):
        plane_stress_elasticity(205_000.0, 0.5)
    clockwise_coordinates = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="non-positive Jacobian"):
        strain_displacement_matrix(clockwise_coordinates, 0.0, 0.0)

    mesh = StructuredMesh(0.002, 0.002, 0.001, 1.0)
    with pytest.raises(ValueError, match="element_stiffness"):
        assemble_stiffness(
            mesh,
            np.zeros((3, 3)),
            mesh.location_matrix(),
        )


def test_hardening_modes_and_von_mises() -> None:
    ludwik, ludwik_derivative = make_hardening(0.25, "ludwik")
    tabular, tabular_derivative = make_hardening(
        0.25,
        "tabular",
        plastic_strain_max=0.2,
        point_count=1_000,
    )
    strain = np.array([0.0, 0.01, 0.25])

    np.testing.assert_allclose(ludwik(strain), np.maximum(strain, 0.0) ** 0.25)
    assert ludwik_derivative(strain)[0] == 0.0
    assert tabular(strain)[-1] == pytest.approx(0.2**0.25)
    assert tabular_derivative(np.array([0.0]))[0] == pytest.approx(1e-6**0.25 / 1e-6)
    assert tabular_derivative(strain)[-1] == 0.0
    equivalent = von_mises(np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 10.0]]))
    np.testing.assert_allclose(equivalent, [100.0, np.sqrt(300.0)])

    with pytest.raises(ValueError, match="unknown hardening mode"):
        make_hardening(0.25, "unknown")


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("exponent", 0.0, "exponent"),
        ("plastic_strain_max", 0.0, "plastic_strain_max"),
        ("point_count", 2, "point_count"),
        ("first_positive_strain", 0.2, "first_positive_strain"),
    ],
)
def test_hardening_rejects_invalid_parameters(keyword, value, message) -> None:
    arguments = {
        "exponent": 0.245,
        "plastic_strain_max": 0.2,
        "point_count": 1_000,
        "first_positive_strain": 1e-6,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        make_hardening(**arguments)


@pytest.mark.parametrize("stress", [1.0, np.zeros((2, 2))])
def test_von_mises_rejects_invalid_stress_shape(stress) -> None:
    with pytest.raises(ValueError, match="final axis"):
        von_mises(stress)


def test_return_mapping_elastic_and_consistent_tangent() -> None:
    elasticity = plane_stress_elasticity(205_000.0, 0.3)
    metric_product = elasticity @ PLANE_STRESS_VON_MISES_METRIC
    cm11, cm12, cm33 = metric_product[0, 0], metric_product[0, 1], metric_product[2, 2]
    hardening, hardening_derivative = make_hardening(0.245, "ludwik")
    yield_stress = np.array([250.0])
    coefficient = np.array([500.0])
    accumulated_strain = np.array([0.0])

    elastic_trial = np.array([[100.0, 20.0, 5.0]])
    elastic_stress, plastic_increment, equivalent_increment = return_mapping(
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
        return return_mapping(
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
    tangent = consistent_tangent(
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


@pytest.mark.parametrize(
    "trial_stress",
    [
        [500.0, 0.0, 0.0],
        [500.0, 500.0, 0.0],
        [0.0, 0.0, 300.0],
    ],
    ids=["uniaxial", "equal-biaxial", "shear"],
)
def test_plastic_return_paths_end_on_yield_surface(trial_stress) -> None:
    elasticity = plane_stress_elasticity(205_000.0, 0.3)
    metric_product = elasticity @ PLANE_STRESS_VON_MISES_METRIC
    hardening, _ = make_hardening(0.245, "ludwik")
    stress, plastic_increment, equivalent_increment = return_mapping(
        np.asarray([trial_stress]),
        np.array([0.0]),
        np.array([250.0]),
        np.array([500.0]),
        hardening,
        metric_product[0, 0],
        metric_product[0, 1],
        metric_product[2, 2],
    )

    assert equivalent_increment[0] > 0
    assert np.linalg.norm(plastic_increment) > 0
    expected_yield = 250.0 + 500.0 * hardening(equivalent_increment)[0]
    assert von_mises(stress)[0] == pytest.approx(expected_yield, rel=1e-9)


def test_tabular_return_clamps_beyond_last_segment() -> None:
    elasticity = plane_stress_elasticity(205_000.0, 0.3)
    metric_product = elasticity @ PLANE_STRESS_VON_MISES_METRIC
    hardening, _ = make_hardening(0.245, "tabular")
    stress, _plastic_increment, equivalent_increment = return_mapping(
        np.array([[100_000.0, 0.0, 0.0]]),
        np.array([0.0]),
        np.array([250.0]),
        np.array([500.0]),
        hardening,
        metric_product[0, 0],
        metric_product[0, 1],
        metric_product[2, 2],
    )

    assert equivalent_increment[0] > 0.2
    expected_clamped_yield = 250.0 + 500.0 * 0.2**0.245
    assert von_mises(stress)[0] == pytest.approx(expected_clamped_yield, rel=1e-9)


def test_tabular_hardening_interpolates_every_segment() -> None:
    exponent = 0.245
    knots = np.concatenate((np.array([0.0]), np.linspace(1e-6, 0.2, 5)))
    values = knots**exponent
    hardening, derivative = make_hardening(
        exponent,
        "tabular",
        plastic_strain_max=0.2,
        point_count=6,
        first_positive_strain=1e-6,
    )
    midpoints = 0.5 * (knots[:-1] + knots[1:])
    expected_slopes = np.diff(values) / np.diff(knots)

    np.testing.assert_allclose(hardening(midpoints), 0.5 * (values[:-1] + values[1:]))
    np.testing.assert_allclose(derivative(midpoints), expected_slopes)
    assert hardening(np.array([1.0]))[0] == pytest.approx(values[-1])
    assert derivative(np.array([1.0]))[0] == 0.0
