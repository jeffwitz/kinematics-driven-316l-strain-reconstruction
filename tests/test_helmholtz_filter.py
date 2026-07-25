import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from fem_inhouse.postprocessing.helmholtz import helmholtz_filter_element_field


def _neumann_1d(size: int, spacing: float) -> sparse.csr_matrix:
    matrix = sparse.lil_matrix((size, size), dtype=float)
    for index in range(size - 1):
        coefficient = 1.0 / spacing**2
        matrix[index, index] += coefficient
        matrix[index + 1, index + 1] += coefficient
        matrix[index, index + 1] -= coefficient
        matrix[index + 1, index] -= coefficient
    return matrix.tocsr()


def _direct_filter(field: np.ndarray, length: float, hx: float, hy: float) -> np.ndarray:
    nx, ny = field.shape
    lx = _neumann_1d(nx, hx)
    ly = _neumann_1d(ny, hy)
    operator = sparse.eye(nx * ny, format="csr") + length**2 * (
        sparse.kron(lx, sparse.eye(ny), format="csr")
        + sparse.kron(sparse.eye(nx), ly, format="csr")
    )
    return np.asarray(spsolve(operator, field.ravel())).reshape(field.shape)


def test_zero_length_returns_exact_independent_float64_copy() -> None:
    source = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = helmholtz_filter_element_field(
        source,
        length_scale_mm=0.0,
        spacing_x_mm=0.2,
        spacing_y_mm=0.3,
    )

    assert result.source_element_field.dtype == np.float64
    assert result.filtered_element_field.dtype == np.float64
    assert np.array_equal(result.filtered_element_field, source)
    assert not np.shares_memory(result.filtered_element_field, source)
    assert not np.shares_memory(result.source_element_field, result.filtered_element_field)
    assert result.mean_drift == 0.0
    assert result.residual_relative == 0.0


def test_constant_mean_and_bounds_are_preserved() -> None:
    constant = np.full((8, 6), 3.25)
    constant_result = helmholtz_filter_element_field(
        constant,
        length_scale_mm=0.7,
        spacing_x_mm=0.2,
        spacing_y_mm=0.3,
    )
    np.testing.assert_allclose(constant_result.filtered_element_field, constant, atol=1e-14)

    rng = np.random.default_rng(7)
    source = rng.normal(size=(9, 11))
    result = helmholtz_filter_element_field(
        source,
        length_scale_mm=0.4,
        spacing_x_mm=0.2,
        spacing_y_mm=0.3,
    )
    assert abs(result.mean_drift) <= 1e-12 * max(1.0, abs(float(np.mean(source))))
    assert np.min(result.filtered_element_field) >= np.min(source) - 1e-13
    assert np.max(result.filtered_element_field) <= np.max(source) + 1e-13


def test_variance_is_nonincreasing_with_length() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(13, 10))
    variances = []
    for length in (0.0, 0.05, 0.2, 0.8):
        result = helmholtz_filter_element_field(
            source,
            length_scale_mm=length,
            spacing_x_mm=0.1,
            spacing_y_mm=0.1,
        )
        variances.append(float(np.var(result.filtered_element_field)))
    assert np.all(np.diff(variances) <= 1e-14)


def test_square_grid_dirac_response_is_symmetric() -> None:
    source = np.zeros((9, 9))
    source[4, 4] = 1.0
    filtered = helmholtz_filter_element_field(
        source,
        length_scale_mm=0.25,
        spacing_x_mm=0.1,
        spacing_y_mm=0.1,
    ).filtered_element_field

    np.testing.assert_allclose(filtered, filtered[::-1, :], atol=1e-14)
    np.testing.assert_allclose(filtered, filtered[:, ::-1], atol=1e-14)
    np.testing.assert_allclose(filtered, filtered.T, atol=1e-14)


def test_unequal_spacing_has_expected_numerical_anisotropy() -> None:
    source = np.zeros((17, 17))
    source[8, 8] = 1.0
    filtered = helmholtz_filter_element_field(
        source,
        length_scale_mm=0.3,
        spacing_x_mm=0.05,
        spacing_y_mm=0.2,
    ).filtered_element_field

    # A one-cell step is physically shorter in x, hence it is more strongly coupled.
    assert filtered[9, 8] > filtered[8, 9]


def test_spectral_solution_matches_direct_sparse_reference() -> None:
    rng = np.random.default_rng(17)
    source = rng.normal(size=(5, 4))
    length, hx, hy = 0.37, 0.16, 0.23
    spectral = helmholtz_filter_element_field(
        source,
        length_scale_mm=length,
        spacing_x_mm=hx,
        spacing_y_mm=hy,
    )
    direct = _direct_filter(source, length, hx, hy)

    np.testing.assert_allclose(spectral.filtered_element_field, direct, rtol=1e-11, atol=1e-12)
    assert spectral.residual_relative < 1e-11


@pytest.mark.parametrize(
    ("field", "length", "hx", "hy", "message"),
    [
        (np.zeros(3), 0.1, 1.0, 1.0, "two-dimensional"),
        (np.empty((0, 3)), 0.1, 1.0, 1.0, "empty"),
        (np.array([[np.nan]]), 0.1, 1.0, 1.0, "finite"),
        (np.zeros((2, 2)), -0.1, 1.0, 1.0, "nonnegative"),
        (np.zeros((2, 2)), np.inf, 1.0, 1.0, "finite"),
        (np.zeros((2, 2)), 0.1, 0.0, 1.0, "strictly positive"),
    ],
)
def test_invalid_inputs_are_rejected(field, length, hx, hy, message) -> None:
    with pytest.raises(ValueError, match=message):
        helmholtz_filter_element_field(
            field,
            length_scale_mm=length,
            spacing_x_mm=hx,
            spacing_y_mm=hy,
        )
