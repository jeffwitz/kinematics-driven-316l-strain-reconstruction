from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.results import load_full_tensor_state


def test_load_full_tensor_state_reads_native_saved_fields(tmp_path) -> None:
    shape = (2, 3)
    stress = np.zeros((*shape, 3, 3))
    stress[..., 0, 0] = 100.0
    stress[..., 2, 2] = 2e-12
    total = np.zeros_like(stress)
    elastic = np.zeros_like(stress)
    plastic = np.zeros_like(stress)
    residual = stress[..., 2, 2].copy()
    residual_vector = np.zeros((*shape, 3))
    residual_vector[..., 0] = residual
    for name, values in (
        ("S_3D", stress),
        ("E_3D", total),
        ("EE_3D", elastic),
        ("PE_3D", plastic),
        ("S33_RESIDUAL_MPA", residual),
        ("PLANE_STRESS_RESIDUAL_MPA", residual_vector),
    ):
        np.save(tmp_path / f"{name}.npy", values)

    state = load_full_tensor_state(tmp_path)

    np.testing.assert_array_equal(state.stress_tensor_mpa, stress)
    np.testing.assert_array_equal(state.plane_stress_residual_mpa, residual)
    np.testing.assert_array_equal(
        state.plane_stress_residual_vector_mpa,
        residual_vector,
    )


def test_load_full_tensor_state_reconstructs_legacy_fields_with_material_data(
    tmp_path,
) -> None:
    stress = np.array([[[300.0, 120.0, 25.0]]])
    total = np.array([[[0.01, 0.001, 0.004]]])
    plastic = np.array([[[0.007, -0.001, 0.003]]])
    for name, values in (("S", stress), ("E", total), ("PE", plastic)):
        np.save(tmp_path / f"{name}.npy", values)

    with pytest.raises(ValueError, match="completion_strategy"):
        load_full_tensor_state(tmp_path)
    with pytest.raises(ValueError, match="poisson_ratio is required"):
        load_full_tensor_state(
            tmp_path,
            completion_strategy="j2_isotropic_analytical",
        )

    state = load_full_tensor_state(
        tmp_path,
        poisson_ratio=0.3,
        completion_strategy="j2_isotropic_analytical",
    )
    np.testing.assert_array_equal(state.stress_tensor_mpa[..., 0, 0], stress[..., 0])
    np.testing.assert_allclose(
        np.trace(state.plastic_strain_tensor, axis1=-2, axis2=-1),
        0.0,
        atol=1e-15,
    )


def test_load_full_tensor_state_rejects_incomplete_new_files(tmp_path) -> None:
    np.save(tmp_path / "S_3D.npy", np.zeros((1, 1, 3, 3)))

    with pytest.raises(RuntimeError, match="incomplete full tensor state"):
        load_full_tensor_state(tmp_path, poisson_ratio=0.3)
