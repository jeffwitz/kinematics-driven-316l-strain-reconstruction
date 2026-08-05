"""Element-based internal-variable reconstruction for two-triangle pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.plane_stress_material import (
    ConstitutiveTrial,
    HookeanPlaneStressMaterialBatch,
    InPlaneConstitutiveTrial,
    ResponseLevel,
    evaluate_in_plane_response,
)
from fem_inhouse.spectral2d.kinematics import EBITwoTriangleKinematics2D

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EBIPlaneStressTrial:
    mean_trial: InPlaneConstitutiveTrial
    sample_strain: FloatArray
    mean_strain: FloatArray
    sample_stress_mpa: FloatArray
    mean_stress_mpa: FloatArray
    elastic_tangent_in_plane_mpa: FloatArray
    algorithmic_tangent_in_plane_mpa: FloatArray | None


class EBIPlaneStressElementBatch:
    """Two kinematic samples sharing one Hookean internal state per pixel."""

    def __init__(
        self,
        material: HookeanPlaneStressMaterialBatch,
        pixel_shape: tuple[int, int],
    ) -> None:
        if material.point_count != pixel_shape[0] * pixel_shape[1]:
            raise ValueError("EBI requires exactly one material state per pixel")
        self.material = material
        self.pixel_shape = pixel_shape

    def evaluate_samples(
        self,
        sample_strain: ArrayLike,
        *,
        time_increment: float,
        response_level: ResponseLevel = "tangent",
        consistent_tangent: bool,
    ) -> EBIPlaneStressTrial:
        values = np.asarray(sample_strain, dtype=np.float64)
        expected = (*self.pixel_shape, 2, 3)
        if values.shape != expected:
            raise ValueError(f"expected EBI sample strain shape {expected}, got {values.shape}")
        mean_strain = 0.5 * (values[..., 0, :] + values[..., 1, :])
        mean_trial = evaluate_in_plane_response(
            self.material,
            mean_strain.reshape(-1, 3),
            time_increment=time_increment,
            response_level=response_level,
            consistent_tangent=consistent_tangent,
        )
        if response_level == "tangent" and mean_trial.tangent_in_plane_mpa is None:
            raise ValueError("EBI tangent action requires the algorithmic tangent")
        mean_stress = np.asarray(mean_trial.stress_in_plane_mpa).reshape(*self.pixel_shape, 3)
        elastic_tangent = np.asarray(
            self.material.elastic_tangent_in_plane_mpa, dtype=np.float64
        ).reshape(*self.pixel_shape, 3, 3)
        algorithmic_tangent = (
            None
            if mean_trial.tangent_in_plane_mpa is None
            else np.asarray(mean_trial.tangent_in_plane_mpa, dtype=np.float64).reshape(
                *self.pixel_shape, 3, 3
            )
        )
        fluctuation = values - mean_strain[..., None, :]
        sample_stress = mean_stress[..., None, :] + np.einsum(
            "xyij,xyqj->xyqi", elastic_tangent, fluctuation
        )
        return EBIPlaneStressTrial(
            mean_trial=mean_trial,
            sample_strain=values.copy(),
            mean_strain=mean_strain,
            sample_stress_mpa=sample_stress,
            mean_stress_mpa=mean_stress,
            elastic_tangent_in_plane_mpa=elastic_tangent,
            algorithmic_tangent_in_plane_mpa=algorithmic_tangent,
        )

    def tangent_action(
        self,
        displacement_increment: ArrayLike,
        *,
        kinematics: EBITwoTriangleKinematics2D,
        trial: EBIPlaneStressTrial,
    ) -> FloatArray:
        delta_sample = kinematics.strain_samples(displacement_increment)
        delta_mean = kinematics.mean_strain(delta_sample)
        if trial.algorithmic_tangent_in_plane_mpa is None:
            raise ValueError("EBI tangent action requires the algorithmic tangent")
        delta_mean_stress = np.einsum(
            "xyij,xyj->xyi", trial.algorithmic_tangent_in_plane_mpa, delta_mean
        )
        delta_fluctuation_stress = np.einsum(
            "xyij,xyqj->xyqi",
            trial.elastic_tangent_in_plane_mpa,
            delta_sample - delta_mean[..., None, :],
        )
        return kinematics.divergence_from_sample_stress(
            delta_mean_stress[..., None, :] + delta_fluctuation_stress
        )

    def complete_trial(self, trial: EBIPlaneStressTrial) -> ConstitutiveTrial:
        return self.material.complete_trial(trial.mean_trial)

    def commit(self) -> None:
        self.material.commit()

    def revert(self) -> None:
        self.material.revert()


def hookean_plane_stress_relative_error(
    trial: ConstitutiveTrial,
    elastic_tangent_in_plane_mpa: ArrayLike,
) -> float:
    """Check the condensed Hookean relation using the full elastic strain state."""

    elastic = np.asarray(trial.elastic_strain_tensor, dtype=np.float64)
    strain = np.stack(
        (elastic[..., 0, 0], elastic[..., 1, 1], 2.0 * elastic[..., 0, 1]),
        axis=-1,
    )
    tangent = np.asarray(elastic_tangent_in_plane_mpa, dtype=np.float64).reshape(
        *strain.shape[:-1], 3, 3
    )
    predicted = np.einsum("...ij,...j->...i", tangent, strain)
    stress = np.asarray(trial.stress_in_plane_mpa, dtype=np.float64).reshape(predicted.shape)
    return float(np.linalg.norm(stress - predicted) / max(np.linalg.norm(stress), 1.0))
