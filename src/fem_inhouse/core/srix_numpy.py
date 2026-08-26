"""Vectorised NumPy implementation of the qualified Forest--Rubin SRIX law.

The module deliberately contains no MGIS/MFront dependency.  It is a small,
transactional material-point backend used as an independent implementation and
as the future NumPy/CuPy seam.  The local Newton system follows
``validation/mfront/Fcc316LForestRubinSrixGeneric3D.mfront`` line for line;
MFront remains the production default and the numerical oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.crystal_orientation import validate_rotations
from fem_inhouse.core.fcc_interaction_matrix import (
    build_interaction_matrix,
    slip_systems,
)
from fem_inhouse.core.linear_solver import LinearSystemMatrixType
from fem_inhouse.core.mfront_runtime import (
    _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
    _KELVIN_TO_ENGINEERING_STRESS_SCALE,
)
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    ConstitutiveTrial,
    InPlaneConstitutiveTrial,
    PlaneStressBatchStatistics,
)
from fem_inhouse.core.srix_parameters import (
    DEFAULT_PARAMETER_SET,
    SrixParameterSet,
    get_parameter_set,
    resolve_srix_parameters,
)
from fem_inhouse.core.tensor_reconstruction import (
    kelvin_3d_to_tensor,
    tensor_to_engineering_stress_2d,
)

FloatArray = NDArray[np.float64]
_SQRT_TWO = np.sqrt(2.0)
_PLANE = np.array([0, 1, 3])
_TRANSVERSE = np.array([2, 4, 5])


def _deviatoric(values: FloatArray) -> FloatArray:
    result = values.copy()
    mean = np.mean(result[..., :3], axis=-1)
    result[..., :3] -= mean[..., None]
    return result


def _schmid_kelvin() -> FloatArray:
    values = []
    for system in slip_systems():
        direction = system.burgers.astype(float)
        normal = system.normal.astype(float)
        direction /= np.linalg.norm(direction)
        normal /= np.linalg.norm(normal)
        tensor = 0.5 * (np.outer(direction, normal) + np.outer(normal, direction))
        values.append(
            np.array(
                [
                    tensor[0, 0],
                    tensor[1, 1],
                    tensor[2, 2],
                    _SQRT_TWO * tensor[0, 1],
                    _SQRT_TWO * tensor[0, 2],
                    _SQRT_TWO * tensor[1, 2],
                ]
            )
        )
    return np.asarray(values, dtype=float)


def _cubic_kelvin(parameters: SrixParameterSet) -> FloatArray:
    c11 = parameters.elasticity.c11_mpa
    c12 = parameters.elasticity.c12_mpa
    c44 = parameters.elasticity.c44_mpa
    result = np.zeros((6, 6), dtype=float)
    result[:3, :3] = c12
    np.fill_diagonal(result[:3, :3], c11)
    result[3:, 3:] = np.eye(3) * (2.0 * c44)
    return result


def _kelvin_rotation(rotation: FloatArray) -> FloatArray:
    """Return the Kelvin strain map ``global -> crystal`` for each point."""

    basis = np.eye(6)
    basis_tensor = kelvin_3d_to_tensor(basis, quantity="strain")
    transformed = np.einsum("nij,kjm,nlm->nkil", rotation, basis_tensor, rotation)
    result = np.empty((rotation.shape[0], 6, 6), dtype=float)
    result[..., 0] = transformed[..., 0, 0]
    result[..., 1] = transformed[..., 1, 1]
    result[..., 2] = transformed[..., 2, 2]
    result[..., 3] = _SQRT_TWO * transformed[..., 0, 1]
    result[..., 4] = _SQRT_TWO * transformed[..., 0, 2]
    result[..., 5] = _SQRT_TWO * transformed[..., 1, 2]
    return result


def _resolve_parameters(
    parameter_set: str | SrixParameterSet | None,
    explicit: dict[str, Any] | None,
) -> SrixParameterSet:
    if isinstance(parameter_set, SrixParameterSet):
        chosen = parameter_set
        identifier = chosen.identifier
    else:
        identifier = parameter_set or DEFAULT_PARAMETER_SET
        chosen = get_parameter_set(identifier)
    if explicit is None:
        return chosen
    # Reuse the registry's validation and vocabulary before replacing fields.
    resolve_srix_parameters(parameter_set=identifier, explicit=explicit)
    values: dict[str, Any] = {}
    mapping = {
        "R_mpa": "overstress_modulus_mpa",
        "tau0_mpa": "tau0_mpa",
        "Q_mpa": "q_mpa",
        "b": "b",
        "C_mpa": "c_mpa",
        "d": "d",
    }
    for source, target in mapping.items():
        if source in explicit:
            values[target] = float(explicit[source])
    if all(name in explicit for name in ("C11_mpa", "C12_mpa", "C44_mpa")):
        from fem_inhouse.core.single_crystal_presets import CubicElasticity

        values["elasticity"] = CubicElasticity(
            c11_mpa=float(explicit["C11_mpa"]),
            c12_mpa=float(explicit["C12_mpa"]),
            c44_mpa=float(explicit["C44_mpa"]),
        )
    return replace(chosen, **values)


@dataclass(frozen=True, slots=True)
class SrixNumpy3DTrial:
    total_strain_kelvin: FloatArray
    stress_kelvin_mpa: FloatArray
    elastic_strain_kelvin: FloatArray
    plastic_slip: FloatArray
    equivalent_plastic_slip: FloatArray
    back_strain: FloatArray
    accumulated_slip: FloatArray
    consistent_tangent_kelvin_mpa: FloatArray
    material_elastic_strain_kelvin: FloatArray


class SrixNumpy3DMaterialPointBatch:
    """Transactional, point-batched 3-D SRIX integrator."""

    def __init__(
        self,
        *,
        point_count: int,
        parameters: SrixParameterSet | str | None = None,
        parameter_set: str | SrixParameterSet | None = None,
        explicit_parameters: dict[str, Any] | None = None,
        rotation_global_to_material: ArrayLike | None = None,
        batch_size: int | None = None,
        maximum_local_iterations: int = 100,
        material_newton_max_iterations: int | None = None,
        local_tolerance: float = 1e-11,
    ) -> None:
        if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 1:
            raise ValueError("point_count must be a positive integer")
        if material_newton_max_iterations is not None:
            maximum_local_iterations = material_newton_max_iterations
        if maximum_local_iterations < 1 or local_tolerance <= 0:
            raise ValueError("invalid local Newton controls")
        if batch_size is not None and (isinstance(batch_size, bool) or batch_size < 1):
            raise ValueError("batch_size must be positive")
        selected = parameters if parameters is not None else parameter_set
        self.parameters = _resolve_parameters(selected, explicit_parameters)
        self._point_count = point_count
        self._batch_size = batch_size
        self._maximum_iterations = int(maximum_local_iterations)
        self._tolerance = float(local_tolerance)
        rotations = (
            np.broadcast_to(np.eye(3), (point_count, 3, 3)).copy()
            if rotation_global_to_material is None
            else validate_rotations(rotation_global_to_material, point_count=point_count)
        )
        self._rotation = rotations
        self._kelvin_rotation = _kelvin_rotation(rotations)
        ce = _cubic_kelvin(self.parameters)
        self._ce = np.broadcast_to(ce, (point_count, 6, 6)).copy()
        self._schmid = np.broadcast_to(_schmid_kelvin(), (point_count, 12, 6)).copy()
        self._ce_material = ce
        self._schmid_material = _schmid_kelvin()
        self._mce = self._schmid_material @ ce
        self._plastic_modulus = self._mce @ self._schmid_material.T
        self._interaction = build_interaction_matrix(self.parameters.interaction_matrix)
        self._elastic = np.zeros((point_count, 6))
        self._g = np.zeros((point_count, 12))
        self._p = np.zeros((point_count, 12))
        self._a = np.zeros((point_count, 12))
        self._committed_total_strain = np.zeros((point_count, 6))
        self._trial: SrixNumpy3DTrial | None = None
        self._trial_state: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None
        self._trial_total_strain: FloatArray | None = None

    def _residual_for(
        self,
        deel: FloatArray,
        dg: FloatArray,
        material_strain: FloatArray,
        ce: FloatArray,
        mus: FloatArray,
        p0: FloatArray,
        a0: FloatArray,
        elastic0: FloatArray,
    ) -> FloatArray:
        """Evaluate the local SRIX residual for a batch of Newton states."""

        elastic = elastic0 + deel
        stress = np.einsum("nij,nj->ni", ce, elastic)
        tau = np.einsum("nsi,ni->ns", mus, stress)
        de = _deviatoric(deel) + np.einsum("nsi,ns->ni", mus, dg)
        deq = np.sqrt(np.maximum(2.0 * np.sum(de * de, axis=1) / 3.0, 0.0))
        abs_dg = np.abs(dg)
        p_trial = p0 + abs_dg
        exp_bp = np.exp(-self.parameters.b * p_trial)
        resistance = self.parameters.tau0_mpa + self.parameters.q_mpa * np.einsum(
            "ij,nj->ni", self._interaction, 1.0 - exp_bp
        )
        da = (dg - self.parameters.d * a0 * abs_dg) / (1.0 + self.parameters.d * abs_dg)
        drive = tau - self.parameters.c_mpa * (a0 + da)
        sgn = np.where(drive > 0.0, 1.0, -1.0)
        overstress = np.maximum(np.abs(drive) - resistance, 0.0)
        slope = deq / self.parameters.overstress_modulus_mpa
        flow = slope[:, None] * overstress * sgn
        return np.concatenate(
            (deel - material_strain + np.einsum("nsi,ns->ni", mus, dg), dg - flow), axis=1
        )

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def backend_name(self) -> str:
        return "numpy-srix-3d"

    @property
    def rotations_global_to_material(self) -> FloatArray:
        return self._rotation.copy()

    @property
    def committed_state(self) -> dict[str, FloatArray]:
        return {
            "elastic_strain": self._elastic.copy(),
            "g": self._g.copy(),
            "p": self._p.copy(),
            "a": self._a.copy(),
            "total_strain": self._committed_total_strain.copy(),
        }

    def _integrate_chunk_full(
        self,
        strain_increment: FloatArray,
        total_strain: FloatArray,
        start: int = 0,
    ) -> SrixNumpy3DTrial:
        n = strain_increment.shape[0]
        stop = start + n
        transform = self._kelvin_rotation[start:stop]
        material_strain = np.einsum("nij,nj->ni", transform, strain_increment)
        deel = material_strain.copy()
        dg = np.zeros((n, 12))
        ce = self._ce[start:stop]
        mus = self._schmid[start:stop]
        p0, a0 = self._p[start:stop], self._a[start:stop]
        elastic0 = self._elastic[start:stop]
        eye6 = np.eye(6)
        converged = False
        jac: FloatArray | None = None
        for iteration in range(self._maximum_iterations):
            elastic = elastic0 + deel
            stress = np.einsum("nij,nj->ni", ce, elastic)
            tau = np.einsum("nsi,ni->ns", mus, stress)
            de = _deviatoric(deel) + np.einsum("nsi,ns->ni", mus, dg)
            deq = np.sqrt(np.maximum(2.0 * np.sum(de * de, axis=1) / 3.0, 0.0))
            abs_dg = np.abs(dg)
            sign_dg = np.where(dg > 0.0, 1.0, np.where(dg < 0.0, -1.0, 0.0))
            p_trial = p0 + abs_dg
            exp_bp = np.exp(-self.parameters.b * p_trial)
            resistance = self.parameters.tau0_mpa + self.parameters.q_mpa * np.einsum(
                "ij,nj->ni", self._interaction, 1.0 - exp_bp
            )
            da = (dg - self.parameters.d * a0 * abs_dg) / (1.0 + self.parameters.d * abs_dg)
            drive = tau - self.parameters.c_mpa * (a0 + da)
            sgn = np.where(drive > 0.0, 1.0, -1.0)
            overstress = np.maximum(np.abs(drive) - resistance, 0.0)
            slope = deq / self.parameters.overstress_modulus_mpa
            flow = slope[:, None] * overstress * sgn
            residual = np.concatenate(
                (deel - material_strain + np.einsum("nsi,ns->ni", mus, dg), dg - flow), axis=1
            )
            residual_norm = np.max(np.abs(residual), axis=1)
            residual_small = residual_norm <= self._tolerance
            active = (overstress > 0.0).astype(float)
            ndeq = np.zeros_like(de)
            nonzero = deq > 1e-14
            ndeq[nonzero] = (2.0 / (3.0 * deq[nonzero, None])) * de[nonzero]
            mus_ce = np.einsum("nsi,nij->nsj", mus, ce)
            jfd = -active[:, :, None] * slope[:, None, None] * mus_ce
            jfd -= (overstress * sgn / self.parameters.overstress_modulus_mpa)[:, :, None] * ndeq[
                :, None, :
            ]
            jgg = np.broadcast_to(np.eye(12), (n, 12, 12)).copy()
            dg_abs_derivative = sign_dg
            for i in range(12):
                den = 1.0 + self.parameters.d * abs_dg[:, i]
                num = dg[:, i] - self.parameters.d * a0[:, i] * abs_dg[:, i]
                dnum = 1.0 - self.parameters.d * a0[:, i] * dg_abs_derivative[:, i]
                dden = self.parameters.d * dg_abs_derivative[:, i]
                dda = (dnum * den - num * dden) / (den * den)
                # The sign of ``abs(drive)`` cancels the flow sign for the
                # back-stress derivative (``ddrive=-C*dda``).
                jgg[:, i, i] += active[:, i] * slope * self.parameters.c_mpa * dda
                dr = (
                    self.parameters.q_mpa
                    * self._interaction[i][None, :]
                    * self.parameters.b
                    * exp_bp
                    * dg_abs_derivative
                )
                jgg[:, i, :] += active[:, i, None] * slope[:, None] * dr * sgn[:, i, None]
                jgg[:, i, :] -= (
                    overstress[:, i] * sgn[:, i] / self.parameters.overstress_modulus_mpa
                )[:, None] * np.einsum("ni,nji->nj", ndeq, mus)
            jac = np.zeros((n, 18, 18))
            jac[:, :6, :6] = eye6
            jac[:, :6, 6:] = np.swapaxes(mus, 1, 2)
            jac[:, 6:, :6] = jfd
            jac[:, 6:, 6:] = jgg
            if np.all(residual_small):
                converged = True
                break
            try:
                delta = np.linalg.solve(jac, -residual[..., None])[..., 0]
            except np.linalg.LinAlgError as error:
                raise ConstitutiveIntegrationError("NumPy SRIX local Newton is singular") from error
            if not np.isfinite(delta).all():
                raise ConstitutiveIntegrationError(
                    "NumPy SRIX local Newton produced non-finite values"
                )
            # Take a full Newton step whenever it decreases the batch
            # residual.  Backtracking is only used when a trial crosses an
            # absolute-value branch or otherwise increases the residual.
            current_norm = residual_norm
            alpha = np.ones(n)
            accepted = residual_small.copy()
            best_norm = np.where(residual_small, current_norm, np.inf)
            best_deel = deel.copy()
            best_dg = dg.copy()
            for _backtrack in range(12):
                candidate_deel = deel + alpha[:, None] * delta[:, :6]
                candidate_dg = dg + alpha[:, None] * delta[:, 6:]
                candidate = self._residual_for(
                    candidate_deel, candidate_dg, material_strain, ce, mus, p0, a0, elastic0
                )
                candidate_norm = np.max(np.abs(candidate), axis=1)
                finite = np.isfinite(candidate_norm)
                better = (~residual_small) & finite & (candidate_norm < best_norm)
                best_norm = np.where(better, candidate_norm, best_norm)
                best_deel = np.where(better[:, None], candidate_deel, best_deel)
                best_dg = np.where(better[:, None], candidate_dg, best_dg)
                newly_accepted = (~accepted) & finite & (candidate_norm <= current_norm)
                accepted |= newly_accepted
                alpha = np.where(accepted, alpha, 0.5 * alpha)
                if np.all(accepted):
                    break
            if not np.all(accepted):
                failed = np.flatnonzero(~accepted)
                worst = int(failed[np.argmax(current_norm[failed])])
                raise ConstitutiveIntegrationError(
                    "NumPy SRIX local Newton backtracking failed to decrease the residual "
                    f"(iteration={iteration + 1}, point={worst}, "
                    f"current_norm={current_norm[worst]:.6e}, "
                    f"best_norm={best_norm[worst]:.6e})"
                )
            deel = best_deel
            dg = best_dg
        if not converged:
            raise ConstitutiveIntegrationError(
                f"NumPy SRIX local Newton did not converge in {self._maximum_iterations} iterations"
            )
        # Rebuild the converged tangent from the same implicit Jacobian.
        rhs = np.zeros((n, 18, 6))
        rhs[:, :6, :] = eye6
        if jac is None:
            tangent_material = ce.copy()
        else:
            implicit = np.linalg.solve(jac, rhs)
            tangent_material = np.einsum("nij,njk->nik", ce, implicit[:, :6, :])
        stress_material = np.einsum("nij,nj->ni", ce, elastic0 + deel)
        elastic_global = np.einsum("nij,nj->ni", np.swapaxes(transform, 1, 2), elastic0 + deel)
        stress_global = np.einsum("nij,nj->ni", np.swapaxes(transform, 1, 2), stress_material)
        tangent_global = np.einsum(
            "nij,njk,nkl->nil", np.swapaxes(transform, 1, 2), tangent_material, transform
        )
        p_final = p0 + np.abs(dg)
        da_final = (dg - self.parameters.d * a0 * np.abs(dg)) / (
            1.0 + self.parameters.d * np.abs(dg)
        )
        return SrixNumpy3DTrial(
            total_strain_kelvin=total_strain.copy(),
            stress_kelvin_mpa=stress_global,
            elastic_strain_kelvin=elastic_global,
            plastic_slip=self._g[start:stop] + dg,
            equivalent_plastic_slip=p_final,
            back_strain=a0 + da_final,
            accumulated_slip=np.sum(p_final, axis=1),
            consistent_tangent_kelvin_mpa=tangent_global,
            material_elastic_strain_kelvin=elastic0 + deel,
        )

    def _reduced_residual(
        self,
        dg: FloatArray,
        tau_trial: FloatArray,
        deq: FloatArray,
        p0: FloatArray,
        a0: FloatArray,
    ) -> FloatArray:
        """Return the 12 slip residuals after eliminating elastic strain."""
        abs_dg = np.abs(dg)
        p_trial = p0 + abs_dg
        exp_bp = np.exp(-self.parameters.b * p_trial)
        resistance = self.parameters.tau0_mpa + self.parameters.q_mpa * np.einsum(
            "ij,nj->ni", self._interaction, 1.0 - exp_bp
        )
        tau = tau_trial - np.einsum("ij,nj->ni", self._plastic_modulus, dg)
        da = (dg - self.parameters.d * a0 * abs_dg) / (1.0 + self.parameters.d * abs_dg)
        drive = tau - self.parameters.c_mpa * (a0 + da)
        sgn = np.where(drive > 0.0, 1.0, -1.0)
        overstress = np.maximum(np.abs(drive) - resistance, 0.0)
        flow = (deq / self.parameters.overstress_modulus_mpa)[:, None] * overstress * sgn
        return dg - flow

    def _integrate_chunk(
        self,
        strain_increment: FloatArray,
        total_strain: FloatArray,
        start: int = 0,
    ) -> SrixNumpy3DTrial:
        """Integrate one chunk with the exact 12-slip Schur reduction."""
        n = strain_increment.shape[0]
        stop = start + n
        transform = self._kelvin_rotation[start:stop]
        material_strain = np.einsum("nij,nj->ni", transform, strain_increment)
        p0, a0 = self._p[start:stop], self._a[start:stop]
        elastic0 = self._elastic[start:stop]
        ce = self._ce_material
        mus = self._schmid_material
        plastic_modulus = self._plastic_modulus
        dg = np.zeros((n, 12))
        tau_trial = np.einsum("si,ij,nj->ns", mus, ce, elastic0 + material_strain)
        de = _deviatoric(material_strain)
        deq = np.sqrt(np.maximum(2.0 * np.sum(de * de, axis=1) / 3.0, 0.0))
        slope = deq / self.parameters.overstress_modulus_mpa
        eye12 = np.eye(12)
        converged = np.zeros(n, dtype=bool)
        jac: FloatArray | None = None
        for iteration in range(self._maximum_iterations):
            abs_dg = np.abs(dg)
            sign_dg = np.where(dg > 0.0, 1.0, np.where(dg < 0.0, -1.0, 0.0))
            p_trial = p0 + abs_dg
            exp_bp = np.exp(-self.parameters.b * p_trial)
            resistance = self.parameters.tau0_mpa + self.parameters.q_mpa * np.einsum(
                "ij,nj->ni", self._interaction, 1.0 - exp_bp
            )
            tau = tau_trial - np.einsum("ij,nj->ni", plastic_modulus, dg)
            da = (dg - self.parameters.d * a0 * abs_dg) / (1.0 + self.parameters.d * abs_dg)
            drive = tau - self.parameters.c_mpa * (a0 + da)
            sgn = np.where(drive > 0.0, 1.0, -1.0)
            overstress = np.maximum(np.abs(drive) - resistance, 0.0)
            flow = slope[:, None] * overstress * sgn
            residual = dg - flow
            residual_norm = np.max(np.abs(residual), axis=1)
            converged = residual_norm <= self._tolerance
            active = (overstress > 0.0).astype(float)
            den = 1.0 + self.parameters.d * abs_dg
            num = dg - self.parameters.d * a0 * abs_dg
            dnum = 1.0 - self.parameters.d * a0 * sign_dg
            dden = self.parameters.d * sign_dg
            dda = (dnum * den - num * dden) / (den * den)
            jac = np.broadcast_to(eye12, (n, 12, 12)).copy()
            jac += (active * slope[:, None])[:, :, None] * plastic_modulus
            dr = (
                self.parameters.q_mpa
                * self.parameters.b
                * self._interaction[None, :, :]
                * exp_bp[:, None, :]
                * sign_dg[:, None, :]
            )
            jac += (active * slope[:, None] * sgn)[:, :, None] * dr
            indices = np.arange(12)
            jac[:, indices, indices] += active * slope[:, None] * self.parameters.c_mpa * dda
            if np.all(converged):
                break
            pending = np.flatnonzero(~converged)
            try:
                delta = np.linalg.solve(jac[pending], -residual[pending, :, None])[..., 0]
            except np.linalg.LinAlgError as error:
                raise ConstitutiveIntegrationError(
                    "NumPy SRIX reduced Newton is singular"
                ) from error
            if not np.isfinite(delta).all():
                raise ConstitutiveIntegrationError(
                    "NumPy SRIX reduced Newton produced non-finite values"
                )
            current_norm = residual_norm[pending]
            alpha = np.ones(pending.size)
            accepted = np.zeros(pending.size, dtype=bool)
            best_norm = np.full(pending.size, np.inf)
            best_dg = dg[pending].copy()
            for _backtrack in range(12):
                candidate_dg = dg[pending] + alpha[:, None] * delta
                candidate = self._reduced_residual(
                    candidate_dg,
                    tau_trial[pending],
                    deq[pending],
                    p0[pending],
                    a0[pending],
                )
                candidate_norm = np.max(np.abs(candidate), axis=1)
                finite = np.isfinite(candidate_norm)
                better = finite & (candidate_norm < best_norm)
                best_norm = np.where(better, candidate_norm, best_norm)
                best_dg = np.where(better[:, None], candidate_dg, best_dg)
                accepted |= finite & (candidate_norm <= current_norm)
                alpha = np.where(accepted, alpha, 0.5 * alpha)
                if np.all(accepted):
                    break
            if not np.all(accepted):
                failed = np.flatnonzero(~accepted)
                worst_local = int(failed[np.argmax(current_norm[failed])])
                worst = int(pending[worst_local])
                raise ConstitutiveIntegrationError(
                    "NumPy SRIX reduced Newton backtracking failed "
                    f"(iteration={iteration + 1}, point={worst}, "
                    f"current_norm={current_norm[worst_local]:.6e}, "
                    f"best_norm={best_norm[worst_local]:.6e})"
                )
            dg[pending] = best_dg
        if not np.all(converged):
            raise ConstitutiveIntegrationError(
                "NumPy SRIX reduced Newton did not converge in "
                f"{self._maximum_iterations} iterations"
            )
        deel = material_strain - np.einsum("si,ns->ni", mus, dg)
        elastic_material = elastic0 + deel
        stress_material = np.einsum("ij,nj->ni", ce, elastic_material)
        stress_global = np.einsum("nij,nj->ni", np.swapaxes(transform, 1, 2), stress_material)
        elastic_global = np.einsum("nij,nj->ni", np.swapaxes(transform, 1, 2), elastic_material)
        # Rebuild the proven 18x18 implicit Jacobian only for the consistent
        # tangent.  The nonlinear solve above uses the exact 12-slip Schur
        # reduction; retaining this oracle construction avoids changing the
        # tangent contract while the reduced formulation is qualified.
        elastic = elastic0 + deel
        stress = np.einsum("ij,nj->ni", ce, elastic)
        tau = np.einsum("si,ni->ns", mus, stress)
        de = _deviatoric(deel) + np.einsum("si,ns->ni", mus, dg)
        deq = np.sqrt(np.maximum(2.0 * np.sum(de * de, axis=1) / 3.0, 0.0))
        abs_dg = np.abs(dg)
        sign_dg = np.where(dg > 0.0, 1.0, np.where(dg < 0.0, -1.0, 0.0))
        p_trial = p0 + abs_dg
        exp_bp = np.exp(-self.parameters.b * p_trial)
        resistance = self.parameters.tau0_mpa + self.parameters.q_mpa * np.einsum(
            "ij,nj->ni", self._interaction, 1.0 - exp_bp
        )
        da = (dg - self.parameters.d * a0 * abs_dg) / (1.0 + self.parameters.d * abs_dg)
        drive = tau - self.parameters.c_mpa * (a0 + da)
        sgn = np.where(drive > 0.0, 1.0, -1.0)
        overstress = np.maximum(np.abs(drive) - resistance, 0.0)
        slope = deq / self.parameters.overstress_modulus_mpa
        active = (overstress > 0.0).astype(float)
        ndeq = np.zeros_like(de)
        nonzero = deq > 1e-14
        ndeq[nonzero] = (2.0 / (3.0 * deq[nonzero, None])) * de[nonzero]
        mus_ce = np.einsum("si,ij->sj", mus, ce)
        jfd = -active[:, :, None] * slope[:, None, None] * mus_ce[None, :, :]
        jfd -= (overstress * sgn / self.parameters.overstress_modulus_mpa)[:, :, None] * ndeq[
            :, None, :
        ]
        jgg = np.broadcast_to(np.eye(12), (n, 12, 12)).copy()
        den = 1.0 + self.parameters.d * abs_dg
        num = dg - self.parameters.d * a0 * abs_dg
        dnum = 1.0 - self.parameters.d * a0 * sign_dg
        dden = self.parameters.d * sign_dg
        dda = (dnum * den - num * dden) / (den * den)
        indices = np.arange(12)
        jgg[:, indices, indices] += active * slope[:, None] * self.parameters.c_mpa * dda
        dr = (
            self.parameters.q_mpa
            * self.parameters.b
            * self._interaction[None, :, :]
            * exp_bp[:, None, :]
            * sign_dg[:, None, :]
        )
        jgg += (active * slope[:, None] * sgn)[:, :, None] * dr
        ndeq_projection = np.einsum("ni,si->ns", ndeq, mus)
        jgg -= (overstress * sgn / self.parameters.overstress_modulus_mpa)[
            :, :, None
        ] * ndeq_projection[:, None, :]
        full_jac = np.zeros((n, 18, 18))
        full_jac[:, :6, :6] = np.eye(6)
        full_jac[:, :6, 6:] = np.swapaxes(mus[None, :, :], 1, 2)
        full_jac[:, 6:, :6] = jfd
        full_jac[:, 6:, 6:] = jgg
        rhs = np.zeros((n, 18, 6))
        rhs[:, :6, :] = np.eye(6)
        implicit = np.linalg.solve(full_jac, rhs)
        tangent_material = np.einsum("ij,njk->nik", ce, implicit[:, :6, :])
        tangent_global = np.einsum(
            "nij,njk,nkl->nil", np.swapaxes(transform, 1, 2), tangent_material, transform
        )
        p_final = p0 + np.abs(dg)
        da_final = (dg - self.parameters.d * a0 * np.abs(dg)) / (
            1.0 + self.parameters.d * np.abs(dg)
        )
        return SrixNumpy3DTrial(
            total_strain_kelvin=total_strain.copy(),
            stress_kelvin_mpa=stress_global,
            elastic_strain_kelvin=elastic_global,
            plastic_slip=self._g[start:stop] + dg,
            equivalent_plastic_slip=p_final,
            back_strain=a0 + da_final,
            accumulated_slip=np.sum(p_final, axis=1),
            consistent_tangent_kelvin_mpa=tangent_global,
            material_elastic_strain_kelvin=elastic_material,
        )

    def evaluate(
        self, total_strain_kelvin: ArrayLike, *, time_increment: float
    ) -> SrixNumpy3DTrial:
        if not np.isfinite(time_increment) or time_increment <= 0:
            raise ValueError("time_increment must be finite and positive")
        values = np.asarray(total_strain_kelvin, dtype=float)
        if values.shape != (self.point_count, 6) or not np.isfinite(values).all():
            raise ValueError(
                f"total_strain_kelvin must have shape {(self.point_count, 6)} and be finite"
            )
        self.revert()
        strain_increment = values - self._committed_total_strain
        # Persistent state is never replaced by a trial until the whole call
        # has succeeded. Chunking limits Newton workspaces, not state.
        if self._batch_size is None or self._batch_size >= self.point_count:
            result = self._integrate_chunk(strain_increment, values)
        else:
            chunks = []
            for start in range(0, self.point_count, self._batch_size):
                stop = min(start + self._batch_size, self.point_count)
                chunks.append(
                    self._integrate_chunk(strain_increment[start:stop], values[start:stop], start)
                )
            result = SrixNumpy3DTrial(
                total_strain_kelvin=np.concatenate([item.total_strain_kelvin for item in chunks]),
                stress_kelvin_mpa=np.concatenate([item.stress_kelvin_mpa for item in chunks]),
                elastic_strain_kelvin=np.concatenate(
                    [item.elastic_strain_kelvin for item in chunks]
                ),
                plastic_slip=np.concatenate([item.plastic_slip for item in chunks]),
                equivalent_plastic_slip=np.concatenate(
                    [item.equivalent_plastic_slip for item in chunks]
                ),
                back_strain=np.concatenate([item.back_strain for item in chunks]),
                accumulated_slip=np.concatenate([item.accumulated_slip for item in chunks]),
                consistent_tangent_kelvin_mpa=np.concatenate(
                    [item.consistent_tangent_kelvin_mpa for item in chunks]
                ),
                material_elastic_strain_kelvin=np.concatenate(
                    [item.material_elastic_strain_kelvin for item in chunks]
                ),
            )
        self._trial = result
        self._trial_state = (
            result.material_elastic_strain_kelvin.copy(),
            result.plastic_slip.copy(),
            result.equivalent_plastic_slip.copy(),
            result.back_strain.copy(),
        )
        self._trial_total_strain = values.copy()
        return result

    def commit(self) -> None:
        if self._trial_state is None:
            raise RuntimeError("no successful NumPy SRIX trial to commit")
        self._elastic, self._g, self._p, self._a = tuple(
            value.copy() for value in self._trial_state
        )
        assert self._trial_total_strain is not None
        self._committed_total_strain = self._trial_total_strain.copy()
        self._trial = None
        self._trial_state = None
        self._trial_total_strain = None

    def revert(self) -> None:
        self._trial = None
        self._trial_state = None
        self._trial_total_strain = None


class SrixNumpyCondensedPlaneStressBatch:
    """Three-equation global-frame plane-stress closure around NumPy SRIX."""

    def __init__(
        self,
        bridge: SrixNumpy3DMaterialPointBatch,
        *,
        local_tolerance_mpa: float = 1e-8,
        maximum_local_iterations: int = 15,
        plane_stress_max_iterations: int | None = None,
    ) -> None:
        self._bridge = bridge
        self._tol = float(local_tolerance_mpa)
        self._max = int(
            maximum_local_iterations
            if plane_stress_max_iterations is None
            else plane_stress_max_iterations
        )
        self._committed_transverse = np.zeros((bridge.point_count, 3))
        self._latest_transverse: FloatArray | None = None
        self._latest: ConstitutiveTrial | None = None

    @property
    def point_count(self) -> int:
        return self._bridge.point_count

    @property
    def backend_name(self) -> str:
        return "numpy-srix-condensed-plane-stress"

    @property
    def completion_strategy(self) -> str:
        return "numpy_srix_3d_local_condensation"

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        return "nonsymmetric"

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return PlaneStressBatchStatistics()

    def evaluate(
        self, in_plane_strain: ArrayLike, *, time_increment: float, consistent_tangent: bool = True
    ) -> ConstitutiveTrial:
        in_plane = np.asarray(in_plane_strain, dtype=float)
        if in_plane.shape != (self.point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self.point_count, 3)}")
        total = np.zeros((self.point_count, 6))
        total[:, _PLANE] = in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        total[:, _TRANSVERSE] = self._committed_transverse
        for _ in range(self._max):
            trial = self._bridge.evaluate(total, time_increment=time_increment)
            stress = trial.stress_kelvin_mpa[:, _TRANSVERSE]
            if float(np.max(np.abs(stress))) <= self._tol:
                tangent = trial.consistent_tangent_kelvin_mpa
                caa = tangent[:, _PLANE][:, :, _PLANE]
                cab = tangent[:, _PLANE][:, :, _TRANSVERSE]
                cba = tangent[:, _TRANSVERSE][:, :, _PLANE]
                cbb = tangent[:, _TRANSVERSE][:, :, _TRANSVERSE]
                # One final correction is unnecessary when already converged;
                # only retain the condensed operator below.
                x = np.linalg.solve(cbb, cba)
                cps = caa - np.einsum("nij,njk->nik", cab, x)
                scale = (
                    _KELVIN_TO_ENGINEERING_STRESS_SCALE[None, :, None]
                    * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, None, :]
                )
                full_stress = kelvin_3d_to_tensor(trial.stress_kelvin_mpa, quantity="stress")
                full_strain = kelvin_3d_to_tensor(total, quantity="strain")
                elastic = kelvin_3d_to_tensor(trial.elastic_strain_kelvin, quantity="strain")
                complete = ConstitutiveTrial(
                    stress_in_plane_mpa=tensor_to_engineering_stress_2d(full_stress),
                    tangent_in_plane_mpa=cps * scale if consistent_tangent else None,
                    full_stress_tensor_mpa=full_stress,
                    full_strain_tensor=full_strain,
                    elastic_strain_tensor=elastic,
                    plastic_strain_tensor=full_strain - elastic,
                    plane_stress_residual_mpa=np.stack(
                        (full_stress[:, 2, 2], full_stress[:, 0, 2], full_stress[:, 1, 2]), axis=-1
                    ),
                    observables={
                        "plastic_slip": trial.plastic_slip,
                        "equivalent_plastic_slip": trial.equivalent_plastic_slip,
                        "accumulated_slip": trial.accumulated_slip,
                    },
                )
                self._latest_transverse = total[:, _TRANSVERSE].copy()
                self._latest = complete
                return complete
            tangent = trial.consistent_tangent_kelvin_mpa
            cbb = tangent[:, _TRANSVERSE][:, :, _TRANSVERSE]
            total[:, _TRANSVERSE] += np.linalg.solve(cbb, -stress[..., None])[..., 0]
        self._bridge.revert()
        raise ConstitutiveIntegrationError("NumPy SRIX plane-stress closure did not converge")

    def evaluate_in_plane(
        self, in_plane_strain: ArrayLike, *, time_increment: float, consistent_tangent: bool = True
    ) -> InPlaneConstitutiveTrial:
        trial = self.evaluate(
            in_plane_strain, time_increment=time_increment, consistent_tangent=consistent_tangent
        )
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=trial.stress_in_plane_mpa,
            tangent_in_plane_mpa=trial.tangent_in_plane_mpa,
            observables=trial.observables,
        )

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        if self._latest is None:
            raise RuntimeError("no NumPy SRIX trial is available")
        return self._latest

    def commit(self) -> None:
        if self._latest_transverse is None:
            raise RuntimeError("no successful NumPy SRIX trial to commit")
        self._bridge.commit()
        self._committed_transverse = self._latest_transverse.copy()
        self._latest_transverse = None
        self._latest = None

    def revert(self) -> None:
        self._bridge.revert()
        self._latest_transverse = None
        self._latest = None
