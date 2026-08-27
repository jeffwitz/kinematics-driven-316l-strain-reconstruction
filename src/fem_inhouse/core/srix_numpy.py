"""Vectorised NumPy implementation of the qualified Forest--Rubin SRIX law.

The module deliberately contains no MGIS/MFront dependency.  It is a small,
transactional material-point backend used as an independent implementation and
as the future NumPy/CuPy seam.  The local Newton system follows
``validation/mfront/Fcc316LForestRubinSrixGeneric3D.mfront`` line for line;
MFront remains the production default and the numerical oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal

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
    ResponseLevel,
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
    consistent_tangent_kelvin_mpa: FloatArray | None
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
        parallel_backend: Literal["serial", "dask-threads"] = "serial",
        dask_workers: int = 1,
        local_linear_solver: Literal["numpy", "numba-lu12"] = "numpy",
    ) -> None:
        if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 1:
            raise ValueError("point_count must be a positive integer")
        if material_newton_max_iterations is not None:
            maximum_local_iterations = material_newton_max_iterations
        if maximum_local_iterations < 1 or local_tolerance <= 0:
            raise ValueError("invalid local Newton controls")
        if batch_size is not None and (isinstance(batch_size, bool) or batch_size < 1):
            raise ValueError("batch_size must be positive")
        if parallel_backend not in {"serial", "dask-threads"}:
            raise ValueError("parallel_backend must be 'serial' or 'dask-threads'")
        if isinstance(dask_workers, bool) or dask_workers < 1:
            raise ValueError("dask_workers must be positive")
        if local_linear_solver not in {"numpy", "numba-lu12"}:
            raise ValueError("local_linear_solver must be 'numpy' or 'numba-lu12'")
        selected = parameters if parameters is not None else parameter_set
        self.parameters = _resolve_parameters(selected, explicit_parameters)
        self._point_count = point_count
        self._batch_size = batch_size
        self._parallel_backend = parallel_backend
        self._dask_workers = int(dask_workers)
        self._local_linear_solver = local_linear_solver
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
        self._timing = {
            "rotation_seconds": 0.0,
            "reduced_residual_seconds": 0.0,
            "reduced_jacobian_seconds": 0.0,
            "reduced_solve_seconds": 0.0,
            "backtracking_seconds": 0.0,
            "tangent_seconds": 0.0,
            "material_newton_iterations": 0,
            "active_point_solves": 0,
        }

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
    def parallel_backend(self) -> str:
        return self._parallel_backend

    @property
    def dask_workers(self) -> int:
        return self._dask_workers

    @property
    def local_linear_solver(self) -> str:
        return self._local_linear_solver

    def _solve12(self, matrix: FloatArray, rhs: FloatArray) -> FloatArray:
        """Solve batched 12x12 systems, optionally with multiple RHS."""
        if matrix.shape[1:] != (12, 12):
            if rhs.ndim == 2:
                return np.linalg.solve(matrix, rhs[..., None])[..., 0]
            return np.linalg.solve(matrix, rhs)
        if self._local_linear_solver == "numba-lu12":
            if rhs.ndim == 2:
                from fem_inhouse.core.small_linear_solvers import solve12_batch_numba

                result, success = solve12_batch_numba(matrix, rhs)
            else:
                # Batched LAPACK is faster than the hand-written kernel for
                # the 3/6-RHS tangent workloads on the qualified CPU.
                return np.linalg.solve(matrix, rhs)
            if not np.all(success):
                raise np.linalg.LinAlgError("Numba LU12 detected a singular system")
            return result
        if rhs.ndim == 2:
            return np.linalg.solve(matrix, rhs[..., None])[..., 0]
        return np.linalg.solve(matrix, rhs)

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

    @property
    def timing_statistics(self) -> dict[str, Any]:
        return dict(self._timing)

    def _integrate_chunk_full(
        self,
        strain_increment: FloatArray,
        total_strain: FloatArray,
        start: int = 0,
    ) -> SrixNumpy3DTrial:
        n = strain_increment.shape[0]
        stop = start + n
        transform = self._kelvin_rotation[start:stop]
        started = perf_counter()
        material_strain = np.einsum("nij,nj->ni", transform, strain_increment)
        self._timing["rotation_seconds"] += perf_counter() - started
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
                delta = self._solve12(jac, -residual)
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
        tangent_mode: Literal["none", "transverse", "full"] = "full",
    ) -> SrixNumpy3DTrial:
        """Integrate one chunk with the exact 12-slip Schur reduction."""
        n = strain_increment.shape[0]
        stop = start + n
        transform = self._kelvin_rotation[start:stop]
        started = perf_counter()
        material_strain = np.einsum("nij,nj->ni", transform, strain_increment)
        self._timing["rotation_seconds"] += perf_counter() - started
        p0, a0 = self._p[start:stop], self._a[start:stop]
        elastic0 = self._elastic[start:stop]
        ce = self._ce_material
        mus = self._schmid_material
        plastic_modulus = self._plastic_modulus
        dg = np.zeros((n, 12))
        tau_trial = (elastic0 + material_strain) @ self._mce.T
        de = _deviatoric(material_strain)
        deq = np.sqrt(np.maximum(2.0 * np.sum(de * de, axis=1) / 3.0, 0.0))
        slope = deq / self.parameters.overstress_modulus_mpa
        eye12 = np.eye(12)
        converged = np.zeros(n, dtype=bool)
        pending = np.arange(n)
        for iteration in range(self._maximum_iterations):
            if pending.size == 0:
                break
            active_indices = pending
            residual_started = perf_counter()
            dg_active = dg[active_indices]
            abs_dg = np.abs(dg_active)
            sign_dg = np.where(dg_active > 0.0, 1.0, np.where(dg_active < 0.0, -1.0, 0.0))
            p_trial = p0[active_indices] + abs_dg
            exp_bp = np.exp(-self.parameters.b * p_trial)
            resistance = self.parameters.tau0_mpa + self.parameters.q_mpa * (
                (1.0 - exp_bp) @ self._interaction.T
            )
            tau = tau_trial[active_indices] - dg_active @ plastic_modulus.T
            da = (dg_active - self.parameters.d * a0[active_indices] * abs_dg) / (
                1.0 + self.parameters.d * abs_dg
            )
            drive = tau - self.parameters.c_mpa * (a0[active_indices] + da)
            sgn = np.where(drive > 0.0, 1.0, -1.0)
            overstress = np.maximum(np.abs(drive) - resistance, 0.0)
            flow = slope[active_indices, None] * overstress * sgn
            residual = dg_active - flow
            residual_norm = np.max(np.abs(residual), axis=1)
            newly_converged = residual_norm <= self._tolerance
            converged[active_indices[newly_converged]] = True
            pending = active_indices[~newly_converged]
            self._timing["reduced_residual_seconds"] += perf_counter() - residual_started
            self._timing["material_newton_iterations"] += int(active_indices.size)
            if pending.size == 0:
                break
            self._timing["active_point_solves"] += int(pending.size)
            jacobian_started = perf_counter()
            still_pending = ~newly_converged
            active_pending = (overstress[still_pending] > 0.0).astype(float)
            den = 1.0 + self.parameters.d * abs_dg[still_pending]
            num = dg_active[still_pending] - self.parameters.d * a0[pending] * abs_dg[still_pending]
            dnum = 1.0 - self.parameters.d * a0[pending] * sign_dg[still_pending]
            dden = self.parameters.d * sign_dg[still_pending]
            dda = (dnum * den - num * dden) / (den * den)
            solve_started = perf_counter()
            try:
                rhs_pending = -residual[still_pending]
                if self._local_linear_solver == "numba-lu12":
                    from fem_inhouse.core.small_linear_solvers import solve12_jacobian_batch_numba

                    delta, success = solve12_jacobian_batch_numba(
                        slope[pending],
                        active_pending,
                        sgn[still_pending],
                        exp_bp[still_pending],
                        sign_dg[still_pending],
                        dda,
                        residual[still_pending],
                        plastic_modulus,
                        self._interaction,
                        self.parameters.q_mpa,
                        self.parameters.b,
                        self.parameters.c_mpa,
                    )
                    if not np.all(success):
                        raise np.linalg.LinAlgError("Numba fused LU12 detected a singular system")
                else:
                    jac_pending = np.broadcast_to(eye12, (pending.size, 12, 12)).copy()
                    jac_pending += (
                        active_pending * slope[pending, None]
                    )[:, :, None] * plastic_modulus
                    dr = (
                        self.parameters.q_mpa
                        * self.parameters.b
                        * self._interaction[None, :, :]
                        * exp_bp[still_pending, None, :]
                        * sign_dg[still_pending, None, :]
                    )
                    jac_pending += (
                        active_pending * slope[pending, None] * sgn[still_pending]
                    )[:, :, None] * dr
                    indices = np.arange(12)
                    jac_pending[:, indices, indices] += (
                        active_pending * slope[pending, None] * self.parameters.c_mpa * dda
                    )
                    delta = self._solve12(jac_pending, rhs_pending)
            except np.linalg.LinAlgError as error:
                raise ConstitutiveIntegrationError(
                    "NumPy SRIX reduced Newton is singular"
                ) from error
            if not np.isfinite(delta).all():
                raise ConstitutiveIntegrationError(
                    "NumPy SRIX reduced Newton produced non-finite values"
                )
            if self._local_linear_solver != "numba-lu12":
                self._timing["reduced_jacobian_seconds"] += perf_counter() - jacobian_started
            self._timing["reduced_solve_seconds"] += perf_counter() - solve_started
            current_norm = residual_norm[still_pending]
            alpha = np.ones(pending.size)
            accepted = np.zeros(pending.size, dtype=bool)
            best_norm = np.full(pending.size, np.inf)
            best_dg = dg[pending].copy()
            backtracking_started = perf_counter()
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
            self._timing["backtracking_seconds"] += perf_counter() - backtracking_started
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
        # Reconstruct the converged constitutive quantities once for stress and
        # (when requested) the sensitivity.  Newton work above only touched
        # points that were still active.
        abs_dg = np.abs(dg)
        sign_dg = np.where(dg > 0.0, 1.0, np.where(dg < 0.0, -1.0, 0.0))
        exp_bp = np.exp(-self.parameters.b * (p0 + abs_dg))
        tau = tau_trial - dg @ plastic_modulus.T
        da = (dg - self.parameters.d * a0 * abs_dg) / (1.0 + self.parameters.d * abs_dg)
        resistance = self.parameters.tau0_mpa + self.parameters.q_mpa * (
            (1.0 - exp_bp) @ self._interaction.T
        )
        drive = tau - self.parameters.c_mpa * (a0 + da)
        sgn = np.where(drive > 0.0, 1.0, -1.0)
        overstress = np.maximum(np.abs(drive) - resistance, 0.0)
        deel = material_strain - dg @ mus
        elastic_material = elastic0 + deel
        stress_material = elastic_material @ ce.T
        stress_global = np.matmul(np.swapaxes(transform, 1, 2), stress_material[..., None])[:, :, 0]
        elastic_global = np.matmul(np.swapaxes(transform, 1, 2), elastic_material[..., None])[
            :, :, 0
        ]
        if tangent_mode == "none":
            tangent_global = None
        else:
            # Rebuild one full reduced Jacobian only for the requested
            # sensitivity. Newton iterations assemble it only for active points.
            tangent_started = perf_counter()
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
            ndeq = np.zeros_like(de)
            nonzero = deq > 1e-14
            ndeq[nonzero] = (2.0 / (3.0 * deq[nonzero, None])) * de[nonzero]
            jfd = -active[:, :, None] * slope[:, None, None] * self._mce[None, :, :]
            jfd -= (overstress * sgn / self.parameters.overstress_modulus_mpa)[:, :, None] * ndeq[
                :, None, :
            ]
            if tangent_mode == "transverse":
                transverse = transform[:, :, _TRANSVERSE]
                dgamma_deps = self._solve12(jac, -np.matmul(jfd, transverse))
                elastic_derivative = transverse - np.einsum(
                    "si,nsj->nij", mus, dgamma_deps
                )
                tangent_material = np.einsum("ij,njk->nik", ce, elastic_derivative)
                tangent_global = np.matmul(np.swapaxes(transform, 1, 2), tangent_material)
            else:
                dgamma_deps = self._solve12(jac, -jfd)
                elastic_derivative = np.eye(6)[None, :, :] - np.einsum(
                    "si,nsj->nij", mus, dgamma_deps
                )
                tangent_material = np.einsum("ij,njk->nik", ce, elastic_derivative)
                tangent_global = np.einsum(
                    "nij,njk,nkl->nil", np.swapaxes(transform, 1, 2), tangent_material, transform
                )
            self._timing["tangent_seconds"] += perf_counter() - tangent_started
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

    def tangent_from_trial(
        self,
        total_strain_kelvin: FloatArray,
        trial: SrixNumpy3DTrial,
        *,
        tangent_mode: Literal["transverse", "full"] = "full",
    ) -> FloatArray:
        """Build a requested tangent from a converged trial without re-integrating."""
        if tangent_mode not in {"transverse", "full"}:
            raise ValueError("tangent_mode must be 'transverse' or 'full'")
        started = perf_counter()
        total = np.asarray(total_strain_kelvin, dtype=float)
        if total.shape != (self.point_count, 6):
            raise ValueError(f"total_strain_kelvin must have shape {(self.point_count, 6)}")
        transform = self._kelvin_rotation
        mus = self._schmid_material
        ce = self._ce_material
        elastic0 = self._elastic
        deel = trial.material_elastic_strain_kelvin - elastic0
        dg = trial.plastic_slip - self._g
        de = _deviatoric(deel) + np.einsum("si,ns->ni", mus, dg)
        deq = np.sqrt(np.maximum(2.0 * np.sum(de * de, axis=1) / 3.0, 0.0))
        abs_dg = np.abs(dg)
        sign_dg = np.where(dg > 0.0, 1.0, np.where(dg < 0.0, -1.0, 0.0))
        exp_bp = np.exp(-self.parameters.b * (self._p + abs_dg))
        tau = (elastic0 + deel) @ self._mce.T
        da = (dg - self.parameters.d * self._a * abs_dg) / (1.0 + self.parameters.d * abs_dg)
        resistance = self.parameters.tau0_mpa + self.parameters.q_mpa * np.einsum(
            "ij,nj->ni", self._interaction, 1.0 - exp_bp
        )
        drive = tau - self.parameters.c_mpa * (self._a + da)
        sgn = np.where(drive > 0.0, 1.0, -1.0)
        overstress = np.maximum(np.abs(drive) - resistance, 0.0)
        slope = deq / self.parameters.overstress_modulus_mpa
        active = (overstress > 0.0).astype(float)
        ndeq = np.zeros_like(de)
        nonzero = deq > 1e-14
        ndeq[nonzero] = (2.0 / (3.0 * deq[nonzero, None])) * de[nonzero]
        jfd = -active[:, :, None] * slope[:, None, None] * self._mce[None, :, :]
        jfd -= (overstress * sgn / self.parameters.overstress_modulus_mpa)[:, :, None] * ndeq[
            :, None, :
        ]
        den = 1.0 + self.parameters.d * abs_dg
        num = dg - self.parameters.d * self._a * abs_dg
        dnum = 1.0 - self.parameters.d * self._a * sign_dg
        dden = self.parameters.d * sign_dg
        dda = (dnum * den - num * dden) / (den * den)
        jac = np.broadcast_to(np.eye(12), (self.point_count, 12, 12)).copy()
        jac += (active * slope[:, None])[:, :, None] * self._plastic_modulus
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
        if tangent_mode == "transverse":
            rhs = -np.matmul(jfd, transform[:, :, _TRANSVERSE])
            dgamma_deps = self._solve12(jac, rhs)
            elastic_derivative = transform[:, :, _TRANSVERSE] - np.einsum(
                "si,nsj->nij", mus, dgamma_deps
            )
            tangent_material = np.einsum("ij,njk->nik", ce, elastic_derivative)
            result = np.matmul(np.swapaxes(transform, 1, 2), tangent_material)
        else:
            dgamma_deps = self._solve12(jac, -jfd)
            elastic_derivative = np.eye(6)[None, :, :] - np.einsum(
                "si,nsj->nij", mus, dgamma_deps
            )
            tangent_material = np.einsum("ij,njk->nik", ce, elastic_derivative)
            result = np.einsum(
                "nij,njk,nkl->nil", np.swapaxes(transform, 1, 2), tangent_material, transform
            )
        self._timing["tangent_seconds"] += perf_counter() - started
        return result

    def evaluate(
        self,
        total_strain_kelvin: ArrayLike,
        *,
        time_increment: float,
        tangent_mode: Literal["none", "transverse", "full"] = "full",
    ) -> SrixNumpy3DTrial:
        if not np.isfinite(time_increment) or time_increment <= 0:
            raise ValueError("time_increment must be finite and positive")
        if tangent_mode not in {"none", "transverse", "full"}:
            raise ValueError("tangent_mode must be 'none', 'transverse', or 'full'")
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
            result = self._integrate_chunk(strain_increment, values, tangent_mode=tangent_mode)
        else:
            starts = range(0, self.point_count, self._batch_size)
            if self._parallel_backend == "dask-threads":
                try:
                    from dask import compute, delayed
                except ImportError as error:
                    raise ImportError(
                        "parallel_backend='dask-threads' requires the optional dask dependency"
                    ) from error
                tasks = [
                    delayed(self._integrate_chunk)(
                        strain_increment[start : min(start + self._batch_size, self.point_count)],
                        values[start : min(start + self._batch_size, self.point_count)],
                        start,
                        tangent_mode,
                    )
                    for start in starts
                ]
                chunks = list(
                    compute(
                        *tasks,
                        scheduler="threads",
                        num_workers=self._dask_workers,
                    )
                )
            else:
                chunks = [
                    self._integrate_chunk(
                        strain_increment[start : min(start + self._batch_size, self.point_count)],
                        values[start : min(start + self._batch_size, self.point_count)],
                        start,
                        tangent_mode,
                    )
                    for start in starts
                ]
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
                consistent_tangent_kelvin_mpa=(
                    None
                    if tangent_mode == "none"
                    else np.concatenate(
                        [item.consistent_tangent_kelvin_mpa for item in chunks]
                    )
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
        local_transverse_predictor: Literal["committed", "tangent"] = "committed",
        plane_stress_solver: Literal["nested", "coupled"] = "nested",
    ) -> None:
        self._bridge = bridge
        self._tol = float(local_tolerance_mpa)
        self._max = int(
            maximum_local_iterations
            if plane_stress_max_iterations is None
            else plane_stress_max_iterations
        )
        if local_transverse_predictor not in {"committed", "tangent"}:
            raise ValueError("local_transverse_predictor must be 'committed' or 'tangent'")
        if plane_stress_solver not in {"nested", "coupled"}:
            raise ValueError("plane_stress_solver must be 'nested' or 'coupled'")
        self._local_transverse_predictor = local_transverse_predictor
        self._plane_stress_solver = plane_stress_solver
        self._committed_transverse = np.zeros((bridge.point_count, 3))
        self._latest_transverse: FloatArray | None = None
        self._latest: ConstitutiveTrial | None = None
        self._accepted_transverse = self._committed_transverse.copy()
        self._accepted_in_plane: FloatArray | None = None
        self._accepted_cbb: FloatArray | None = None
        self._accepted_cba: FloatArray | None = None
        self._latest_in_plane: FloatArray | None = None
        self._latest_cbb: FloatArray | None = None
        self._latest_cba: FloatArray | None = None
        self._plane_stress_seconds = 0.0
        self._plane_stress_iterations = 0
        self._srix_iterations_per_plane_stress: list[float] = []

    @property
    def point_count(self) -> int:
        return self._bridge.point_count

    @property
    def backend_name(self) -> str:
        return "numpy-srix-condensed-plane-stress"

    @property
    def local_transverse_predictor(self) -> str:
        return self._local_transverse_predictor

    @property
    def plane_stress_solver(self) -> str:
        return self._plane_stress_solver

    @property
    def timing_statistics(self) -> dict[str, Any]:
        values = self._bridge.timing_statistics
        values["plane_stress_seconds"] = self._plane_stress_seconds
        values["plane_stress_iterations"] = self._plane_stress_iterations
        if self._srix_iterations_per_plane_stress:
            samples = np.asarray(self._srix_iterations_per_plane_stress)
            values["srix_equivalent_iterations_total"] = float(np.sum(samples))
            values["srix_equivalent_iterations_mean_per_plane_stress"] = float(np.mean(samples))
            values["srix_equivalent_iterations_median_per_plane_stress"] = float(np.median(samples))
            values["srix_equivalent_iterations_p95_per_plane_stress"] = float(
                np.percentile(samples, 95.0)
            )
            values["srix_equivalent_iterations_per_plane_stress"] = samples.tolist()
        return values

    @property
    def completion_strategy(self) -> str:
        return "numpy_srix_3d_local_condensation"

    @property
    def linear_system_matrix_type(self) -> LinearSystemMatrixType:
        return "nonsymmetric"

    @property
    def statistics(self) -> PlaneStressBatchStatistics:
        return PlaneStressBatchStatistics()

    def _solve3(self, matrix: FloatArray, rhs: FloatArray) -> FloatArray:
        """Solve batched 3x3 systems using the configured local accelerator."""
        if self._bridge.local_linear_solver == "numba-lu12":
            if rhs.ndim == 2:
                # The small 3x3 single-RHS calls are below the crossover
                # point for the compiled kernel; use LAPACK here.
                return np.linalg.solve(matrix, rhs[..., None])[..., 0]
            else:
                from fem_inhouse.core.small_linear_solvers import solve3_batch_rhs_numba

                solution, success = solve3_batch_rhs_numba(matrix, rhs)
                result = solution
            if not np.all(success):
                raise np.linalg.LinAlgError("Numba LU3 detected a singular system")
            return result
        if rhs.ndim == 2:
            return np.linalg.solve(matrix, rhs[..., None])[..., 0]
        return np.linalg.solve(matrix, rhs)

    def _evaluate_coupled(
        self,
        total: FloatArray,
        in_plane: FloatArray,
        *,
        time_increment: float,
        response_level: ResponseLevel,
        consistent_tangent: bool,
    ) -> ConstitutiveTrial | InPlaneConstitutiveTrial:
        """Solve SRIX and the three plane-stress equations in one local Newton."""
        bridge = self._bridge
        bridge.revert()
        n = self.point_count
        transform = bridge._kelvin_rotation
        strain_increment = total - bridge._committed_total_strain
        material_strain = np.einsum("nij,nj->ni", transform, strain_increment)
        p0, a0 = bridge._p, bridge._a
        elastic0 = bridge._elastic
        ce = bridge._ce_material
        mus = bridge._schmid_material
        eye12 = np.eye(12)
        dg = np.zeros((n, 12))
        converged = np.zeros(n, dtype=bool)

        def state(values: FloatArray, slips: FloatArray, indices: NDArray[np.int64] | None = None):
            p_base = p0 if indices is None else p0[indices]
            a_base = a0 if indices is None else a0[indices]
            e_base = elastic0 if indices is None else elastic0[indices]
            t_base = transform if indices is None else transform[indices]
            de = _deviatoric(values)
            deq = np.sqrt(np.maximum(2.0 * np.sum(de * de, axis=1) / 3.0, 0.0))
            slope = deq / bridge.parameters.overstress_modulus_mpa
            abs_dg = np.abs(slips)
            sign_dg = np.where(slips > 0.0, 1.0, np.where(slips < 0.0, -1.0, 0.0))
            exp_bp = np.exp(-bridge.parameters.b * (p_base + abs_dg))
            resistance = bridge.parameters.tau0_mpa + bridge.parameters.q_mpa * (
                (1.0 - exp_bp) @ bridge._interaction.T
            )
            tau_trial = (e_base + values) @ bridge._mce.T
            tau = tau_trial - slips @ bridge._plastic_modulus.T
            da = (slips - bridge.parameters.d * a_base * abs_dg) / (
                1.0 + bridge.parameters.d * abs_dg
            )
            drive = tau - bridge.parameters.c_mpa * (a_base + da)
            sgn = np.where(drive > 0.0, 1.0, -1.0)
            overstress = np.maximum(np.abs(drive) - resistance, 0.0)
            residual = slips - slope[:, None] * overstress * sgn
            stress_material = (e_base + values - slips @ mus) @ ce.T
            stress_global = np.einsum("nij,nj->ni", np.swapaxes(t_base, 1, 2), stress_material)
            return (
                residual,
                stress_global[:, _TRANSVERSE],
                de,
                deq,
                slope,
                abs_dg,
                sign_dg,
                exp_bp,
                sgn,
                overstress,
                stress_material,
            )

        for _iteration in range(bridge._maximum_iterations):
            current = state(material_strain, dg)
            (
                residual, stress_b, de, deq, slope, abs_dg, sign_dg,
                exp_bp, sgn, overstress, _
            ) = current
            residual_norm = np.maximum(
                np.max(np.abs(residual), axis=1),
                np.max(np.abs(stress_b), axis=1) / max(bridge.parameters.tau0_mpa, 1.0),
            )
            newly_converged = (
                np.max(np.abs(residual), axis=1) <= bridge._tolerance
            ) & (np.max(np.abs(stress_b), axis=1) <= self._tol)
            converged |= newly_converged
            pending = np.flatnonzero(~converged)
            if pending.size == 0:
                break
            active = (overstress[pending] > 0.0).astype(float)
            den = 1.0 + bridge.parameters.d * abs_dg[pending]
            num = dg[pending] - bridge.parameters.d * a0[pending] * abs_dg[pending]
            dnum = 1.0 - bridge.parameters.d * a0[pending] * sign_dg[pending]
            dden = bridge.parameters.d * sign_dg[pending]
            dda = (dnum * den - num * dden) / (den * den)
            a = np.broadcast_to(eye12, (pending.size, 12, 12)).copy()
            a += (active * slope[pending, None])[:, :, None] * bridge._plastic_modulus
            dr = (
                bridge.parameters.q_mpa
                * bridge.parameters.b
                * bridge._interaction[None]
                * exp_bp[pending, None, :]
                * sign_dg[pending, None, :]
            )
            a += (active * slope[pending, None] * sgn[pending])[:, :, None] * dr
            ii = np.arange(12)
            a[:, ii, ii] += active * slope[pending, None] * bridge.parameters.c_mpa * dda
            ndeq = np.zeros_like(de)
            nz = deq > 1.0e-14
            ndeq[nz] = (2.0 / (3.0 * deq[nz, None])) * de[nz]
            jfd = -active[:, :, None] * slope[pending, None, None] * bridge._mce[None]
            jfd -= (
                overstress[pending] * sgn[pending] / bridge.parameters.overstress_modulus_mpa
            )[:, :, None] * ndeq[pending, None, :]
            b = np.matmul(jfd, transform[pending][:, :, _TRANSVERSE])
            ce_global = np.matmul(
                np.swapaxes(transform[pending], 1, 2),
                np.matmul(ce[None], transform[pending]),
            )
            dmat = ce_global[:, _TRANSVERSE][:, :, _TRANSVERSE]
            c_base = -np.matmul(
                np.swapaxes(transform[pending], 1, 2), (ce @ mus.T)[None]
            )[:, _TRANSVERSE, :]
            inv_a_r = bridge._solve12(a, residual[pending][..., None])[:, :, 0]
            inv_a_b = bridge._solve12(a, b)
            schur = dmat - np.matmul(c_base, inv_a_b)
            rhs_b = -stress_b[pending] + np.einsum("nij,nj->ni", c_base, inv_a_r)
            delta_b = self._solve3(schur, rhs_b)
            delta_g = -inv_a_r - np.einsum("nij,nj->ni", inv_a_b, delta_b)
            current_metric = residual_norm[pending]
            alpha = np.ones(pending.size)
            accepted = np.zeros(pending.size, dtype=bool)
            best_dg = dg[pending].copy()
            best_total_b = total[pending][:, _TRANSVERSE].copy()
            best_metric = np.full(pending.size, np.inf)
            for _ in range(10):
                cand_dg = dg[pending] + alpha[:, None] * delta_g
                cand_total = total[pending].copy()
                cand_total[:, _TRANSVERSE] += alpha[:, None] * delta_b
                cand_material = np.einsum(
                    "nij,nj->ni", transform[pending],
                    cand_total - bridge._committed_total_strain[pending],
                )
                cand_r, cand_s, *_ = state(cand_material, cand_dg, pending)
                cand_metric = np.maximum(
                    np.max(np.abs(cand_r), axis=1),
                    np.max(np.abs(cand_s), axis=1) / max(bridge.parameters.tau0_mpa, 1.0),
                )
                good = np.isfinite(cand_metric) & (cand_metric < best_metric)
                best_metric = np.where(good, cand_metric, best_metric)
                best_dg = np.where(good[:, None], cand_dg, best_dg)
                best_total_b = np.where(good[:, None], cand_total[:, _TRANSVERSE], best_total_b)
                accepted |= np.isfinite(cand_metric) & (cand_metric <= current_metric)
                alpha = np.where(accepted, alpha, 0.5 * alpha)
                if np.all(accepted):
                    break
            if not np.all(accepted):
                raise ConstitutiveIntegrationError(
                    "coupled SRIX plane-stress Newton line search failed"
                )
            dg[pending] = best_dg
            for component, index in enumerate(_TRANSVERSE):
                total[pending, index] = best_total_b[:, component]
            material_strain[pending] = np.einsum(
                "nij,nj->ni", transform[pending],
                total[pending] - bridge._committed_total_strain[pending],
            )
        if not np.all(converged):
            raise ConstitutiveIntegrationError(
                "coupled SRIX plane-stress Newton did not converge in "
                f"{bridge._maximum_iterations} iterations"
            )
        final = state(material_strain, dg)
        _, stress_b, _, _, _, abs_dg, _, _, _, _, stress_material = final
        elastic_material = elastic0 + material_strain - dg @ mus
        elastic_global = np.einsum(
            "nij,nj->ni", np.swapaxes(transform, 1, 2), elastic_material
        )
        stress_global = np.einsum(
            "nij,nj->ni", np.swapaxes(transform, 1, 2), stress_material
        )
        full_stress = kelvin_3d_to_tensor(stress_global, quantity="stress")
        p_final = p0 + abs_dg
        da_final = (dg - bridge.parameters.d * a0 * abs_dg) / (
            1.0 + bridge.parameters.d * abs_dg
        )
        trial = SrixNumpy3DTrial(
            total_strain_kelvin=total.copy(),
            stress_kelvin_mpa=stress_global,
            elastic_strain_kelvin=elastic_global,
            plastic_slip=bridge._g + dg,
            equivalent_plastic_slip=p_final,
            back_strain=a0 + da_final,
            accumulated_slip=np.sum(p_final, axis=1),
            consistent_tangent_kelvin_mpa=None,
            material_elastic_strain_kelvin=elastic_material,
        )
        bridge._trial = trial
        bridge._trial_state = (
            elastic_material.copy(),
            trial.plastic_slip.copy(),
            p_final.copy(),
            trial.back_strain.copy(),
        )
        bridge._trial_total_strain = total.copy()
        stress_in_plane = np.stack(
            (stress_global[:, 0], stress_global[:, 1], stress_global[:, 3] / _SQRT_TWO), axis=-1
        )
        if response_level == "residual":
            result: ConstitutiveTrial | InPlaneConstitutiveTrial = InPlaneConstitutiveTrial(
                stress_in_plane_mpa=stress_in_plane, tangent_in_plane_mpa=None
            )
            self._latest = None
        else:
            tangent = bridge.tangent_from_trial(total, trial, tangent_mode="full")
            caa = tangent[:, _PLANE][:, :, _PLANE]
            cab = tangent[:, _PLANE][:, :, _TRANSVERSE]
            cba = tangent[:, _TRANSVERSE][:, :, _PLANE]
            cbb = tangent[:, _TRANSVERSE][:, :, _TRANSVERSE]
            cps = caa - np.einsum("nij,njk->nik", cab, self._solve3(cbb, cba))
            scale = (
                _KELVIN_TO_ENGINEERING_STRESS_SCALE[None, :, None]
                * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, None, :]
            )
            full_strain = kelvin_3d_to_tensor(total, quantity="strain")
            elastic_tensor = kelvin_3d_to_tensor(elastic_global, quantity="strain")
            complete_result = ConstitutiveTrial(
                stress_in_plane_mpa=tensor_to_engineering_stress_2d(full_stress),
                tangent_in_plane_mpa=cps * scale if consistent_tangent else None,
                full_stress_tensor_mpa=full_stress,
                full_strain_tensor=full_strain,
                elastic_strain_tensor=elastic_tensor,
                plastic_strain_tensor=full_strain - elastic_tensor,
                plane_stress_residual_mpa=np.stack(
                    (full_stress[:, 2, 2], full_stress[:, 0, 2], full_stress[:, 1, 2]),
                    axis=-1,
                ),
                observables={
                    "plastic_slip": trial.plastic_slip,
                    "equivalent_plastic_slip": trial.equivalent_plastic_slip,
                    "accumulated_slip": trial.accumulated_slip,
                },
            )
            self._latest = complete_result
            result = (
                complete_result
                if response_level == "complete"
                else InPlaneConstitutiveTrial(
                    stress_in_plane_mpa=complete_result.stress_in_plane_mpa,
                    tangent_in_plane_mpa=complete_result.tangent_in_plane_mpa,
                )
            )
            self._latest_cbb = cbb.copy()
            self._latest_cba = cba.copy()
        self._latest_transverse = total[:, _TRANSVERSE].copy()
        self._latest_in_plane = in_plane.copy()
        return result

    def _evaluate(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        response_level: ResponseLevel,
        consistent_tangent: bool = True,
    ) -> ConstitutiveTrial | InPlaneConstitutiveTrial:
        in_plane = np.asarray(in_plane_strain, dtype=float)
        if in_plane.shape != (self.point_count, 3):
            raise ValueError(f"in_plane_strain must have shape {(self.point_count, 3)}")
        if response_level not in {"residual", "tangent", "complete"}:
            raise ValueError("response_level must be 'residual', 'tangent', or 'complete'")
        total = np.zeros((self.point_count, 6))
        total[:, _PLANE] = in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        transverse_initial = self._committed_transverse
        if (
            self._local_transverse_predictor == "tangent"
            and self._accepted_in_plane is not None
            and self._accepted_cbb is not None
            and self._accepted_cba is not None
        ):
            delta_in_plane = (
                in_plane - self._accepted_in_plane
            ) * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
            try:
                correction = self._solve3(
                    self._accepted_cbb,
                    -np.einsum("nij,nj->ni", self._accepted_cba, delta_in_plane),
                )
                if np.isfinite(correction).all():
                    transverse_initial = self._accepted_transverse + correction
            except np.linalg.LinAlgError:
                pass
        total[:, _TRANSVERSE] = transverse_initial
        if self._plane_stress_solver == "coupled":
            return self._evaluate_coupled(
                total,
                in_plane,
                time_increment=time_increment,
                response_level=response_level,
                consistent_tangent=consistent_tangent,
            )
        plane_stress_started = perf_counter()
        for _iteration in range(self._max):
            self._plane_stress_iterations += 1
            material_iterations_before = self._bridge.timing_statistics[
                "material_newton_iterations"
            ]
            trial = self._bridge.evaluate(
                total,
                time_increment=time_increment,
                tangent_mode="none",
            )
            material_iterations_after = self._bridge.timing_statistics[
                "material_newton_iterations"
            ]
            self._srix_iterations_per_plane_stress.append(
                (
                    float(material_iterations_after) - float(material_iterations_before)
                )
                / self.point_count
            )
            stress = trial.stress_kelvin_mpa[:, _TRANSVERSE]
            if float(np.max(np.abs(stress))) <= self._tol:
                if response_level == "residual":
                    stress_in_plane = np.stack(
                        (
                            trial.stress_kelvin_mpa[:, 0],
                            trial.stress_kelvin_mpa[:, 1],
                            trial.stress_kelvin_mpa[:, 3] / _SQRT_TWO,
                        ),
                        axis=-1,
                    )
                    result: ConstitutiveTrial | InPlaneConstitutiveTrial = InPlaneConstitutiveTrial(
                        stress_in_plane_mpa=stress_in_plane,
                        tangent_in_plane_mpa=None,
                    )
                    self._latest_transverse = total[:, _TRANSVERSE].copy()
                    self._latest = None
                    self._latest_in_plane = in_plane.copy()
                    self._latest_cbb = None
                    self._latest_cba = None
                    self._plane_stress_seconds += perf_counter() - plane_stress_started
                    return result

                # Tangent/complete responses request a sensitivity only once,
                # after the plane-stress residual has converged.  Reuse the
                # converged constitutive state; do not integrate it again.
                tangent = self._bridge.tangent_from_trial(total, trial, tangent_mode="full")
                caa = tangent[:, _PLANE][:, :, _PLANE]
                cab = tangent[:, _PLANE][:, :, _TRANSVERSE]
                cba = tangent[:, _TRANSVERSE][:, :, _PLANE]
                cbb = tangent[:, _TRANSVERSE][:, :, _TRANSVERSE]
                x = self._solve3(cbb, cba)
                cps = caa - np.einsum("nij,njk->nik", cab, x)
                scale = (
                    _KELVIN_TO_ENGINEERING_STRESS_SCALE[None, :, None]
                    * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, None, :]
                )
                full_stress = kelvin_3d_to_tensor(trial.stress_kelvin_mpa, quantity="stress")
                full_strain = kelvin_3d_to_tensor(total, quantity="strain")
                elastic = kelvin_3d_to_tensor(trial.elastic_strain_kelvin, quantity="strain")
                stress_in_plane = tensor_to_engineering_stress_2d(full_stress)
                if response_level == "complete":
                    result = ConstitutiveTrial(
                        stress_in_plane_mpa=stress_in_plane,
                        tangent_in_plane_mpa=cps * scale if consistent_tangent else None,
                        full_stress_tensor_mpa=full_stress,
                        full_strain_tensor=full_strain,
                        elastic_strain_tensor=elastic,
                        plastic_strain_tensor=full_strain - elastic,
                        plane_stress_residual_mpa=np.stack(
                            (full_stress[:, 2, 2], full_stress[:, 0, 2], full_stress[:, 1, 2]),
                            axis=-1,
                        ),
                        observables={
                            "plastic_slip": trial.plastic_slip,
                            "equivalent_plastic_slip": trial.equivalent_plastic_slip,
                            "accumulated_slip": trial.accumulated_slip,
                        },
                    )
                else:
                    result = InPlaneConstitutiveTrial(
                        stress_in_plane_mpa=stress_in_plane,
                        tangent_in_plane_mpa=(cps * scale if response_level == "tangent" else None),
                    )
                self._latest_transverse = total[:, _TRANSVERSE].copy()
                self._latest = result if isinstance(result, ConstitutiveTrial) else None
                self._latest_in_plane = in_plane.copy()
                self._latest_cbb = cbb.copy()
                self._latest_cba = cba.copy()
                self._plane_stress_seconds += perf_counter() - plane_stress_started
                return result
            tangent = self._bridge.tangent_from_trial(total, trial, tangent_mode="transverse")
            cbb = tangent[:, _TRANSVERSE, :]
            total[:, _TRANSVERSE] += self._solve3(cbb, -stress)
        self._bridge.revert()
        self._plane_stress_seconds += perf_counter() - plane_stress_started
        raise ConstitutiveIntegrationError("NumPy SRIX plane-stress closure did not converge")

    def evaluate(
        self, in_plane_strain: ArrayLike, *, time_increment: float, consistent_tangent: bool = True
    ) -> ConstitutiveTrial:
        result = self._evaluate(
            in_plane_strain,
            time_increment=time_increment,
            response_level="complete",
            consistent_tangent=consistent_tangent,
        )
        if not isinstance(result, ConstitutiveTrial):
            raise RuntimeError("complete NumPy response unexpectedly returned a light trial")
        return result

    def evaluate_in_plane_response(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        response_level: ResponseLevel,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial | ConstitutiveTrial:
        return self._evaluate(
            in_plane_strain,
            time_increment=time_increment,
            response_level=response_level,
            consistent_tangent=consistent_tangent,
        )

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
        self.accept_global_trial()
        self._latest_transverse = None
        self._latest = None

    def accept_global_trial(self) -> None:
        """Keep the latest converged closure as predictor for the next trial."""

        if self._latest_transverse is None:
            return
        self._accepted_transverse = self._latest_transverse.copy()
        if self._latest_in_plane is not None:
            self._accepted_in_plane = self._latest_in_plane.copy()
        if self._latest_cbb is not None:
            self._accepted_cbb = self._latest_cbb.copy()
        if self._latest_cba is not None:
            self._accepted_cba = self._latest_cba.copy()

    def revert(self) -> None:
        self._bridge.revert()
        self._accepted_transverse = self._committed_transverse.copy()
        self._accepted_in_plane = None
        self._accepted_cbb = None
        self._accepted_cba = None
        self._latest_transverse = None
        self._latest = None
        self._latest_in_plane = None
        self._latest_cbb = None
        self._latest_cba = None
