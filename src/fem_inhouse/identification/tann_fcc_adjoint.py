"""Discrete trajectory adjoint of the causal TANN-FCC sequence.

Per the preregistered structure (`validation/tann_fcc_preregistration.md`,
section "Differentiation"): the forward trajectory is a chain of
equilibrium solves

    R_n(u_n; q_{n-1}, theta) = 0,     q_n = Q_n(u_n, q_{n-1}, theta),

and the gradient of the sum of per-state losses is computed by a backward
sweep over the converged records -- never by differentiating through the
Newton/GMRES iterations. The mechanical adjoint is solved matrix-free on
the interior degrees of freedom; the material VJPs are batched autograd
(see `TannFCCBatch.increment_vjp`).

Signs follow the discrete Lagrangian stationarity conditions; the FD gate
in `tests/unit/constitutive/test_tann_fcc_adjoint.py` verifies them
against central differences before any training run is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.constitutive.tann_fcc import TannFCCBatch
from fem_inhouse.identification.tann_fcc_sequence import TannFCCStateRecord
from fem_inhouse.spectral2d import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AdjointIncrementDiagnostics:
    state: int
    gmres_iterations: int
    gmres_flag: int
    material_vjp_seconds: float


class TannFCCTrajectoryAdjoint:
    """Backward sweep over a recorded forward trajectory."""

    def __init__(
        self,
        *,
        grid: StructuredGrid2D,
        material: TannFCCBatch,
        records: tuple[TannFCCStateRecord, ...],
        whitener: Callable[[FloatArray], FloatArray] | None = None,
        gmres_tolerance: float = 1.0e-10,
    ) -> None:
        self.grid = grid
        self.material = material
        self.records = records
        self.kinematics = TwoSubcellDiagnostic2D(grid)
        self.whitener = whitener
        self.gmres_tolerance = gmres_tolerance
        # The solver's divergence operator carries the per-triangle area
        # while `strain_samples` is the plain strain operator. The discrete
        # adjoint bookkeeping below needs the constant that relates them.
        self.triangle_area = (
            0.5 * (grid.length_x / grid.nx) * (grid.length_y / grid.ny)
        )

    # -- mechanical actions ---------------------------------------------------

    def _strain_of_nodal(self, nodal_field: FloatArray) -> FloatArray:
        return self.kinematics.strain_samples(nodal_field)

    def _tangent_action(
        self, interior_vector: FloatArray, tangent_mpa: FloatArray, transpose: bool
    ) -> FloatArray:
        """`B^T C B v` (transpose=False) or `B^T C^T B v` on interior vectors."""

        nodal = unpack_interior(interior_vector, self.grid)
        d_eps = self._strain_of_nodal(nodal)
        tangent = np.asarray(tangent_mpa).reshape(*self.grid.pixel_shape, 2, 3, 3)
        if transpose:
            tangent = tangent.transpose(0, 1, 2, 4, 3)
        d_sigma = np.einsum("xyqij,xyqj->xyqi", tangent, d_eps)
        return pack_interior(self.kinematics.divergence_from_sample_stress(d_sigma))

    def dot_product_test(self, record_index: int, rng: np.random.Generator) -> float:
        """`<J v, w> - <v, J^T w>`, relative, on the record's committed tangent."""

        record = self.records[record_index]
        if record.committed_tangent_mpa is None:
            raise ValueError("the record has no committed tangent")
        from scipy.sparse.linalg import aslinearoperator, LinearOperator

        size = 2 * (self.grid.nx - 1) * (self.grid.ny - 1)
        v = rng.normal(size=size)
        w = rng.normal(size=size)
        J_v = self._tangent_action(v, record.committed_tangent_mpa, transpose=False)
        Jt_w = self._tangent_action(w, record.committed_tangent_mpa, transpose=True)
        left = float(np.dot(J_v, w))
        right = float(np.dot(v, Jt_w))
        scale = max(abs(left), abs(right), 1.0)
        return abs(left - right) / scale

    # -- the sweep -------------------------------------------------------------

    def sweep(self) -> tuple[list[FloatArray], list[AdjointIncrementDiagnostics]]:
        """Gradient of the training loss w.r.t. the network parameters.

        Holdout states contribute no loss (`ell_n = 0`), but the co-state
        passes through them: their dynamics remain constrained by the
        training states after them, exactly as the preregistration intends.
        """

        import time

        from scipy.sparse.linalg import LinearOperator, gmres

        point_count = self.material.point_count
        state_dim = 1 + self.material.config.latent_dim
        v_n = np.zeros((point_count, 12, state_dim), dtype=np.float64)
        dtheta: list[FloatArray] | None = None
        diagnostics: list[AdjointIncrementDiagnostics] = []
        interior_shape = (self.grid.nx - 1, self.grid.ny - 1, 2)

        for n in range(len(self.records) - 1, -1, -1):
            record = self.records[n]
            if record.committed_tangent_mpa is None:
                raise ValueError(f"record of state {record.state} has no committed tangent")
            strain_prev = (
                np.zeros((point_count, 3), dtype=np.float64)
                if n == 0
                else self.records[n - 1].strain_in_plane_mpa
            )
            state_prev = (
                np.zeros((point_count, 12, state_dim), dtype=np.float64)
                if n == 0
                else self.records[n - 1].committed_state
            )
            strain_trial = record.strain_in_plane_mpa

            started = time.perf_counter()
            # state-channel VJPs with the incoming co-state, zero stress
            # cotangent: (dQ/du)^T v_n feeds the mechanical adjoint RHS.
            v_strain_q, v_qprev_q, dtheta_q = self.material.increment_vjp(
                strain_prev, state_prev, strain_trial, v_n,
                np.zeros((point_count, 3), dtype=np.float64),
            )

            # Mechanical adjoint, from the discrete Lagrangian stationarity:
            #
            #     J^T lam = - d ell / d u - (dQ/du)^T v_n,
            #
            # with J = D C B the solver's tangent operator (D carries the
            # per-triangle area, B is the plain strain operator), so
            # (dQ/du)^T v_n = B^T (dQ/deps)^T v_n = D(v_eps) / area.
            if record.holdout:
                rhs_field = np.zeros(interior_shape, dtype=np.float64)
            else:
                residual = (
                    record.displacement[1:-1, 1:-1]
                    - record.measured_displacement[1:-1, 1:-1]
                )
                if self.whitener is not None:
                    rhs_field = self.whitener(self.whitener(residual))
                else:
                    rhs_field = residual
            rhs = -(rhs_field.reshape(-1).copy())
            rhs = rhs - pack_interior(
                self.kinematics.divergence_from_sample_stress(
                    v_strain_q.reshape(*self.grid.pixel_shape, 2, 3)
                )
            ) / self.triangle_area

            counter = [0]
            # The solver's equilibrium residual is R = -D sigma (external
            # minus internal forces; its Newton operator is the documented
            # J v = -B^T C_alg B v), so the mechanical adjoint operator is
            # J^T = -D C^T B. Signs verified against central FD by the gate
            # in test_tann_fcc_adjoint.py, not assumed.
            operator = LinearOperator(
                shape=(rhs.size, rhs.size),
                matvec=lambda vector, tangent=record.committed_tangent_mpa: -self._tangent_action(
                    vector, tangent, transpose=True
                ),
                dtype=np.float64,
            )
            lam, info = gmres(
                operator, rhs, rtol=self.gmres_tolerance, atol=0.0,
                callback=lambda _: counter.__setitem__(0, counter[0] + 1),
                callback_type="pr_norm",
            )
            if info != 0:
                raise RuntimeError(f"mechanical adjoint GMRES failed at state {record.state}: info={info}")
            # The stress cotangent is the adjoint of the divergence:
            # w = D^T lam = area * B lam.
            w_n = (
                self.triangle_area
                * self._strain_of_nodal(unpack_interior(lam, self.grid)).reshape(
                    point_count, 3
                )
            )

            # stress-channel VJPs with the mechanical cotangent:
            # (dR/dq_prev)^T lam and (dR/dtheta)^T lam.
            _, v_qprev_s, dtheta_s = self.material.increment_vjp(
                strain_prev, state_prev, strain_trial,
                np.zeros((point_count, 12, state_dim), dtype=np.float64), w_n,
            )
            diagnostics.append(
                AdjointIncrementDiagnostics(
                    state=record.state,
                    gmres_iterations=counter[0],
                    gmres_flag=int(info),
                    material_vjp_seconds=time.perf_counter() - started,
                )
            )
            # Co-state recursion and parameter contribution, same convention:
            # v_{n-1} = (dQ/dq_prev)^T v_n + (dR/dq_prev)^T lam_n,
            # dL/dtheta += (dQ/dtheta)^T v_n + (dR/dtheta)^T lam_n.
            v_n = v_qprev_q + v_qprev_s
            if dtheta is None:
                dtheta = [q + s for q, s in zip(dtheta_q, dtheta_s, strict=True)]
            else:
                for target, q, s in zip(dtheta, dtheta_q, dtheta_s, strict=True):
                    target += q + s
        assert dtheta is not None
        diagnostics.reverse()
        return dtheta, diagnostics
