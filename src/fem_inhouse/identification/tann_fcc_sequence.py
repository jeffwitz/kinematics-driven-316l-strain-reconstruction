"""The causal TANN-FCC DIC trajectory: one masked-state sequence.

The forward path of the identification, played start to end by the
existing two-state spectral solver. The TANN supplies the constitutive
law; the solver supplies equilibrium; the interior DIC enters only the
loss of the training states -- never the material input. At a holdout
state `h` the trajectory continues from the predicted state `q_h_pred`,
never from a state recalibrated with the DIC of `h` (that is structural:
the solver commits the material increment after increment, and the DIC of
a holdout state is not touched before its solve).

The module is forward-only. Gradients follow the discrete trajectory
adjoint (see `tann_fcc_adjoint.py`), never a differentiation through the
Newton/GMRES iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.constitutive.tann_fcc import TannFCCBatch
from fem_inhouse.spectral2d import EBISpectralSolverConfig, StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TannFCCStateRecord:
    """Everything one converged increment needs for loss and adjoint."""

    state: int  # absolute DIC state index
    holdout: bool
    displacement: FloatArray  # simulated, (nx+1, ny+1, 2)
    measured_displacement: FloatArray  # measured DIC, same layout
    stress_in_plane_mpa: FloatArray  # (nx, ny, 2, 3)
    plastic_strain_tensor: FloatArray | None  # (nx, ny, 2, 3, 3)
    committed_state: FloatArray  # q_n after commit, (P, 12, 1+d)
    strain_in_plane_mpa: FloatArray  # committed strain eps_n, (P, 3)
    committed_tangent_mpa: FloatArray | None  # accepted C_alg, (P, 3, 3)
    dissipation: FloatArray  # accepted generalised dissipation, (P,)
    slip_work: FloatArray  # accepted slip-channel work, (P,)
    loss_raw: float
    loss_whitened: float | None
    equilibrium_residual: float


@dataclass(frozen=True, slots=True)
class TannFCCSequenceResult:
    records: tuple[TannFCCStateRecord, ...]
    solver_diagnostics: object
    total_loss_raw: float
    total_loss_whitened: float | None


class TannFCCSequence:
    """Play the boundary history through the solver and score the DIC.

    `boundary_history` is `(S, nx+1, ny+1, 2)` in the same displacement
    units as the measurement; its first state must be all-zero (the
    reference). `measured_interior` is the measured interior displacement
    per state, same layout -- only the interior degrees of freedom are
    scored. `holdout` holds absolute DIC state indices; their loss is
    computed but must never enter the training objective.
    """

    def __init__(
        self,
        *,
        grid: StructuredGrid2D,
        material: TannFCCBatch,
        boundary_history: FloatArray,
        measured_interior: FloatArray,
        state_indices: list[int],
        holdout: set[int],
        whitener: Callable[[FloatArray], FloatArray] | None = None,
        solver_config: EBISpectralSolverConfig | None = None,
    ) -> None:
        history = np.asarray(boundary_history, dtype=np.float64)
        measured = np.asarray(measured_interior, dtype=np.float64)
        # The boundary history carries the zero reference plus one entry per
        # increment; the measured interior is aligned to the increments.
        if history.ndim != 4 or history.shape[0] != len(state_indices) + 1:
            raise ValueError(
                "expected a boundary history with len(state_indices) + 1 states"
            )
        if measured.ndim != 4 or measured.shape[0] != len(state_indices):
            raise ValueError(
                "expected the measured interior aligned to the increments"
            )
        if history.shape[1:] != measured.shape[1:]:
            raise ValueError("boundary and interior histories must share the field shape")
        if not np.allclose(history[0], 0.0):
            raise ValueError("the first boundary state must be the zero reference")
        self.grid = grid
        self.material = material
        self.boundary_history = history
        self.measured_interior = measured
        self.state_indices = list(state_indices)
        self.holdout = set(holdout)
        self.whitener = whitener
        self.solver_config = solver_config

    @staticmethod
    def _interior_mask(shape: tuple[int, ...]) -> np.ndarray:
        interior = np.zeros(shape[:2], dtype=bool)
        interior[1:-1, 1:-1] = True
        return interior

    def rollout(self) -> TannFCCSequenceResult:
        records: list[TannFCCStateRecord] = []

        def observe(fields: TwoStateIncrementFields) -> None:
            index = fields.increment - 1  # the solver counts increments from 1
            if index < 0 or index >= len(self.state_indices):
                return
            state = self.state_indices[index]
            displacement = np.array(fields.displacement, copy=True)
            measured = self.measured_interior[index]
            holdout = state in self.holdout
            interior = self._interior_mask(displacement.shape)
            residual_field = displacement - measured
            loss_raw = 0.5 * float(np.sum(residual_field[interior] ** 2))
            loss_whitened = None
            if self.whitener is not None:
                weighted = self.whitener(residual_field[1:-1, 1:-1])
                loss_whitened = 0.5 * float(np.sum(weighted**2))
            stress = np.array(fields.stress_in_plane_mpa, copy=True)
            plastic = (
                None
                if fields.plastic_strain_tensor is None
                else np.array(fields.plastic_strain_tensor, copy=True)
            )
            records.append(
                TannFCCStateRecord(
                    state=state,
                    holdout=holdout,
                    displacement=displacement,
                    measured_displacement=measured,
                    stress_in_plane_mpa=stress,
                    plastic_strain_tensor=plastic,
                    # the observer runs after the converged commit, so these
                    # are exactly the q_n / eps_n the next increment starts from
                    committed_state=np.array(self.material.committed_state, copy=True),
                    strain_in_plane_mpa=np.array(
                        self.material.committed_strain, copy=True
                    ),
                    committed_tangent_mpa=(
                        None
                        if self.material.last_committed_tangent is None
                        else np.array(self.material.last_committed_tangent, copy=True)
                    ),
                    dissipation=np.array(
                        self.material.last_committed_dissipation, copy=True
                    ),
                    slip_work=np.array(self.material.last_committed_slip_work, copy=True),
                    loss_raw=loss_raw,
                    loss_whitened=loss_whitened,
                    equilibrium_residual=0.0,  # filled after the solve
                )
            )

        result = solve_two_state_dirichlet_plane_stress(
            grid=self.grid,
            material=self.material,
            boundary_displacement_history=self.boundary_history,
            config=self.solver_config,
            increment_observer=observe,
        )
        total_raw = sum(record.loss_raw for record in records if not record.holdout)
        total_whitened = (
            None
            if self.whitener is None
            else sum(
                record.loss_whitened for record in records if not record.holdout
            )
        )
        return TannFCCSequenceResult(
            records=tuple(records),
            solver_diagnostics=result.diagnostics,
            total_loss_raw=total_raw,
            total_loss_whitened=total_whitened,
        )
