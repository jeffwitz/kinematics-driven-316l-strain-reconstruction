"""Non-production shadow tangent diagnostics for the GPS adapter."""

from __future__ import annotations

# mypy: ignore-errors
import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.mfront_condensation import condense_kelvin_tangent_to_engineering

_ENGINEERING_TO_KELVIN_STRAIN_SCALE = np.array([1.0, 1.0, 1.0 / np.sqrt(2.0)])
_KELVIN_TO_ENGINEERING_STRESS_SCALE = np.array([1.0, 1.0, 1.0 / np.sqrt(2.0)])
_PLANE_STRESS_COMPONENTS = np.array([0, 1, 3])
_TRANSVERSE_COMPONENTS_3D = np.array([2, 4, 5])


class GPSDiagnosticsMixin:
    _last_shadow_diagnostics: dict[str, object] | None

    def _shadow_condensed_tangent(
        self,
        in_plane: NDArray,
        internal_state_variables: NDArray,
        time_increment: float,
    ) -> NDArray | None:
        """Evaluate the runtime raw full-step shadow tangent.

        A DERIVATIVE CALCULATOR, nothing else. The shadow has no life of its
        own: its committed state is transplanted from the GPS's at every call
        -- internal variables by name, committed stress as is, both being
        material-frame in either law, and the committed gradient rotated from
        the GPS's global convention into the crystal one the raw bridge uses --
        it is then integrated once at the imposed in-plane strain WITH the
        transverse strain the GPS closure converged, and reverted. This is a
        separate full-step constitutive trajectory, not a same-state
        derivative oracle when GPS used sub-stepping.

        Runtime diagnostics compare this tangent and the final internal
        variables pointwise with the GPS trial. A mismatch can therefore be a
        full-step versus sub-stepped trajectory difference, not an algebraic
        tangent defect.

        Returns `None` on any failure, leaving the DSL tangent in place.
        """

        if self._shadow is None:
            return None
        source = self._manager.s0
        target = self._shadow._manager.s0
        internal = np.asarray(source.internal_state_variables)
        destination = np.asarray(target.internal_state_variables)
        for source_slice, target_slice in self._shadow_isv_map:
            destination[:, target_slice] = internal[:, source_slice]
        target.internal_state_variables[:, :] = destination
        # Stress is material-frame in both laws; the gradient is not.
        target.thermodynamic_forces[:, :] = np.asarray(source.thermodynamic_forces)
        gradients = np.ascontiguousarray(np.asarray(source.gradients).reshape(-1).copy())
        if self._mgis_rotations is not None:
            self._mgis.rotateGradients(
                gradients, self._shadow._behaviour, self._mgis_rotations
            )
        target.gradients[:, :] = gradients.reshape(self._point_count, 6)
        self._mgis.revert(self._shadow._manager)

        total = np.zeros((self._point_count, 6), dtype=float)
        total[:, _PLANE_STRESS_COMPONENTS] = (
            in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        )
        total[:, _TRANSVERSE_COMPONENTS_3D] = np.asarray(self._manager.s1.gradients)[
            :, _TRANSVERSE_COMPONENTS_3D
        ]
        try:
            trial = self._shadow.evaluate(
                total, time_increment=time_increment, collect_observables=False
            )
            engineering, _ = condense_kelvin_tangent_to_engineering(
                trial.consistent_tangent_kelvin_mpa, check_condition=False
            )
            gps_tangent = np.asarray(self._manager.K).copy()
            in_plane_operator = np.zeros((6, 6), dtype=float)
            in_plane_operator[_PLANE_STRESS_COMPONENTS, _PLANE_STRESS_COMPONENTS] = 1.0
            gps_tangent = gps_tangent @ in_plane_operator
            if self._mgis_rotations is not None:
                for column in range(6):
                    flat = np.ascontiguousarray(gps_tangent[:, :, column].reshape(-1))
                    self._mgis.rotateThermodynamicForces(
                        flat, self._behaviour, self._mgis_rotations
                    )
                    gps_tangent[:, :, column] = flat.reshape(self._point_count, 6)
            gps_tangent = gps_tangent[:, _PLANE_STRESS_COMPONENTS][
                :, :, _PLANE_STRESS_COMPONENTS
            ]
            gps_tangent = gps_tangent * _KELVIN_TO_ENGINEERING_STRESS_SCALE[None, :, None]
            gps_tangent = gps_tangent * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, None, :]
            self._gps_tangent_engineering = gps_tangent
            tangent_error = np.linalg.norm(engineering - gps_tangent, axis=(1, 2)) / np.maximum(
                np.linalg.norm(gps_tangent, axis=(1, 2)), 1.0e-30
            )
            state_differences: dict[str, NDArray] = {}
            gps_isv = np.asarray(source.internal_state_variables)
            shadow_isv = np.asarray(target.internal_state_variables)
            for source_slice, target_slice in self._shadow_isv_map:
                name = str(source_slice)
                state_differences[name] = np.max(
                    np.abs(gps_isv[:, source_slice] - shadow_isv[:, target_slice]), axis=1
                )
            self._last_shadow_diagnostics = {
                "substep": self._last_substep_mask.copy(),
                "divisions": self._last_substep_divisions.copy(),
                "tangent_relative_error": tangent_error,
                "state_differences": state_differences,
                "gps_tangent": np.asarray(gps_tangent).copy(),
                "shadow_tangent": np.asarray(engineering).copy(),
                "scope": self._shadow_tangent_scope,
            }
        except Exception:
            self._shadow_failures += 1
            self._last_shadow_diagnostics = {"failure": True}
            return None
        finally:
            # No committed evolution of its own, ever.
            self._shadow.revert()
        if not np.isfinite(engineering).all():
            self._shadow_failures += 1
            return None
        selected = np.ones(self._point_count, dtype=bool)
        if self._shadow_tangent_scope == "substepped":
            selected = self._last_substep_mask
        elif self._shadow_tangent_scope == "non_substepped":
            selected = ~self._last_substep_mask
        result = np.asarray(self._gps_tangent_engineering, dtype=float).copy()
        result[selected] = np.asarray(engineering, dtype=float)[selected]
        return result

    @property
    def shadow_failures(self) -> int:
        return self._shadow_failures

    @property
    def maximum_kinematic_defect(self) -> float:
        """Worst relative violation of `tr(eel) = tr(eps_total)` so far."""

        return self._maximum_kinematic_defect
