"""Quasi-Newton corrections of the element Jacobian, and nothing else.

Sections 15 to 19 of the 2026-08-04 Broyden specification.

The contract of this module is narrow on purpose, and the narrowness is the
safety argument. A `NonlinearJacobianCorrection` is shown the displacements and
the stabilising forces the solver has *already* computed, and it returns a
matrix. It cannot reach a constitutive model, it is never asked for a force, and
nothing it returns enters the residual. So whatever it learns -- or fails to
learn -- the converged solution is the one the un-accelerated solver would have
reached, to the same tolerance, and the only observable is the iteration count.

Three transaction rules make the memory a device of the increment rather than
state of the material (section 17):

- it is cleared at the start of every increment, because the secant information
  of a previous load step describes a tangent that has since moved;
- a pair is offered only from states the Newton loop has **accepted**, never
  from a line-search trial, which is a probe and not an iterate;
- it is purged on cutback, along with everything else the failed attempt built.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.assumed_strain import CentralOperators
from fem_inhouse.core.hourglass_modal_coordinates import (
    MODAL_PROJECTION_TOLERANCE,
    modal_coordinates,
)
from fem_inhouse.core.limited_memory_broyden import (
    DEFAULT_RANK_TOLERANCE,
    MAXIMUM_MEMORY,
    ElementBroydenMemories,
)

FloatArray = NDArray[np.float64]

#: Section 24 sweeps `1, 3, 5`. Five is the reduced dimension, so it is the
#: largest memory that can carry independent information, and it is the starting
#: point rather than a result: the default only changes if a sweep says so.
DEFAULT_BROYDEN_MEMORY = 5


class NonlinearJacobianCorrection(Protocol):
    """What the Newton loop is allowed to ask a correction for, section 18."""

    @property
    def name(self) -> str: ...

    def begin_increment(self) -> None:
        """Called once per load increment, before its first iteration."""

    def observe(
        self, displacements: ArrayLike, stabilisation_forces: ArrayLike
    ) -> None:
        """Offer one accepted iterate: nodal displacements and stabilising forces.

        Both are per element, shaped `(n, 8)`. Called at the top of a Newton
        iteration and nowhere else, so every state it sees was accepted.
        """

    def matrix(self, stabilisation_tangents: ArrayLike) -> FloatArray | None:
        """The `(n, 8, 8)` correction to add to the element tangent, or `None`.

        `None` means "add nothing", and is what a correction with no information
        yet returns -- distinct from a zero array, which would cost an assembly.
        """

    def discard(self) -> None:
        """Called when an increment fails, before it is retried smaller."""

    @property
    def diagnostics(self) -> dict[str, float]: ...


class NoJacobianCorrection:
    """The default: the consistent tangent, untouched.

    Present as an object rather than as a `None` branch so the solver has one
    code path, and so a run that used no correction says so in its manifest.
    """

    @property
    def name(self) -> str:
        return "none"

    def begin_increment(self) -> None:
        return None

    def observe(self, displacements: ArrayLike, stabilisation_forces: ArrayLike) -> None:
        return None

    def matrix(self, stabilisation_tangents: ArrayLike) -> FloatArray | None:
        return None

    def discard(self) -> None:
        return None

    @property
    def diagnostics(self) -> dict[str, float]:
        return {}


class BroydenHourglassCorrection:
    """Limited-memory local multisecant correction of the hourglass Jacobian.

    Status: **experimental_falsified**. Measured on the registered SRIX case it
    costs iterations rather than saving them -- 50, 57, 64 against 47 for
    memories 1, 3, 5 -- and the global secant defect of the assembled matrix
    grows by up to a factor of five while every local secant condition is met to
    `1e-15`. It is kept, off by default, as the record of that measurement.
    Read `validation/cps4r_as_broyden_results.md` before switching it on.

    Do not attempt to rescue it with a different memory, a different rank
    tolerance, a relaxation coefficient or a local damping: the failure is that
    a local rectangular fit does not improve the global Jacobian, and none of
    those touch it.

    The defect being repaired is measured, not supposed: the physical element
    tangent is consistent to `1.9e-6`, and the stabilisation tangent is wrong by
    `370 %` because `f_stab(u, C(u))` is differentiated holding `C` fixed. The
    correction is therefore learned on the stabilisation alone, in the five
    coordinates that carry it, and expanded as `H^T dG T` -- a form in which the
    rigid modes are structurally in the kernel.

    An element whose stabilising force has left the span of its two hourglass
    modes is excluded and its memory cleared: the reduced pair would then be a
    projection that lost part of the force, and learning from it would be
    learning something false. Measured at `1e-16` against a `1e-10` bound on
    every geometry tested, so the branch is a guard rather than a common path.
    """

    def __init__(
        self,
        operators: CentralOperators,
        element_count: int,
        *,
        memory: int = DEFAULT_BROYDEN_MEMORY,
        rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
        projection_tolerance: float = MODAL_PROJECTION_TOLERANCE,
    ) -> None:
        if not 1 <= memory <= MAXIMUM_MEMORY:
            raise ValueError(
                f"jacobian_correction_memory must lie in 1..{MAXIMUM_MEMORY}, got {memory}"
            )
        if not 0.0 < rank_tolerance < 1.0:
            raise ValueError("rank_tolerance must lie in (0, 1)")
        if projection_tolerance <= 0.0:
            raise ValueError("projection_tolerance must be positive")
        self._coordinates = modal_coordinates(operators)
        self._memories = ElementBroydenMemories(element_count, memory=memory)
        self._memory = int(memory)
        self._rank_tolerance = float(rank_tolerance)
        self._projection_tolerance = float(projection_tolerance)
        self._previous_state: FloatArray | None = None
        self._previous_force: FloatArray | None = None
        self._latest: dict[str, float] = {}
        self._pairs_offered = 0
        self._pairs_accepted = 0
        self._excluded = 0
        self._purges = 0
        self._maximum_projection_defect = 0.0
        self._corrections_applied = 0

    @property
    def name(self) -> str:
        return "broyden"

    def begin_increment(self) -> None:
        self._memories.clear()
        self._previous_state = None
        self._previous_force = None

    def discard(self) -> None:
        self._purges += 1
        self.begin_increment()

    def observe(self, displacements: ArrayLike, stabilisation_forces: ArrayLike) -> None:
        nodal = np.asarray(displacements, dtype=float)
        forces = np.asarray(stabilisation_forces, dtype=float)
        if nodal.shape != forces.shape or nodal.ndim != 2 or nodal.shape[1] != 8:
            raise ValueError(
                "displacements and stabilisation forces must both have shape "
                f"(n, 8), got {nodal.shape} and {forces.shape}"
            )
        state = self._coordinates.reduced_state(nodal)
        modal = self._coordinates.modal_force(forces)
        defect = self._coordinates.modal_projection_defect(forces)
        self._maximum_projection_defect = max(
            self._maximum_projection_defect, float(defect.max())
        )
        representable = defect <= self._projection_tolerance

        if self._previous_state is not None and self._previous_force is not None:
            self._pairs_offered += int(np.count_nonzero(representable))
            self._pairs_accepted += self._memories.add_batch(
                state - self._previous_state,
                modal - self._previous_force,
                np.linalg.norm(state, axis=1),
                mask=representable,
            )
        for index in np.flatnonzero(~representable):
            self._excluded += 1
            self._memories[int(index)].clear()
        self._previous_state = state
        self._previous_force = modal

    def matrix(self, stabilisation_tangents: ArrayLike) -> FloatArray | None:
        base = self._coordinates.reduced_jacobian(stabilisation_tangents)
        reduced, measured = self._memories.build_batch(
            base, rank_tolerance=self._rank_tolerance
        )
        self._latest = measured
        if measured["broyden_correction_norm"] == 0.0:
            return None
        self._corrections_applied += 1
        return self._coordinates.expand_correction(reduced)

    @property
    def diagnostics(self) -> dict[str, float]:
        """Section 21, plus what the transaction rules did over the whole run."""

        return {
            **self._latest,
            "broyden_memory": float(self._memory),
            "broyden_pairs_offered": float(self._pairs_offered),
            "broyden_pairs_accepted": float(self._pairs_accepted),
            "broyden_elements_excluded": float(self._excluded),
            "broyden_maximum_projection_defect": self._maximum_projection_defect,
            "broyden_memory_purges": float(self._purges),
            "broyden_corrected_iterations": float(self._corrections_applied),
        }


#: Selectable corrections, section 18. A name is the whole of the public API:
#: nothing outside this module constructs a correction directly.
JACOBIAN_CORRECTIONS: tuple[str, ...] = ("none", "broyden")


def make_jacobian_correction(
    name: str,
    *,
    operators: CentralOperators | None = None,
    element_count: int = 0,
    memory: int = DEFAULT_BROYDEN_MEMORY,
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> NonlinearJacobianCorrection:
    """Build a correction by name, refusing an unusable combination early.

    `broyden` needs the assumed-strain geometry, so it is only constructible for
    the formulation that has one. Refusing here rather than silently doing
    nothing is what keeps a manifest that says `broyden` from describing a run
    that had no correction at all.
    """

    if name == "none":
        return NoJacobianCorrection()
    if name == "broyden":
        if operators is None or element_count < 1:
            raise ValueError(
                "the broyden jacobian correction repairs the assumed-strain "
                "hourglass tangent and is only available for element_formulation="
                "'cps4r_as'"
            )
        return BroydenHourglassCorrection(
            operators,
            element_count,
            memory=memory,
            rank_tolerance=rank_tolerance,
        )
    known = ", ".join(JACOBIAN_CORRECTIONS)
    raise ValueError(f"unknown jacobian_correction {name!r}; available: {known}")
