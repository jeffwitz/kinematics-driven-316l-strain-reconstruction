"""Limited-memory local multisecant least-change correction, per element.

Status: **experimental_falsified**. Not qualified for a solver, off by default,
kept as the reproducible record of a negative result. See
`validation/cps4r_as_broyden_results.md` before using any of this.

Sections 9 to 14 of the 2026-08-04 specification.

A note on the name, because the first version of this module got it wrong. This
is **not** good Broyden. Broyden's method and its convergence theory concern a
*square* Jacobian of the global residual `R: R^n -> R^n`, updated from global
pairs `(s_k, y_k)`. What is built here is a rectangular `2 x 5` least-change
multisecant regression of a *local* map, one per element, assembled afterwards.
The algebra is well posed; the convergence results do not transfer, and
measurement confirms they do not: the local secant conditions are met to
`1e-15` while the **global** secant defect of the assembled matrix grows by a
factor of five.

What this learns, and what it must never touch. The residual, the stresses, the
slips, the internal variables and the stabilising force are all left exactly as
`assumed_strain_energy` computes them. Only the **matrix** used to obtain the
Newton correction is improved, from the secant pairs the iteration has already
produced -- so nothing here costs a constitutive call.

The correction solves

    min ||dG||_F   subject to   dG S = Z

whose solution is `dG = Z S^+`. `S^+` is taken through an SVD with rank
detection rather than through `(S^T S)^{-1}`, because dependent search
directions are the normal case, not the exception, and inverting a Gram matrix
of dependent columns is how a quasi-Newton scheme produces an arbitrarily large
correction instead of no correction.

There is no relaxation coefficient and no damping. Section 14 forbids them in
the main variant, and the whole point of the previous element work was to remove
a free coefficient rather than add one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Reduced state dimension: three central strains plus two hourglass amplitudes.
REDUCED_DIMENSION = 5
#: Generalised hourglass force dimension.
MODAL_DIMENSION = 2

#: Singular values below this fraction of the largest are dropped. A numerical
#: tolerance; section 13 forbids tuning it on CPS4.
DEFAULT_RANK_TOLERANCE = 1.0e-8
#: A step shorter than this fraction of the current reduced state carries no
#: information and would amplify round-off into the correction.
DEFAULT_STEP_TOLERANCE = 1.0e-12

#: More pairs than the input dimension add no independent information.
MAXIMUM_MEMORY = REDUCED_DIMENSION


@dataclass(slots=True)
class BroydenMemory:
    """Circular memory of normalised secant pairs for one element.

    Pairs are stored normalised by `||s||`, which preserves the secant condition
    `G s = y` exactly while keeping the columns of `S` comparable in magnitude.
    Section 11; the norms are taken in the reduced coordinates, with no weighting.
    """

    memory: int
    steps: list[FloatArray] = field(default_factory=list)
    increments: list[FloatArray] = field(default_factory=list)
    rejected: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.memory <= MAXIMUM_MEMORY:
            raise ValueError(
                f"memory must lie in 1..{MAXIMUM_MEMORY}; the reduced space has "
                f"{REDUCED_DIMENSION} dimensions and more pairs than that carry no "
                f"independent information, got {self.memory}"
            )

    @property
    def pair_count(self) -> int:
        return len(self.steps)

    def clear(self) -> None:
        self.steps.clear()
        self.increments.clear()

    def add(self, step: ArrayLike, force_increment: ArrayLike, *, scale: float) -> bool:
        """Store one pair, or refuse it and say so.

        `scale` is the norm of the current reduced state; a step short relative
        to it is round-off rather than information.
        """

        reduced = np.asarray(step, dtype=float).reshape(-1)
        modal = np.asarray(force_increment, dtype=float).reshape(-1)
        if reduced.shape != (REDUCED_DIMENSION,) or modal.shape != (MODAL_DIMENSION,):
            raise ValueError(
                f"a pair is ({REDUCED_DIMENSION},) and ({MODAL_DIMENSION},), got "
                f"{reduced.shape} and {modal.shape}"
            )
        length = float(np.linalg.norm(reduced))
        if not np.isfinite(length) or not np.isfinite(modal).all():
            self.rejected += 1
            return False
        if length <= DEFAULT_STEP_TOLERANCE * max(abs(scale), 1.0):
            self.rejected += 1
            return False
        self.steps.append(reduced / length)
        self.increments.append(modal / length)
        if len(self.steps) > self.memory:
            self.steps.pop(0)
            self.increments.pop(0)
        return True


@dataclass(frozen=True, slots=True)
class BroydenCorrection:
    """The reduced correction of one element, with what it cost to get it."""

    correction: FloatArray
    rank: int
    pairs_used: int
    secant_defect_before: float
    secant_defect_after: float


def _secant_defect(
    jacobian: FloatArray, steps: FloatArray, increments: FloatArray
) -> float:
    """`max_i |y_i - G s_i| / |y_i|` over the stored pairs."""

    if steps.size == 0:
        return 0.0
    predicted = steps @ jacobian.T
    residual = np.linalg.norm(increments - predicted, axis=1)
    return float(np.max(residual / (np.linalg.norm(increments, axis=1) + 1e-300)))


def build_correction(
    memory: BroydenMemory,
    base_jacobian: ArrayLike,
    *,
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> BroydenCorrection:
    """Solve `min ||dG||_F` subject to `dG S = Z`, with `Z = Y - G_0 S`.

    The defects `Z` are recomputed against the **current** base Jacobian every
    time, as section 9 requires: accumulating a correction relative to an older
    base while the constitutive tangent has moved would be learning the wrong
    thing.

    Returns a zero correction rather than raising on any degeneracy. Section 28
    makes the fallback the safe path: the element then uses its base Jacobian,
    the residual is untouched, and the iteration continues.
    """

    jacobian = np.asarray(base_jacobian, dtype=float)
    if jacobian.shape != (MODAL_DIMENSION, REDUCED_DIMENSION):
        raise ValueError(
            f"a base reduced Jacobian is {(MODAL_DIMENSION, REDUCED_DIMENSION)}, got "
            f"{jacobian.shape}"
        )
    zero = np.zeros((MODAL_DIMENSION, REDUCED_DIMENSION))
    if memory.pair_count == 0 or not np.isfinite(jacobian).all():
        return BroydenCorrection(zero, 0, memory.pair_count, 0.0, 0.0)

    steps = np.array(memory.steps)
    increments = np.array(memory.increments)
    before = _secant_defect(jacobian, steps, increments)
    defects = increments - steps @ jacobian.T

    # `S` has the pairs as COLUMNS, so `dG S = Z` with S of shape (5, m).
    step_matrix = steps.T
    defect_matrix = defects.T
    try:
        left, singular, right = np.linalg.svd(step_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return BroydenCorrection(zero, 0, memory.pair_count, before, before)
    if singular.size == 0 or not np.isfinite(singular).all():
        return BroydenCorrection(zero, 0, memory.pair_count, before, before)
    kept = singular > rank_tolerance * float(singular.max())
    rank = int(np.count_nonzero(kept))
    if rank == 0:
        return BroydenCorrection(zero, 0, memory.pair_count, before, before)

    # `S^+ = V diag(1/sigma) U^T`, restricted to the kept directions.
    pseudo_inverse = (right[kept].T / singular[kept]) @ left[:, kept].T
    correction = defect_matrix @ pseudo_inverse
    if not np.isfinite(correction).all():
        return BroydenCorrection(zero, 0, memory.pair_count, before, before)
    after = _secant_defect(jacobian + correction, steps, increments)
    return BroydenCorrection(correction, rank, memory.pair_count, before, after)


class ElementBroydenMemories:
    """One memory per element, with the transaction behaviour of section 17.

    The memory is a numerical device of the increment, not material state: it is
    cleared at the start of every increment and after any cutback, and it is
    never carried into the constitutive archive.
    """

    def __init__(self, element_count: int, *, memory: int) -> None:
        if element_count < 1:
            raise ValueError("element_count must be positive")
        self._memories = [BroydenMemory(memory=memory) for _ in range(element_count)]

    def __len__(self) -> int:
        return len(self._memories)

    def __getitem__(self, index: int) -> BroydenMemory:
        return self._memories[index]

    def clear(self) -> None:
        for item in self._memories:
            item.clear()

    @property
    def total_pairs(self) -> int:
        return sum(item.pair_count for item in self._memories)

    @property
    def total_rejected(self) -> int:
        return sum(item.rejected for item in self._memories)

    def add_batch(
        self,
        steps: ArrayLike,
        force_increments: ArrayLike,
        scales: ArrayLike,
        *,
        mask: ArrayLike | None = None,
    ) -> int:
        """Offer one pair per element; return how many were accepted.

        A `False` in `mask` means the pair is not offered at all, which is not
        the same as being refused: the caller has established that this
        element's pair would not be meaningful, so it must not be counted
        against the step-length filter of `BroydenMemory.add`.
        """

        reduced = np.asarray(steps, dtype=float)
        modal = np.asarray(force_increments, dtype=float)
        magnitudes = np.asarray(scales, dtype=float)
        offered = (
            np.ones(len(self._memories), dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        if offered.shape != (len(self._memories),):
            raise ValueError(
                f"mask must have shape {(len(self._memories),)}, got {offered.shape}"
            )
        accepted = 0
        for index, item in enumerate(self._memories):
            if not offered[index]:
                continue
            if item.add(reduced[index], modal[index], scale=float(magnitudes[index])):
                accepted += 1
        return accepted

    def build_batch(
        self,
        base_jacobians: ArrayLike,
        *,
        rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
    ) -> tuple[FloatArray, dict[str, float]]:
        """Corrections for every element, plus the diagnostics of section 21."""

        jacobians = np.asarray(base_jacobians, dtype=float)
        corrections = np.zeros((len(self._memories), MODAL_DIMENSION, REDUCED_DIMENSION))
        ranks: list[int] = []
        before: list[float] = []
        after: list[float] = []
        for index, item in enumerate(self._memories):
            result = build_correction(
                item, jacobians[index], rank_tolerance=rank_tolerance
            )
            corrections[index] = result.correction
            ranks.append(result.rank)
            before.append(result.secant_defect_before)
            after.append(result.secant_defect_after)
        diagnostics = {
            "broyden_pairs_total": float(self.total_pairs),
            "broyden_pairs_mean": float(self.total_pairs / len(self._memories)),
            "broyden_pairs_rejected": float(self.total_rejected),
            "broyden_rank_mean": float(np.mean(ranks)) if ranks else 0.0,
            "broyden_rank_max": float(np.max(ranks)) if ranks else 0.0,
            "broyden_correction_norm": float(np.linalg.norm(corrections)),
            "broyden_secant_defect_before": float(np.mean(before)) if before else 0.0,
            "broyden_secant_defect_after": float(np.mean(after)) if after else 0.0,
            "broyden_elements_without_correction": float(
                np.count_nonzero(np.array(ranks) == 0)
            ),
        }
        return corrections, diagnostics
