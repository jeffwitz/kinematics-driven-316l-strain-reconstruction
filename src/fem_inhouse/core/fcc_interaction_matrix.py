"""The FCC interaction matrix: slip systems, interaction classes, and the 12x12.

Section 7 of the 2026-08-03 specification.

The sources this project takes its hardening coefficients from state **six**
numbers, one per physical interaction class. MFront takes **seven**. Nothing in
either convention says which of the seven slots each of the six belongs in, and
until now that correspondence was carried by the order of the numbers in one
literal and by nothing else.

This module derives it instead of asserting it. The twelve octahedral systems
are declared explicitly, every one of the 144 pairs is classified from its
geometry -- same plane, same Burgers direction, perpendicular directions,
whether the junction can glide -- and the mapping from class to MFront slot
falls out of that classification. The result is checked against MFront's own
`--interaction-matrix` query in the test suite, so the two cannot drift.

The thing worth knowing before reading further: MFront splits the glissile
junction into **two** ranks, 4 and 6, according to which of the two systems can
glide the junction. The publication convention has one glissile coefficient, so
it must be written into both slots. Doing so is not a formatting detail -- see
`RANK_MATRIX`, which is *not* symmetric. The numerical matrix is symmetric only
because those two slots hold the same number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]

#: The twelve octahedral slip systems, in MFront's own order.
#:
#: Verified against `mfront-query --slip-systems-by-index`. The order matters:
#: every per-system array in a result -- `PlasticSlip`, `EquivalentPlasticSlip`,
#: `BackStrain` -- is indexed by it, and so is every row and column below.
SLIP_SYSTEMS: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((0, 1, -1), (1, 1, 1)),
    ((1, 0, -1), (1, 1, 1)),
    ((1, -1, 0), (1, 1, 1)),
    ((0, 1, 1), (1, 1, -1)),
    ((1, 0, 1), (1, 1, -1)),
    ((1, -1, 0), (1, 1, -1)),
    ((0, 1, -1), (1, -1, -1)),
    ((1, 0, 1), (1, -1, -1)),
    ((1, 1, 0), (1, -1, -1)),
    ((0, 1, 1), (1, -1, 1)),
    ((1, 0, -1), (1, -1, 1)),
    ((1, 1, 0), (1, -1, 1)),
)

SLIP_SYSTEM_COUNT = len(SLIP_SYSTEMS)

InteractionClass = Literal[
    "self",
    "coplanar",
    "collinear",
    "hirth_lock",
    "glissile_junction_gliding_in_first_plane",
    "glissile_junction_gliding_in_second_plane",
    "lomer_sessile_junction",
]

#: MFront slot for each interaction class, derived by `build_rank_matrix` and
#: frozen here so a change in MFront's ordering fails a test rather than
#: silently reassigning coefficients.
CLASS_TO_RANK: dict[InteractionClass, int] = {
    "self": 0,
    "coplanar": 1,
    "hirth_lock": 2,
    "lomer_sessile_junction": 3,
    "glissile_junction_gliding_in_first_plane": 4,
    "collinear": 5,
    "glissile_junction_gliding_in_second_plane": 6,
}

#: The single glissile coefficient of the six-number publication convention
#: reaches MFront through both of these slots.
GLISSILE_RANKS: tuple[int, int] = (4, 6)

#: Order of the six coefficients as the sources state them.
PUBLICATION_CLASS_ORDER: tuple[InteractionClass, ...] = (
    "self",
    "coplanar",
    "hirth_lock",
    "lomer_sessile_junction",
    "glissile_junction_gliding_in_first_plane",
    "collinear",
)


@dataclass(frozen=True, slots=True)
class SlipSystem:
    """One octahedral system, with integer Miller indices kept exact."""

    burgers: IntArray
    normal: IntArray

    def __post_init__(self) -> None:
        if int(np.dot(self.burgers, self.normal)) != 0:
            raise ValueError("a slip direction must lie in its slip plane")


def slip_systems() -> tuple[SlipSystem, ...]:
    return tuple(
        SlipSystem(burgers=np.array(b, dtype=int), normal=np.array(n, dtype=int))
        for b, n in SLIP_SYSTEMS
    )


def _parallel(first: IntArray, second: IntArray) -> bool:
    return bool(np.all(np.cross(first, second) == 0))


def _is_110(vector: IntArray) -> bool:
    return sorted(np.abs(vector).tolist()) == [0, 1, 1]


def classify_pair(first: SlipSystem, second: SlipSystem) -> InteractionClass:
    """Name the interaction between two systems from their geometry alone.

    The classes are the standard FCC ones. `self` and `coplanar` share a slip
    plane; `collinear` shares a Burgers direction across two planes, and is the
    strongest interaction because the two dislocations can annihilate. The rest
    form a junction, named after what that junction can do: a `hirth_lock` from
    perpendicular Burgers vectors, a `lomer_sessile_junction` whose product lies
    in neither plane and therefore cannot glide at all, and a glissile junction
    which can glide -- in exactly one of the two planes, which is what splits it
    into MFront's two ranks.
    """

    if _parallel(first.normal, second.normal):
        return "self" if _parallel(first.burgers, second.burgers) else "coplanar"
    if _parallel(first.burgers, second.burgers):
        return "collinear"
    if int(np.dot(first.burgers, second.burgers)) == 0:
        return "hirth_lock"
    for sign in (1, -1):
        junction = first.burgers + sign * second.burgers
        if _is_110(junction):
            break
    else:  # pragma: no cover - impossible for the octahedral family
        raise AssertionError("two non-parallel <110> directions must combine to <110>")
    in_first = int(np.dot(junction, first.normal)) == 0
    in_second = int(np.dot(junction, second.normal)) == 0
    if in_first and in_second:  # pragma: no cover - would mean coplanar
        raise AssertionError("a junction gliding in both planes means one plane")
    if in_first:
        return "glissile_junction_gliding_in_first_plane"
    if in_second:
        return "glissile_junction_gliding_in_second_plane"
    return "lomer_sessile_junction"


def build_class_matrix() -> tuple[tuple[InteractionClass, ...], ...]:
    """The 12x12 of interaction class names."""

    systems = slip_systems()
    return tuple(
        tuple(classify_pair(row, column) for column in systems) for row in systems
    )


def build_rank_matrix() -> IntArray:
    """The 12x12 of MFront coefficient slots.

    **Not symmetric.** Entry `(s, r)` is rank 4 where `(r, s)` is rank 6,
    whenever the junction they form glides in the plane of `s` but not of `r`.
    Only the numerical matrix built on top of it comes out symmetric, and only
    when both glissile slots carry the same value.
    """

    classes = build_class_matrix()
    return np.array(
        [[CLASS_TO_RANK[name] for name in row] for row in classes], dtype=int
    )


def build_interaction_matrix(coefficients: tuple[float, ...] | list[float]) -> FloatArray:
    """Expand seven MFront coefficients into the 12x12 hardening matrix."""

    values = np.asarray(coefficients, dtype=float)
    if values.shape != (7,):
        raise ValueError(f"an FCC interaction matrix has seven coefficients, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("interaction coefficients must be finite")
    return values[build_rank_matrix()]


def is_symmetric(coefficients: tuple[float, ...] | list[float]) -> bool:
    """Whether these seven coefficients give a symmetric hardening matrix.

    They do exactly when the two glissile slots agree. A set that fails this
    makes system `s` harden `r` differently from how `r` hardens `s`, which
    leaves the six-coefficient convention of the sources without saying so.
    """

    values = np.asarray(coefficients, dtype=float)
    return bool(values[GLISSILE_RANKS[0]] == values[GLISSILE_RANKS[1]])


def from_publication_coefficients(six: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Convert the six published coefficients into MFront's seven.

    The order is `PUBLICATION_CLASS_ORDER`: self, coplanar, Hirth, Lomer,
    glissile, collinear. The glissile value is written into both MFront glissile
    slots, which is the whole content of the conversion and the reason this
    function exists rather than a comment next to a literal.
    """

    values = [float(value) for value in np.asarray(six, dtype=float)]
    if len(values) != 6:
        raise ValueError(f"the publication convention has six coefficients, got {len(values)}")
    seven = [0.0] * 7
    for name, value in zip(PUBLICATION_CLASS_ORDER, values, strict=True):
        seven[CLASS_TO_RANK[name]] = value
    seven[GLISSILE_RANKS[1]] = seven[GLISSILE_RANKS[0]]
    return tuple(seven)


def class_pair_counts() -> dict[InteractionClass, int]:
    """How many of the 144 ordered pairs fall in each class."""

    counts: dict[InteractionClass, int] = {}
    for row in build_class_matrix():
        for name in row:
            counts[name] = counts.get(name, 0) + 1
    return counts
