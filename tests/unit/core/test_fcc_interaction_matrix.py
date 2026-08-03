"""The FCC interaction matrix, checked against an explicitly archived one.

Section 7. The archived matrix below is the output of
`mfront-query --interaction-matrix-structure` on
`mfront/Fcc316LForestRubinSrix.mfront`, typed in by hand so that the derivation
in `fcc_interaction_matrix` is compared against MFront rather than against
itself. If TFEL ever reorders its slip systems, this file fails and the
correspondence between published coefficients and MFront slots has to be
re-established rather than silently reassigned.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.fcc_interaction_matrix import (
    CLASS_TO_RANK,
    GLISSILE_RANKS,
    PUBLICATION_CLASS_ORDER,
    SLIP_SYSTEM_COUNT,
    SLIP_SYSTEMS,
    build_class_matrix,
    build_interaction_matrix,
    build_rank_matrix,
    class_pair_counts,
    classify_pair,
    from_publication_coefficients,
    is_symmetric,
    slip_systems,
)
from fem_inhouse.core.srix_parameters import get_parameter_set

#: `mfront-query --interaction-matrix`, TFEL 5.1.0, transcribed.
ARCHIVED_RANKS = (
    (0, 1, 1, 2, 3, 4, 5, 6, 6, 2, 4, 3),
    (1, 0, 1, 3, 2, 4, 4, 2, 3, 6, 5, 6),
    (1, 1, 0, 6, 6, 5, 4, 3, 2, 3, 4, 2),
    (2, 3, 4, 0, 1, 1, 2, 4, 3, 5, 6, 6),
    (3, 2, 4, 1, 0, 1, 6, 5, 6, 4, 2, 3),
    (6, 6, 5, 1, 1, 0, 3, 4, 2, 4, 3, 2),
    (5, 6, 6, 2, 4, 3, 0, 1, 1, 2, 3, 4),
    (4, 2, 3, 6, 5, 6, 1, 0, 1, 3, 2, 4),
    (4, 3, 2, 3, 4, 2, 1, 1, 0, 6, 6, 5),
    (2, 4, 3, 5, 6, 6, 2, 3, 4, 0, 1, 1),
    (6, 5, 6, 4, 2, 3, 3, 2, 4, 1, 0, 1),
    (3, 4, 2, 4, 3, 2, 6, 6, 5, 1, 1, 0),
)

#: `mfront-query --slip-systems-by-index`, same run.
ARCHIVED_SYSTEMS = (
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

HISTORICAL = "316l_srix_transposed_from_nasri2018_rate_1e-3"


def test_the_declared_slip_systems_are_the_ones_mfront_uses() -> None:
    assert SLIP_SYSTEMS == ARCHIVED_SYSTEMS
    assert SLIP_SYSTEM_COUNT == 12


def test_every_slip_direction_lies_in_its_plane() -> None:
    """Cheap, and it catches a transcription slip immediately."""

    for system in slip_systems():
        assert int(np.dot(system.burgers, system.normal)) == 0


def test_the_derived_ranks_reproduce_mfront_exactly() -> None:
    """The load-bearing test of this file.

    Nothing here reads MFront's answer: the ranks are derived from the geometry
    of the pairs. Agreeing with the archived query on all 144 entries is what
    establishes that the classification and MFront's ordering are the same map.
    """

    np.testing.assert_array_equal(build_rank_matrix(), np.array(ARCHIVED_RANKS))


def test_the_rank_matrix_is_not_symmetric() -> None:
    """Stated because it is surprising and because it has a consequence.

    Entry (0, 7) is a glissile junction gliding in the plane of system 7, entry
    (7, 0) the same junction gliding in the plane of system 0. MFront gives them
    different slots. The numerical matrix is symmetric only when both slots hold
    the same number.
    """

    ranks = build_rank_matrix()

    assert not np.array_equal(ranks, ranks.T)
    assert (ranks[0, 7], ranks[7, 0]) == (6, 4)


class TestClassification:
    def test_the_classes_partition_the_pairs_as_expected(self) -> None:
        counts = class_pair_counts()

        assert counts["self"] == 12
        assert counts["collinear"] == 12
        assert counts["coplanar"] == 24
        assert counts["hirth_lock"] == 24
        assert counts["lomer_sessile_junction"] == 24
        assert counts["glissile_junction_gliding_in_first_plane"] == 24
        assert counts["glissile_junction_gliding_in_second_plane"] == 24
        assert sum(counts.values()) == 144

    def test_the_diagonal_is_self_interaction(self) -> None:
        classes = build_class_matrix()

        for index in range(SLIP_SYSTEM_COUNT):
            assert classes[index][index] == "self"

    def test_systems_sharing_a_plane_are_coplanar(self) -> None:
        """0, 1 and 2 all live on (1,1,1)."""

        classes = build_class_matrix()

        assert classes[0][1] == "coplanar"
        assert classes[1][2] == "coplanar"

    def test_systems_sharing_a_burgers_direction_are_collinear(self) -> None:
        """System 0 and system 6 are both [0,1,-1], on (1,1,1) and (1,-1,-1)."""

        systems = slip_systems()

        assert np.array_equal(systems[0].burgers, systems[6].burgers)
        assert classify_pair(systems[0], systems[6]) == "collinear"

    def test_collinear_carries_the_largest_coefficient(self) -> None:
        """The physical reason the sources single it out: two dislocations on the
        same Burgers vector can annihilate, so the interaction is far stronger
        than any junction."""

        coefficients = get_parameter_set(HISTORICAL).interaction_matrix
        collinear = coefficients[CLASS_TO_RANK["collinear"]]

        assert collinear == pytest.approx(12.3)
        assert collinear > max(
            value
            for index, value in enumerate(coefficients)
            if index != CLASS_TO_RANK["collinear"]
        )

    def test_perpendicular_burgers_vectors_give_a_hirth_lock(self) -> None:
        systems = slip_systems()
        first, second = systems[0], systems[3]

        assert int(np.dot(first.burgers, second.burgers)) == 0
        assert classify_pair(first, second) == "hirth_lock"

    def test_the_two_glissile_classes_are_the_same_junction_seen_twice(self) -> None:
        """Which is why the publication convention has one coefficient for both."""

        systems = slip_systems()
        forward = classify_pair(systems[0], systems[7])
        backward = classify_pair(systems[7], systems[0])

        assert forward == "glissile_junction_gliding_in_second_plane"
        assert backward == "glissile_junction_gliding_in_first_plane"


class TestExpansion:
    def test_the_historical_coefficients_give_a_symmetric_matrix(self) -> None:
        coefficients = get_parameter_set(HISTORICAL).interaction_matrix
        matrix = build_interaction_matrix(coefficients)

        assert is_symmetric(coefficients)
        np.testing.assert_allclose(matrix, matrix.T)

    def test_unequal_glissile_slots_break_the_symmetry(self) -> None:
        """The consequence of the split, made visible.

        A set that does this has left the six-coefficient convention of the
        sources, and system s then hardens r differently from how r hardens s.
        """

        asymmetric = (1.0, 1.0, 0.6, 1.8, 1.6, 12.3, 9.9)
        matrix = build_interaction_matrix(asymmetric)

        assert not is_symmetric(asymmetric)
        assert not np.allclose(matrix, matrix.T)

    def test_the_diagonal_carries_the_self_coefficient(self) -> None:
        matrix = build_interaction_matrix((7.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0))

        np.testing.assert_allclose(np.diag(matrix), 7.0)

    def test_a_matrix_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="seven coefficients"):
            build_interaction_matrix((1.0,) * 6)

    def test_a_non_finite_coefficient_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            build_interaction_matrix((1.0, 1.0, 0.6, 1.8, 1.6, float("inf"), 1.6))


class TestPublicationConversion:
    def test_six_published_coefficients_reproduce_the_registered_seven(self) -> None:
        """The conversion the specification asks to formalise, checked end to end."""

        published = (1.0, 1.0, 0.6, 1.8, 1.6, 12.3)

        assert from_publication_coefficients(published) == get_parameter_set(
            HISTORICAL
        ).interaction_matrix

    def test_the_glissile_value_lands_in_both_slots(self) -> None:
        seven = from_publication_coefficients((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

        assert seven[GLISSILE_RANKS[0]] == 5.0
        assert seven[GLISSILE_RANKS[1]] == 5.0
        assert is_symmetric(seven)

    def test_the_conversion_is_ordered_as_the_sources_state_it(self) -> None:
        assert PUBLICATION_CLASS_ORDER == (
            "self",
            "coplanar",
            "hirth_lock",
            "lomer_sessile_junction",
            "glissile_junction_gliding_in_first_plane",
            "collinear",
        )

    def test_a_wrong_number_of_published_coefficients_is_refused(self) -> None:
        with pytest.raises(ValueError, match="six coefficients"):
            from_publication_coefficients((1.0, 2.0, 3.0))

    def test_the_result_is_plain_floats(self) -> None:
        for value in from_publication_coefficients((1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
            assert type(value) is float
