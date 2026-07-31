from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.otsu_morphology import (
    describe_morphology,
    morphology_distance,
    otsu_threshold,
)

SHAPE = (120, 160)


def _two_bands(amplitude=1.0, minor=8.0, angle=0.0, background=0.1) -> np.ndarray:
    r = np.arange(SHAPE[0], dtype=np.float64)[:, None]
    c = np.arange(SHAPE[1], dtype=np.float64)[None, :]
    field = np.full(SHAPE, background)
    for offset in (-30.0, 30.0):
        d = (r - SHAPE[0] / 2 - offset) * np.cos(np.radians(angle)) - (
            c - SHAPE[1] / 2
        ) * np.sin(np.radians(angle))
        field = field + amplitude * np.exp(-0.5 * (d / minor) ** 2)
    return field


def test_otsu_separates_a_clean_bimodal_field() -> None:
    field = np.concatenate([np.zeros(500), np.ones(500)]).reshape(20, 50)

    assert 0.0 < otsu_threshold(field) < 1.0


def test_otsu_rejects_a_constant_field() -> None:
    with pytest.raises(ValueError, match="constant field"):
        otsu_threshold(np.full((10, 10), 3.0))


def test_otsu_uses_valid_values_only() -> None:
    field = np.zeros((20, 20))
    field[:10] = 1.0
    mask = np.zeros((20, 20), dtype=bool)
    mask[:10] = True  # only the high half is valid, so no bimodality remains

    with pytest.raises(ValueError, match="constant field"):
        otsu_threshold(field, valid_mask=mask)


def test_the_threshold_is_never_recomputed_per_field() -> None:
    reference = _two_bands()
    weak = _two_bands(amplitude=0.4)
    threshold = otsu_threshold(reference)

    frozen = describe_morphology(weak, threshold=threshold, label_name="frozen")
    per_field = describe_morphology(
        weak, threshold=otsu_threshold(weak), label_name="per_field"
    )
    strong = describe_morphology(reference, threshold=threshold, label_name="ref")

    # Under the frozen threshold the weak field collapses: a seventh of the
    # active area and a much thinner band.
    assert frozen.active_fraction < 0.2 * strong.active_fraction
    assert frozen.objects[0].axis_minor_pixels < 0.3 * strong.objects[0].axis_minor_pixels
    # Recomputing Otsu per field would restore the area and hide the loss, which
    # is exactly why the threshold is derived once from the DIC.
    assert per_field.active_fraction > 3.0 * frozen.active_fraction


def test_two_bands_are_eccentric_and_a_blob_is_not() -> None:
    bands = _two_bands()
    blob = np.full(SHAPE, 0.1)
    r, c = np.ogrid[: SHAPE[0], : SHAPE[1]]
    blob = blob + 1.2 * np.exp(-0.5 * (((r - 60) / 30) ** 2 + ((c - 80) / 30) ** 2))
    threshold = otsu_threshold(bands)

    band_props = describe_morphology(bands, threshold=threshold, label_name="bands")
    blob_props = describe_morphology(blob, threshold=threshold, label_name="blob")

    assert band_props.object_count == 2
    assert min(o.eccentricity for o in band_props.objects) > 0.9
    assert blob_props.object_count == 1
    assert blob_props.objects[0].eccentricity < 0.6


def test_a_rotated_pattern_is_caught_by_orientation() -> None:
    reference = _two_bands(angle=0.0)
    rotated = _two_bands(angle=35.0)
    threshold = otsu_threshold(reference)

    result = morphology_distance(
        describe_morphology(reference, threshold=threshold, label_name="ref"),
        describe_morphology(rotated, threshold=threshold, label_name="rot"),
    )

    assert result["orientation_error_degrees"] > 20.0


def test_orientation_error_is_modulo_one_hundred_and_eighty() -> None:
    reference = _two_bands(angle=1.0)
    flipped = _two_bands(angle=179.0)
    threshold = otsu_threshold(reference)

    result = morphology_distance(
        describe_morphology(reference, threshold=threshold, label_name="ref"),
        describe_morphology(flipped, threshold=threshold, label_name="flip"),
    )

    # A band has no head or tail; 1 and 179 degrees are the same direction.
    assert result["orientation_error_degrees"] < 5.0


def test_a_widened_band_is_caught_by_the_minor_axis() -> None:
    reference = _two_bands(minor=8.0)
    wide = _two_bands(minor=16.0)
    threshold = otsu_threshold(reference)

    result = morphology_distance(
        describe_morphology(reference, threshold=threshold, label_name="ref"),
        describe_morphology(wide, threshold=threshold, label_name="wide"),
    )

    assert result["axis_minor_ratio"] > 1.3


def test_a_vanished_pattern_reports_no_objects_rather_than_failing() -> None:
    reference = _two_bands()
    flat = np.full(SHAPE, 0.1)
    threshold = otsu_threshold(reference)

    result = morphology_distance(
        describe_morphology(reference, threshold=threshold, label_name="ref"),
        describe_morphology(flat, threshold=threshold, label_name="flat"),
    )

    assert result["object_count_candidate"] == 0.0
    assert result["active_fraction_candidate"] == 0.0
    assert np.isnan(result["eccentricity_error"])


def test_comparing_at_different_thresholds_is_rejected() -> None:
    field = _two_bands()
    a = describe_morphology(field, threshold=0.5, label_name="a")
    b = describe_morphology(field, threshold=0.6, label_name="b")

    with pytest.raises(ValueError, match="same threshold"):
        morphology_distance(a, b)


def test_area_alone_can_agree_while_morphology_does_not() -> None:
    # The empirical point from P43: the translated control matches the DIC on
    # active fraction to within a point while being a different shape.
    bands = _two_bands()
    threshold = otsu_threshold(bands)
    reference = describe_morphology(bands, threshold=threshold, label_name="ref")

    r, c = np.ogrid[: SHAPE[0], : SHAPE[1]]
    blob = np.full(SHAPE, 0.1) + 1.1 * np.exp(
        -0.5 * (((r - 60) / 38) ** 2 + ((c - 80) / 26) ** 2)
    )
    candidate = describe_morphology(blob, threshold=threshold, label_name="blob")

    result = morphology_distance(reference, candidate)

    assert 0.6 < result["active_fraction_ratio"] < 1.6
    assert result["object_count_difference"] != 0.0 or result["eccentricity_error"] > 0.2
