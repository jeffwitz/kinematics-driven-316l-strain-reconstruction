from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.band_geometry import (
    band_corridor,
    label_band_objects,
    order_centreline,
    prune_skeleton_spurs,
    quantile_thresholds,
    resample_polyline,
    smooth_centreline,
    tangents_and_normals,
    zhang_suen_thinning,
)


def _straight_band(shape=(60, 40), row=30, half_width=4) -> np.ndarray:
    field = np.zeros(shape)
    rows = np.arange(shape[0])[:, None]
    field += np.exp(-0.5 * ((rows - row) / half_width) ** 2)
    return field


def _diagonal_band(shape=(60, 60), half_width=3) -> np.ndarray:
    r = np.arange(shape[0])[:, None]
    c = np.arange(shape[1])[None, :]
    return np.exp(-0.5 * ((r - c) / half_width) ** 2)


def test_thresholds_use_valid_values_only() -> None:
    field = np.arange(100, dtype=float).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5] = True  # only the low half is valid

    everywhere = quantile_thresholds(field, quantiles=(0.9,))[0.9]
    masked = quantile_thresholds(field, valid_mask=mask, quantiles=(0.9,))[0.9]

    assert masked < everywhere


def test_thresholds_reject_an_empty_valid_region() -> None:
    with pytest.raises(ValueError, match="finite and non-empty"):
        quantile_thresholds(np.zeros((4, 4)), valid_mask=np.zeros((4, 4), dtype=bool))


def test_small_objects_are_dropped_by_the_declared_bound() -> None:
    field = np.zeros((40, 40))
    field[10:30, 10:30] = 1.0  # 400 px
    field[0:2, 0:2] = 1.0  # 4 px speck

    _, objects = label_band_objects(
        field, threshold_value=0.5, threshold_quantile=0.9, minimum_area_pixels=64
    )

    assert len(objects) == 1
    assert objects[0].area_pixels == 400


def test_objects_are_numbered_largest_first() -> None:
    field = np.zeros((60, 60))
    field[5:15, 5:35] = 1.0  # 300 px
    field[30:50, 30:55] = 1.0  # 500 px

    labels, objects = label_band_objects(
        field, threshold_value=0.5, threshold_quantile=0.9, minimum_area_pixels=64
    )

    assert [o.area_pixels for o in objects] == [500, 300]
    assert set(np.unique(labels)) == {0, 1, 2}
    assert int(np.count_nonzero(labels == 1)) == 500


def test_orientation_distinguishes_a_horizontal_from_a_vertical_band() -> None:
    horizontal = np.zeros((60, 60))
    horizontal[28:32, 5:55] = 1.0
    vertical = horizontal.T.copy()

    _, h = label_band_objects(horizontal, threshold_value=0.5, threshold_quantile=0.9)
    _, v = label_band_objects(vertical, threshold_value=0.5, threshold_quantile=0.9)

    # Index 0 is x, so a band extending in index 1 lies at 90 degrees.
    assert abs(abs(h[0].orientation_degrees) - 90.0) < 1.0
    assert abs(v[0].orientation_degrees) < 1.0
    assert h[0].elongation > 5.0


def test_thinning_reduces_a_thick_bar_to_one_pixel_per_column() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[12:19, 5:35] = True

    skeleton = zhang_suen_thinning(mask)

    assert skeleton.sum() < mask.sum() / 5
    interior = skeleton[:, 12:28]
    assert np.all(interior.sum(axis=0) == 1)


def test_thinning_is_deterministic() -> None:
    mask = _diagonal_band() > 0.5

    assert np.array_equal(zhang_suen_thinning(mask), zhang_suen_thinning(mask))


def test_thinning_preserves_connectivity_of_a_diagonal_band() -> None:
    from scipy import ndimage

    skeleton = zhang_suen_thinning(_diagonal_band() > 0.5)
    _, count = ndimage.label(skeleton, structure=np.ones((3, 3)))

    assert count == 1


def test_pruning_shortens_a_spur_and_keeps_the_trunk() -> None:
    skeleton = np.zeros((21, 41), dtype=bool)
    skeleton[10, 5:36] = True  # trunk, 31 px
    skeleton[7:10, 20] = True  # spur, 3 px

    pruned = prune_skeleton_spurs(skeleton, minimum_branch_pixels=8)

    assert pruned[10, 5:36].all()
    # Under eight-connectivity the pixel adjacent to the trunk touches three
    # trunk pixels, so its degree is three and the walk stops before it. One
    # residue survives; the next test shows it does not reach the centreline.
    assert int(pruned[7:10, 20].sum()) == 1


def test_a_pruning_residue_does_not_bend_the_centreline() -> None:
    skeleton = np.zeros((21, 41), dtype=bool)
    skeleton[10, 5:36] = True
    skeleton[7:10, 20] = True

    path = order_centreline(prune_skeleton_spurs(skeleton, minimum_branch_pixels=8))

    # A hop-count longest path would happily detour through the residue, since
    # the diagonal shortcut costs the same number of nodes. Euclidean weighting
    # keeps the centreline straight.
    assert np.all(path[:, 0] == 10.0)


def test_centreline_follows_the_longest_path() -> None:
    skeleton = np.zeros((21, 41), dtype=bool)
    skeleton[10, 5:36] = True

    path = order_centreline(skeleton)

    assert len(path) == 31
    assert np.allclose(path[:, 0], 10.0)
    # Ordered end to end, not scattered.
    assert np.all(np.diff(path[:, 1]) == 1.0) or np.all(np.diff(path[:, 1]) == -1.0)


def test_centreline_of_a_curved_band_is_ordered_and_continuous() -> None:
    skeleton = np.zeros((60, 60), dtype=bool)
    columns = np.arange(5, 55)
    rows = (30 + 12 * np.sin(columns / 12.0)).astype(int)
    skeleton[rows, columns] = True

    path = order_centreline(zhang_suen_thinning(skeleton))
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)

    assert len(path) >= 45
    assert float(steps.max()) <= np.sqrt(2.0) + 1e-9


def test_order_centreline_rejects_an_empty_skeleton() -> None:
    with pytest.raises(ValueError, match="empty"):
        order_centreline(np.zeros((5, 5), dtype=bool))


def test_smoothing_preserves_the_endpoints() -> None:
    generator = np.random.default_rng(3)
    path = np.stack((np.arange(40.0), 20 + generator.normal(0, 1, 40)), axis=-1)

    smoothed = smooth_centreline(path, window=9)

    np.testing.assert_array_equal(smoothed[0], path[0])
    np.testing.assert_array_equal(smoothed[-1], path[-1])
    assert float(np.std(np.diff(smoothed[:, 1]))) < float(np.std(np.diff(path[:, 1])))


def test_smoothing_rejects_an_even_window() -> None:
    with pytest.raises(ValueError, match="odd"):
        smooth_centreline(np.zeros((10, 2)), window=8)


def test_resampling_gives_a_regular_arc_length_step() -> None:
    path = np.stack((np.zeros(11), np.arange(11.0)), axis=-1)

    resampled = resample_polyline(path, spacing_pixels=2.5)
    steps = np.linalg.norm(np.diff(resampled, axis=0), axis=1)

    np.testing.assert_allclose(steps, 2.5, atol=1e-9)


def test_normals_are_unit_and_perpendicular() -> None:
    path = np.stack((np.arange(20.0), 2.0 * np.arange(20.0)), axis=-1)

    tangents, normals = tangents_and_normals(path)

    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(tangents * normals, axis=1), 0.0, atol=1e-12)


def test_corridor_covers_the_declared_half_width() -> None:
    path = np.stack((np.full(30, 15.0), np.arange(5.0, 35.0)), axis=-1)

    corridor = band_corridor((30, 40), path, half_width_pixels=3.0)

    assert corridor[15, 20]
    assert corridor[18, 20]
    assert not corridor[19, 20]
