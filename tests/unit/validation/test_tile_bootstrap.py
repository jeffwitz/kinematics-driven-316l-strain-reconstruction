from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.tile_bootstrap import (
    TileDesign,
    bootstrap_defects,
    defects_on,
    prepare_fields,
    tile_indices,
)

SHAPE = (120, 98)
SCALE = 24


def _pair(factor: float = 1.3, noise: float = 0.05, seed: int = 0):
    generator = np.random.default_rng(seed)
    reference = np.abs(generator.normal(1.0, 0.3, SHAPE))
    candidate = factor * reference + noise * generator.normal(size=SHAPE)
    return prepare_fields("c", candidate, reference, scale_pixels=SCALE)


def test_tiles_cover_whole_squares_only() -> None:
    tiles = tile_indices((100, 70), tile_pixels=24)

    # 4 whole rows of tiles by 2 whole columns; the remainder is dropped
    # rather than resampled at a different weight.
    assert len(tiles) == 4 * 2
    assert all(tile.size == 24 * 24 for tile in tiles)
    assert len(set(np.concatenate(tiles).tolist())) == 8 * 24 * 24


def test_a_tile_larger_than_the_support_is_an_error() -> None:
    with pytest.raises(ValueError, match="no whole tile"):
        tile_indices((10, 10), tile_pixels=24)


def test_the_tile_aggregation_reproduces_the_pixel_computation_exactly() -> None:
    """The optimisation must not change a single number.

    Shape, presence and localisation are rebuilt from per-tile partial sums
    instead of the pixels; that is algebraically exact and this pins it.
    """

    fields = _pair()
    tiles = tile_indices(SHAPE, tile_pixels=SCALE)
    design = TileDesign(tile_pixels=SCALE, draws=1, seed=1)

    # Reproduce the single draw the bootstrap will make with this seed.
    generator = np.random.default_rng(design.seed)
    drawn = generator.integers(0, len(tiles), size=len(tiles))
    index = np.concatenate([tiles[position] for position in drawn])

    expected = defects_on(fields, index)
    obtained = bootstrap_defects({"c": fields}, shape=SHAPE, design=design)

    for name, value in expected.items():
        assert obtained["c"][name][0] == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_an_identical_candidate_scores_zero_on_every_draw() -> None:
    generator = np.random.default_rng(3)
    reference = np.abs(generator.normal(1.0, 0.3, SHAPE))
    fields = prepare_fields("same", reference, reference, scale_pixels=SCALE)

    samples = bootstrap_defects({"same": fields}, shape=SHAPE, design=TileDesign(SCALE, 20, 5))

    for name, values in samples["same"].items():
        assert np.allclose(values, 0.0, atol=1e-12), name


def test_candidates_are_paired_on_the_same_draw() -> None:
    """A shared draw is what makes a difference between two candidates mean
    anything: otherwise it mixes model difference with resampling difference."""

    first = _pair(factor=1.3, seed=0)
    second = prepare_fields(
        "d",
        1.3 * np.asarray(first.reference).reshape(SHAPE),
        np.asarray(first.reference).reshape(SHAPE),
        scale_pixels=SCALE,
    )
    design = TileDesign(SCALE, 30, 11)

    together = bootstrap_defects({"c": first, "d": second}, shape=SHAPE, design=design)
    alone = bootstrap_defects({"d": second}, shape=SHAPE, design=design)

    # The same seed and the same tiles must give the same draws for d whether
    # or not c is scored alongside it.
    np.testing.assert_allclose(together["d"]["D_presence"], alone["d"]["D_presence"], rtol=1e-12)


def test_the_design_is_reproducible_from_its_seed() -> None:
    fields = _pair()
    design = TileDesign(SCALE, 25, 20260801)

    first = bootstrap_defects({"c": fields}, shape=SHAPE, design=design)
    second = bootstrap_defects({"c": fields}, shape=SHAPE, design=design)

    for name in first["c"]:
        np.testing.assert_array_equal(first["c"][name], second["c"][name])


def test_a_larger_tile_does_not_pretend_to_more_independence() -> None:
    """Coarser tiles mean fewer independent units and a wider spread.

    The registered tile is 49 px because 8 px sits far below the measured
    38.2 px coherence; this is the property that makes the choice matter.
    """

    fields = _pair(noise=0.3, seed=4)
    fine = bootstrap_defects({"c": fields}, shape=SHAPE, design=TileDesign(14, 300, 2))["c"][
        "D_shape"
    ]
    coarse = bootstrap_defects({"c": fields}, shape=SHAPE, design=TileDesign(28, 300, 2))["c"][
        "D_shape"
    ]

    assert float(np.std(coarse)) > float(np.std(fine))
