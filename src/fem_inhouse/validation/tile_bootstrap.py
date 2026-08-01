"""Paired spatial tile bootstrap for the P43 selection defects.

Protocol: `validation/p0043_small_parameter_matrix_preregistration.md`,
amendment A2. Tiles are squares of the principal scale rather than the 8-unit
blocks of the earlier section bootstrap: this resampling is two-dimensional and
an 8 px tile sits far below the measured 38.2 px coherence of the observable,
so it would treat correlated pixels as independent and understate the
uncertainty.

Every defect is recomputed exactly on the drawn pixels. Nothing is
approximated: shape is a correlation, amplitude a quantile, presence a ratio of
sums of squares, and localisation a ratio of spatial means of the fraction
fields, which are built once on the whole core at the registered neighbourhood
size and only then averaged over the drawn tiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.validation.fractions_skill_score import active_fraction_field

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class TileDesign:
    """The registered resampling design."""

    tile_pixels: int
    draws: int
    seed: int

    def as_dict(self) -> dict[str, int]:
        return {"tile_pixels": self.tile_pixels, "draws": self.draws, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class PairedFields:
    """Everything one candidate needs, precomputed on the whole core."""

    label: str
    candidate: FloatArray
    reference: FloatArray
    candidate_fraction: FloatArray
    reference_fraction: FloatArray


def tile_indices(shape: tuple[int, int], *, tile_pixels: int) -> list[IntArray]:
    """Flat pixel indices of each whole square tile of the core.

    Partial tiles at the far edge are dropped rather than resampled at a
    different weight: a half tile carries half the pixels and would bias every
    draw it appears in.
    """

    if tile_pixels < 1:
        raise ValueError("tile_pixels must be positive")
    rows, columns = shape
    flat = np.arange(rows * columns, dtype=np.intp).reshape(shape)
    tiles: list[IntArray] = []
    for row in range(0, rows - tile_pixels + 1, tile_pixels):
        for column in range(0, columns - tile_pixels + 1, tile_pixels):
            block = flat[row : row + tile_pixels, column : column + tile_pixels]
            tiles.append(np.ascontiguousarray(block.ravel()))
    if not tiles:
        raise ValueError("no whole tile fits in the support")
    return tiles


def prepare_fields(
    label: str,
    candidate: FloatArray,
    reference: FloatArray,
    *,
    scale_pixels: int,
    threshold_quantile: float = 0.90,
) -> PairedFields:
    """Precompute the maps every draw reuses, including the fraction fields."""

    threshold = float(np.quantile(reference, threshold_quantile))
    return PairedFields(
        label=label,
        candidate=np.ascontiguousarray(candidate, dtype=np.float64).ravel(),
        reference=np.ascontiguousarray(reference, dtype=np.float64).ravel(),
        candidate_fraction=active_fraction_field(
            candidate >= threshold, scale_pixels=scale_pixels
        ).ravel(),
        reference_fraction=active_fraction_field(
            reference >= threshold, scale_pixels=scale_pixels
        ).ravel(),
    )


def defects_on(fields: PairedFields, index: IntArray) -> dict[str, float]:
    """The four defects, recomputed on one drawn pixel multiset."""

    candidate = fields.candidate[index]
    reference = fields.reference[index]

    centred_candidate = candidate - candidate.mean()
    centred_reference = reference - reference.mean()
    denominator = float(np.sqrt(np.sum(centred_candidate**2) * np.sum(centred_reference**2)))
    shape = (
        1.0 - float(np.sum(centred_candidate * centred_reference) / denominator)
        if denominator > 0.0
        else float("nan")
    )

    numerator = float(np.quantile(candidate, 0.95))
    reference_quantile = float(np.quantile(reference, 0.95))
    amplitude = (
        abs(float(np.log(numerator / reference_quantile)))
        if numerator > 0.0 and reference_quantile > 0.0
        else float("nan")
    )

    candidate_energy = float(np.sum(candidate**2))
    reference_energy = float(np.sum(reference**2))
    presence = (
        abs(float(np.log(candidate_energy / reference_energy)))
        if candidate_energy > 0.0 and reference_energy > 0.0
        else float("nan")
    )

    fraction_candidate = fields.candidate_fraction[index]
    fraction_reference = fields.reference_fraction[index]
    mean_square_error = float(np.mean((fraction_candidate - fraction_reference) ** 2))
    mean_square_total = float(np.mean(fraction_candidate**2 + fraction_reference**2))
    localisation = (
        mean_square_error / mean_square_total if mean_square_total > 0.0 else float("nan")
    )

    return {
        "D_shape": shape,
        "D_amplitude": amplitude,
        "D_localisation": localisation,
        "D_presence": presence,
    }


def _tile_aggregates(fields: PairedFields, tiles: list[IntArray]) -> dict[str, FloatArray]:
    """Per-tile partial sums, so a draw costs O(tiles) instead of O(pixels).

    Exact, not approximate: shape, presence and localisation are all built from
    sums, and a sum over a multiset of tiles is the weighted sum of the tile
    sums. Only the amplitude quantile still needs the pixels themselves.
    """

    names = (
        "count",
        "candidate",
        "reference",
        "candidate_square",
        "reference_square",
        "cross",
        "fraction_error_square",
        "fraction_total_square",
        "candidate_energy",
        "reference_energy",
    )
    aggregates = {name: np.empty(len(tiles), dtype=np.float64) for name in names}
    for position, index in enumerate(tiles):
        candidate = fields.candidate[index]
        reference = fields.reference[index]
        fraction_candidate = fields.candidate_fraction[index]
        fraction_reference = fields.reference_fraction[index]
        aggregates["count"][position] = candidate.size
        aggregates["candidate"][position] = candidate.sum()
        aggregates["reference"][position] = reference.sum()
        aggregates["candidate_square"][position] = np.sum(candidate**2)
        aggregates["reference_square"][position] = np.sum(reference**2)
        aggregates["cross"][position] = np.sum(candidate * reference)
        aggregates["fraction_error_square"][position] = np.sum(
            (fraction_candidate - fraction_reference) ** 2
        )
        aggregates["fraction_total_square"][position] = np.sum(
            fraction_candidate**2 + fraction_reference**2
        )
        aggregates["candidate_energy"][position] = aggregates["candidate_square"][position]
        aggregates["reference_energy"][position] = aggregates["reference_square"][position]
    return aggregates


def _defects_from_aggregates(
    aggregates: dict[str, FloatArray],
    weights: FloatArray,
) -> dict[str, float]:
    total = float(weights @ aggregates["count"])
    candidate = float(weights @ aggregates["candidate"])
    reference = float(weights @ aggregates["reference"])
    candidate_square = float(weights @ aggregates["candidate_square"])
    reference_square = float(weights @ aggregates["reference_square"])
    cross = float(weights @ aggregates["cross"])

    covariance = cross - candidate * reference / total
    candidate_variance = candidate_square - candidate**2 / total
    reference_variance = reference_square - reference**2 / total
    denominator = float(np.sqrt(candidate_variance * reference_variance))
    shape = 1.0 - covariance / denominator if denominator > 0.0 else float("nan")

    error = float(weights @ aggregates["fraction_error_square"])
    whole = float(weights @ aggregates["fraction_total_square"])
    localisation = error / whole if whole > 0.0 else float("nan")

    presence = (
        abs(float(np.log(candidate_square / reference_square)))
        if candidate_square > 0.0 and reference_square > 0.0
        else float("nan")
    )
    return {"D_shape": shape, "D_localisation": localisation, "D_presence": presence}


def bootstrap_defects(
    fields: dict[str, PairedFields],
    *,
    shape: tuple[int, int],
    design: TileDesign,
) -> dict[str, dict[str, FloatArray]]:
    """Resample tiles once per draw and score every candidate on the same draw.

    Paired on purpose: all candidates see the identical tile multiset in a given
    draw, so a difference between two of them is not contaminated by a
    difference between two resamplings.
    """

    tiles = tile_indices(shape, tile_pixels=design.tile_pixels)
    aggregates = {label: _tile_aggregates(payload, tiles) for label, payload in fields.items()}
    generator = np.random.default_rng(design.seed)
    names = ("D_shape", "D_amplitude", "D_localisation", "D_presence")
    samples: dict[str, dict[str, list[float]]] = {
        label: {name: [] for name in names} for label in fields
    }
    tile_count = len(tiles)
    for _ in range(design.draws):
        drawn = generator.integers(0, tile_count, size=tile_count)
        weights = np.bincount(drawn, minlength=tile_count).astype(np.float64)
        index = np.concatenate([tiles[position] for position in drawn])
        # The reference is shared by every candidate in a draw, so its quantile
        # is computed once rather than seventeen times.
        first = next(iter(fields.values()))
        reference_quantile = float(np.quantile(first.reference[index], 0.95))
        for label, payload in fields.items():
            values = _defects_from_aggregates(aggregates[label], weights)
            candidate_quantile = float(np.quantile(payload.candidate[index], 0.95))
            values["D_amplitude"] = (
                abs(float(np.log(candidate_quantile / reference_quantile)))
                if candidate_quantile > 0.0 and reference_quantile > 0.0
                else float("nan")
            )
            for name in names:
                samples[label][name].append(values[name])
    return {
        label: {name: np.asarray(values, dtype=np.float64) for name, values in per_defect.items()}
        for label, per_defect in samples.items()
    }
