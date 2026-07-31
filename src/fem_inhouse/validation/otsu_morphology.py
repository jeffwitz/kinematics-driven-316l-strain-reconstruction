"""Otsu segmentation of the DIC field, frozen, with regionprops morphology.

The threshold is chosen by the data rather than by picking a quantile: Otsu
maximises between-class variance, so nothing has to be argued about why 80 or
90 per cent. It is computed **once on the DIC** and applied unchanged to every
candidate, so no candidate can move the boundary it is judged against.

Morphology comes from `skimage.measure.regionprops`, a standard and tested
descriptor set, rather than from hand-rolled skeleton statistics.

Why morphology and not area: on this ROI the active fraction of the DIC and of
the translated-map control agree to within one point, `26.2 %` against
`27.0 %`, while their shapes are unmistakably different — two elongated bands
against one cellular object. An area-based score cannot separate them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

FloatArray = NDArray[np.float64]

#: Objects below this area are not asserted as bands. Four by sixty-four
#: pixels: narrower in one direction than the chain's MTF-50 can resolve.
DEFAULT_MINIMUM_AREA_PIXELS = 256


@dataclass(frozen=True, slots=True)
class ObjectMorphology:
    """Standard regionprops descriptors of one segmented object."""

    rank: int
    area_pixels: int
    perimeter_pixels: float
    eccentricity: float
    solidity: float
    extent: float
    axis_major_pixels: float
    axis_minor_pixels: float
    orientation_degrees: float
    euler_number: int
    centroid: tuple[float, float]


@dataclass(frozen=True, slots=True)
class FieldMorphology:
    """The segmentation of one field under a frozen threshold."""

    label: str
    threshold: float
    active_fraction: float
    object_count: int
    objects: tuple[ObjectMorphology, ...]


def otsu_threshold(
    field: NDArray[np.generic],
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Return the Otsu threshold of the valid values of a field."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must be two-dimensional")
    if valid_mask is None:
        selected = values.ravel()
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match the field shape")
        selected = values[mask]
    if selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("the valid field must be finite and non-empty")
    if float(np.ptp(selected)) == 0.0:
        raise ValueError("a constant field has no Otsu threshold")
    return float(threshold_otsu(selected))


def describe_morphology(
    field: NDArray[np.generic],
    *,
    threshold: float,
    label_name: str,
    minimum_area_pixels: int = DEFAULT_MINIMUM_AREA_PIXELS,
    valid_mask: NDArray[np.bool_] | None = None,
) -> FieldMorphology:
    """Segment a field at a **given** threshold and describe every object.

    The threshold is an argument, never recomputed here: the whole point is
    that one boundary derived from the DIC is applied to all fields.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must be two-dimensional")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if minimum_area_pixels < 1:
        raise ValueError("minimum_area_pixels must be positive")
    active = values >= threshold
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match the field shape")
        active &= mask
        denominator = float(np.count_nonzero(mask))
    else:
        denominator = float(values.size)

    labels = label(active, connectivity=2)
    regions = sorted(
        (r for r in regionprops(labels) if r.area >= minimum_area_pixels),
        key=lambda r: -r.area,
    )
    objects = tuple(
        ObjectMorphology(
            rank=index,
            area_pixels=int(region.area),
            perimeter_pixels=float(region.perimeter),
            eccentricity=float(region.eccentricity),
            solidity=float(region.solidity),
            extent=float(region.extent),
            axis_major_pixels=float(region.axis_major_length),
            axis_minor_pixels=float(region.axis_minor_length),
            orientation_degrees=float(np.degrees(region.orientation)),
            euler_number=int(region.euler_number),
            centroid=(float(region.centroid[0]), float(region.centroid[1])),
        )
        for index, region in enumerate(regions, start=1)
    )
    return FieldMorphology(
        label=label_name,
        threshold=float(threshold),
        active_fraction=float(np.count_nonzero(active) / denominator),
        object_count=len(objects),
        objects=objects,
    )


def morphology_distance(
    reference: FieldMorphology,
    candidate: FieldMorphology,
) -> dict[str, float]:
    """Compare two segmentations object by object, largest first.

    Reports the count mismatch separately from the shape differences, because a
    candidate producing the wrong number of objects has already failed in a way
    no per-object difference can express.
    """

    if reference.threshold != candidate.threshold:
        raise ValueError("both fields must be segmented at the same threshold")
    nan = float("nan")
    paired = min(len(reference.objects), len(candidate.objects))
    result: dict[str, float] = {
        "object_count_reference": float(reference.object_count),
        "object_count_candidate": float(candidate.object_count),
        "object_count_difference": float(candidate.object_count - reference.object_count),
        "active_fraction_reference": reference.active_fraction,
        "active_fraction_candidate": candidate.active_fraction,
        "active_fraction_ratio": (
            candidate.active_fraction / reference.active_fraction
            if reference.active_fraction > 0.0
            else nan
        ),
    }
    if paired == 0:
        for key in (
            "eccentricity_error",
            "axis_minor_ratio",
            "axis_major_ratio",
            "orientation_error_degrees",
            "solidity_error",
        ):
            result[key] = nan
        return result

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else nan

    pairs = list(zip(reference.objects[:paired], candidate.objects[:paired], strict=True))
    result["eccentricity_error"] = _mean([abs(c.eccentricity - r.eccentricity) for r, c in pairs])
    result["solidity_error"] = _mean([abs(c.solidity - r.solidity) for r, c in pairs])
    result["axis_minor_ratio"] = _mean(
        [c.axis_minor_pixels / r.axis_minor_pixels for r, c in pairs if r.axis_minor_pixels > 0]
    )
    result["axis_major_ratio"] = _mean(
        [c.axis_major_pixels / r.axis_major_pixels for r, c in pairs if r.axis_major_pixels > 0]
    )
    # Orientation is modulo 180 degrees: a band has no head or tail.
    result["orientation_error_degrees"] = _mean(
        [
            float(
                abs((c.orientation_degrees - r.orientation_degrees + 90.0) % 180.0 - 90.0)
            )
            for r, c in pairs
        ]
    )
    return result
