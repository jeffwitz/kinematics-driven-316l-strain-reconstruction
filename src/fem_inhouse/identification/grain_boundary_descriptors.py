"""Geometry-only descriptors for segmented FCC grain maps.

The functions in this module deliberately stop at geometric descriptors.  They
do not infer a constitutive law or claim that a grain-boundary metric predicts
slip transfer.  A caller must provide a defensible grain-label map; an Euler
angle field alone is not treated as a segmentation.
"""

from __future__ import annotations

from itertools import permutations, product
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import distance_transform_edt

from fem_inhouse.core.crystal_orientation import validate_rotations
from fem_inhouse.core.fcc_interaction_matrix import slip_systems

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def validate_grain_ids(grain_ids: ArrayLike) -> IntArray:
    """Return a two-dimensional integer grain-label map."""

    values = np.asarray(grain_ids)
    if values.ndim != 2:
        raise ValueError(f"grain_ids must be two-dimensional, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("grain_ids must be finite")
    if not np.equal(values, np.rint(values)).all():
        raise ValueError("grain_ids must contain integer labels")
    return np.asarray(values, dtype=int)


def grain_area_map(grain_ids: ArrayLike) -> dict[int, int]:
    """Return pixel area for every non-negative grain label."""

    labels = validate_grain_ids(grain_ids)
    return {
        int(label): int(count)
        for label, count in zip(*np.unique(labels[labels >= 0], return_counts=True), strict=True)
    }


def equivalent_diameter_map(grain_ids: ArrayLike) -> FloatArray:
    """Return the equivalent 2-D diameter at every labelled pixel."""

    labels = validate_grain_ids(grain_ids)
    areas = grain_area_map(labels)
    output = np.full(labels.shape, np.nan, dtype=np.float64)
    for label, area in areas.items():
        output[labels == label] = 2.0 * np.sqrt(area / np.pi)
    return output


def boundary_mask(grain_ids: ArrayLike) -> NDArray[np.bool_]:
    """Mark pixels adjacent to a different valid grain label."""

    labels = validate_grain_ids(grain_ids)
    valid = labels >= 0
    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in (0, 1):
        for shift in (-1, 1):
            neighbour = np.roll(labels, shift, axis=axis)
            edge = np.roll(valid, shift, axis=axis)
            boundary |= valid & edge & (labels != neighbour)
    boundary[[0, -1], :] |= valid[[0, -1], :]
    boundary[:, [0, -1]] |= valid[:, [0, -1]]
    return boundary


def distance_to_boundary(grain_ids: ArrayLike) -> FloatArray:
    """Return distance in label pixels to the nearest grain boundary."""

    labels = validate_grain_ids(grain_ids)
    valid = labels >= 0
    boundary = boundary_mask(labels)
    distance = distance_transform_edt(~boundary)
    return np.where(valid, distance, np.nan)


def neighbour_pairs(grain_ids: ArrayLike) -> tuple[dict[str, Any], ...]:
    """Return deterministic four-connected grain-boundary contacts."""

    labels = validate_grain_ids(grain_ids)
    contacts: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for axis in (0, 1):
        source = np.take(labels, indices=range(labels.shape[axis] - 1), axis=axis)
        target = np.take(labels, indices=range(1, labels.shape[axis]), axis=axis)
        differing = (source >= 0) & (target >= 0) & (source != target)
        for index in np.argwhere(differing):
            coordinate = index.tolist()
            first = int(source[tuple(coordinate)])
            second = int(target[tuple(coordinate)])
            pair = tuple(sorted((first, second)))
            if axis == 0:
                point = (int(coordinate[0]), int(coordinate[1]))
            else:
                point = (int(coordinate[0]), int(coordinate[1]))
            contacts.setdefault(pair, []).append(point)
    return tuple(
        {
            "grain_a": pair[0],
            "grain_b": pair[1],
            "contact_pixels": len(points),
            "points": tuple(points),
        }
        for pair, points in sorted(contacts.items())
    )


def normalize_descriptor(values: ArrayLike, *, valid: ArrayLike | None = None) -> FloatArray:
    """Center and RMS-normalize a scalar descriptor on a declared support."""

    array = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(array)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if not np.any(mask):
        raise ValueError("descriptor has no finite support")
    mean = float(np.mean(array[mask]))
    centered = array - mean
    scale = float(np.sqrt(np.mean(centered[mask] ** 2)))
    if scale == 0.0:
        raise ValueError("descriptor is constant on its support")
    return np.where(mask, centered / scale, np.nan)


def cubic_symmetry_matrices() -> NDArray[np.float64]:
    """Return the 24 proper signed-permutation cubic symmetries."""

    matrices: list[np.ndarray] = []
    for permutation_axes in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            matrix[range(3), permutation_axes] = signs
            if np.linalg.det(matrix) > 0.0:
                matrices.append(matrix)
    return np.asarray(matrices)


def cubic_misorientation_angle(
    first_global_to_material: ArrayLike,
    second_global_to_material: ArrayLike,
) -> float:
    """Return the minimum cubic-symmetry misorientation angle in degrees."""

    first = validate_rotations(np.asarray(first_global_to_material)[None, ...])[0]
    second = validate_rotations(np.asarray(second_global_to_material)[None, ...])[0]
    relative = first @ second.T
    traces = np.trace(
        np.einsum("sij,jk->sik", cubic_symmetry_matrices(), relative),
        axis1=1,
        axis2=2,
    )
    cosine = np.clip((np.max(traces) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotated_fcc_slip_systems(global_to_material: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return normalized specimen-frame slip directions and normals."""

    rotation = validate_rotations(np.asarray(global_to_material)[None, ...])[0]
    systems = slip_systems()
    directions = np.asarray([system.burgers for system in systems], dtype=np.float64)
    normals = np.asarray([system.normal for system in systems], dtype=np.float64)
    material_to_global = rotation.T
    directions = directions @ material_to_global.T
    normals = normals @ material_to_global.T
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return directions, normals


def luster_morris_matrix(
    first_global_to_material: ArrayLike,
    second_global_to_material: ArrayLike,
) -> FloatArray:
    """Return the 12x12 geometric Luster--Morris compatibility matrix."""

    first_directions, first_normals = rotated_fcc_slip_systems(first_global_to_material)
    second_directions, second_normals = rotated_fcc_slip_systems(second_global_to_material)
    matrix = np.abs(first_normals @ second_normals.T) * np.abs(
        first_directions @ second_directions.T
    )
    return np.clip(matrix, 0.0, 1.0)


def residual_burgers_matrix(
    first_global_to_material: ArrayLike,
    second_global_to_material: ArrayLike,
) -> FloatArray:
    """Return sign-invariant normalized residual Burgers magnitudes."""

    first_directions, _ = rotated_fcc_slip_systems(first_global_to_material)
    second_directions, _ = rotated_fcc_slip_systems(second_global_to_material)
    difference = np.linalg.norm(
        first_directions[:, None, :] - second_directions[None, :, :], axis=-1
    )
    sum_norm = np.linalg.norm(
        first_directions[:, None, :] + second_directions[None, :, :], axis=-1
    )
    return np.minimum(difference, sum_norm)
