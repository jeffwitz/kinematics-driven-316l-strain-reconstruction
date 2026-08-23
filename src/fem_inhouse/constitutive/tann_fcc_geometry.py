"""EBSD interface of the TANN-FCC: orientations -> specimen-frame systems.

One geometry, shared with the SRIX/Meric convention: the twelve octahedral
systems of `fem_inhouse.core.fcc_interaction_matrix.SLIP_SYSTEMS`, rotated
per material point from the Bunge angles of the qualified EBSD map. No
second FCC convention exists here; the coherence test ties
`tau^alpha = sigma : P^alpha` to the canonical closed form.

Non-indexed EBSD pixels (the `1449` sentinel in all three angles, and any
orientation whose archived maximum Schmid factor falls outside
`[0.2722, 0.5]`) are refilled from their nearest valid neighbour on the
node grid before element averaging, carrying a validity flag alongside --
the strategy qualified in the earlier campaigns (`schmid_channels`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS

if TYPE_CHECKING:
    pass

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

EBSD_SENTINEL = 1449
#: The largest of the twelve Schmid factors lies in this interval for any
#: FCC orientation; a value outside it marks a non-indexed pixel.
MAX_SCHMID_RANGE = (0.2722, 0.5)


def material_slip_tensors() -> FloatArray:
    """The twelve normalised Schmid tensors in the material frame."""

    tensors = np.empty((12, 3, 3), dtype=np.float64)
    for index, (burgers, normal) in enumerate(SLIP_SYSTEMS):
        s = np.asarray(burgers, dtype=np.float64)
        m = np.asarray(normal, dtype=np.float64)
        s /= np.linalg.norm(s)
        m /= np.linalg.norm(m)
        tensors[index] = 0.5 * (np.outer(s, m) + np.outer(m, s))
    return tensors


def _nearest_valid_fill(node_field: FloatArray, validity: BoolArray) -> FloatArray:
    """Refill invalid nodes from the nearest valid neighbour (scipy EDT)."""

    from scipy.ndimage import distance_transform_edt

    filled = node_field.copy()
    if validity.all():
        return filled
    invalid = ~validity
    indices = distance_transform_edt(
        invalid, return_distances=False, return_indices=True
    )  # (ndim, nx+1, ny+1)
    filled[invalid] = filled[tuple(indices[:, invalid])]
    return filled


def systems_from_bunge_node_grid(
    angles_bunge_deg: FloatArray,
    *,
    max_schmid_factor: FloatArray | None = None,
) -> tuple[FloatArray, BoolArray]:
    """Per-node rotations to per-point specimen-frame systems.

    `angles_bunge_deg` is the `(nx + 1, ny + 1, 3)` node map of Bunge
    angles. Returns `(systems, validity)` with `systems` of shape
    `(nx * ny * 2, 12, 3, 3)` -- one tensor triple per material point, the
    two subcells of a pixel sharing the pixel orientation (the repository
    convention) -- and `validity` the per-material-point flag of the
    original (pre-fill) mask.
    """

    angles = np.asarray(angles_bunge_deg, dtype=np.float64)
    if angles.ndim != 3 or angles.shape[-1] != 3:
        raise ValueError(f"expected a (nx+1, ny+1, 3) Bunge map, got {angles.shape}")
    sentinel = np.all(np.isclose(angles, EBSD_SENTINEL), axis=-1)
    validity = ~sentinel
    if max_schmid_factor is not None:
        schmid = np.asarray(max_schmid_factor, dtype=np.float64)
        if schmid.shape != angles.shape[:2]:
            raise ValueError("max_schmid_factor must share the node grid shape")
        lower, upper = MAX_SCHMID_RANGE
        validity &= (schmid >= lower) & (schmid <= upper) & np.isfinite(schmid)

    filled_angles = _nearest_valid_fill(angles, validity)
    rotations = rotations_from_euler_bunge_deg(filled_angles.reshape(-1, 3))
    material_to_global = np.swapaxes(rotations, 1, 2)  # Q^T

    material = material_slip_tensors()
    node_systems = np.einsum(
        "pia,cab,pjb->pijc", material_to_global, material, material_to_global
    )  # (nodes, 3, 3, 12)
    node_systems = np.moveaxis(node_systems, -1, 1)  # (nodes, 12, 3, 3)

    nx, ny = angles.shape[0] - 1, angles.shape[1] - 1
    node_grid = node_systems.reshape(nx + 1, ny + 1, 12, 3, 3)
    element_mean = 0.25 * (
        node_grid[1:, 1:] + node_grid[:-1, :-1] + node_grid[1:, :-1] + node_grid[:-1, 1:]
    )  # (nx, ny, 12, 3, 3)
    systems = np.repeat(element_mean[:, :, None, :, :], 2, axis=2).reshape(-1, 12, 3, 3)
    # An element is valid iff its four corners are; both subcells inherit.
    element_valid = validity[:-1, :-1] & validity[1:, 1:] & validity[1:, :-1] & validity[:-1, 1:]
    element_validity = np.repeat(element_valid[:, :, None], 2, axis=2).reshape(-1)
    return systems, element_validity
