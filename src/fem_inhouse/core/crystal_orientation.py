"""Crystallographic orientations for the material points of a batch.

## The convention, stated once

`Q_global_to_material` maps the components of a vector from the specimen frame
to the crystal frame. A strain and the stress it produces therefore transform as

    eps_crystal = Q eps_global Q^T,      sigma_global = Q^T sigma_crystal Q.

The identity means the crystal axes are aligned with the specimen axes.

## MGIS wants the other one

Measured against a hand-rotated cubic stiffness, the rotation MGIS expects is
the **material-to-global** matrix, flattened row-major, which is the transpose
of the convention above. Passing the wrong one is not detectable by inspection:
it produces a plausible field rotated the wrong way. `mgis_rotation_argument`
is the only place that transpose is applied, and
`test_crystal_orientation.py` pins it against an analytical rotation of the
cubic stiffness rather than against MGIS itself.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Departure from orthogonality tolerated in a rotation matrix.
#:
#: Orientations reaching this module come from EBSD indexing or from hand-typed
#: configuration, so they are rounded rather than exact. This is loose enough
#: for six-decimal input and far too tight for a matrix that is not a rotation.
ORTHOGONALITY_TOLERANCE = 1e-8


@runtime_checkable
class OrientationProvider(Protocol):
    """Supplies one `Q_global_to_material` per material point."""

    def rotations_global_to_material(self, point_count: int) -> NDArray:
        """Return a `(point_count, 3, 3)` array of rotation matrices."""
        ...


def validate_rotations(rotations: ArrayLike, *, point_count: int | None = None) -> NDArray:
    """Check a rotation array and return it as a fresh contiguous float array.

    Rejects reflections as well as non-orthogonal matrices: a determinant of
    `-1` passes every orthogonality check and silently mirrors the crystal,
    which would swap the handedness of the slip systems.
    """

    array = np.array(rotations, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(
            f"rotations must have shape (n_points, 3, 3), got {array.shape}"
        )
    if point_count is not None and array.shape[0] != point_count:
        raise ValueError(
            f"expected {point_count} orientations, got {array.shape[0]}"
        )
    if not np.isfinite(array).all():
        raise ValueError("rotations must be finite")

    products = np.einsum("nij,nkj->nik", array, array)
    departure = np.abs(products - np.eye(3)).max(axis=(1, 2))
    worst = int(np.argmax(departure))
    if departure[worst] > ORTHOGONALITY_TOLERANCE:
        raise ValueError(
            f"rotation {worst} is not orthogonal: max |Q Q^T - I| is "
            f"{departure[worst]:.3e}, tolerance {ORTHOGONALITY_TOLERANCE:.1e}"
        )

    determinants = np.linalg.det(array)
    reflections = np.flatnonzero(determinants < 0.0)
    if reflections.size:
        raise ValueError(
            f"rotation {int(reflections[0])} is a reflection (determinant "
            f"{determinants[reflections[0]]:+.3f}); a mirrored crystal would "
            "reverse the handedness of the slip systems"
        )
    unit = np.abs(determinants - 1.0).max()
    if unit > ORTHOGONALITY_TOLERANCE:
        raise ValueError(f"rotation determinants depart from unity by {unit:.3e}")
    return array


def mgis_rotation_argument(rotations_global_to_material: ArrayLike) -> NDArray:
    """Convert to the flat material-to-global layout MGIS consumes.

    The single point where the convention of this module is reconciled with the
    one MGIS uses. Nothing else in the codebase may transpose an orientation.
    """

    validated = validate_rotations(rotations_global_to_material)
    material_to_global = np.swapaxes(validated, 1, 2)
    return np.ascontiguousarray(material_to_global.reshape(validated.shape[0] * 9))


def rotation_from_euler_bunge_deg(phi1: float, capital_phi: float, phi2: float) -> NDArray:
    """`Q_global_to_material` from Bunge ZXZ Euler angles, in degrees.

    The Bunge convention is the one EBSD systems export, so this is the form the
    orientation maps will arrive in.
    """

    first, second, third = (math.radians(angle) for angle in (phi1, capital_phi, phi2))
    c1, s1 = math.cos(first), math.sin(first)
    c2, s2 = math.cos(second), math.sin(second)
    c3, s3 = math.cos(third), math.sin(third)
    return np.array(
        [
            [c1 * c3 - s1 * s3 * c2, s1 * c3 + c1 * s3 * c2, s3 * s2],
            [-c1 * s3 - s1 * c3 * c2, -s1 * s3 + c1 * c3 * c2, c3 * s2],
            [s1 * s2, -c1 * s2, c2],
        ]
    )


def rotations_from_euler_bunge_deg(angles: ArrayLike) -> NDArray:
    """Convert an array of Bunge angles with final axis ``(phi1, Phi, phi2)``."""

    values = np.asarray(angles, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("Bunge angle maps must have final shape (3)")
    if not np.isfinite(values).all():
        raise ValueError("Bunge angle maps must be finite")
    phi1, capital_phi, phi2 = np.deg2rad(np.moveaxis(values, -1, 0))
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c2, s2 = np.cos(capital_phi), np.sin(capital_phi)
    c3, s3 = np.cos(phi2), np.sin(phi2)
    rotations = np.empty((*values.shape[:-1], 3, 3), dtype=float)
    rotations[..., 0, 0] = c1 * c3 - s1 * s3 * c2
    rotations[..., 0, 1] = s1 * c3 + c1 * s3 * c2
    rotations[..., 0, 2] = s3 * s2
    rotations[..., 1, 0] = -c1 * s3 - s1 * c3 * c2
    rotations[..., 1, 1] = -s1 * s3 + c1 * c3 * c2
    rotations[..., 1, 2] = c3 * s2
    rotations[..., 2, 0] = s1 * s2
    rotations[..., 2, 1] = -c1 * s2
    rotations[..., 2, 2] = c2
    return rotations


class HomogeneousOrientationProvider:
    """One orientation shared by every material point.

    A validation step, not a polycrystal: every point being the same crystal is
    exactly what a real aggregate is not. It exists so the rotation plumbing can
    be exercised before an EBSD map is wired in, and so that a non-trivial
    orientation can be shown to change the answer.
    """

    def __init__(self, rotation_global_to_material: ArrayLike) -> None:
        rotation = np.array(rotation_global_to_material, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(
                f"a homogeneous orientation is one 3x3 matrix, got {rotation.shape}"
            )
        self._rotation = validate_rotations(rotation[None, :, :])[0]

    @property
    def rotation_global_to_material(self) -> NDArray:
        return self._rotation.copy()

    @classmethod
    def identity(cls) -> HomogeneousOrientationProvider:
        """Crystal axes aligned with the specimen axes."""

        return cls(np.eye(3))

    @classmethod
    def from_euler_bunge_deg(
        cls, phi1: float, capital_phi: float, phi2: float
    ) -> HomogeneousOrientationProvider:
        return cls(rotation_from_euler_bunge_deg(phi1, capital_phi, phi2))

    def rotations_global_to_material(self, point_count: int) -> NDArray:
        if isinstance(point_count, bool) or not isinstance(point_count, int):
            raise TypeError("point_count must be an integer")
        if point_count < 1:
            raise ValueError("point_count must be at least one")
        return np.broadcast_to(self._rotation, (point_count, 3, 3)).copy()


class PixelOrientationProvider:
    """Per-pixel orientation map replicated over the material states.

    The default ``C`` order is the convention of the spectral pixel solver.
    Classical ``StructuredMesh`` FEM uses ``F`` element numbering and must
    request ``element_order="F"`` explicitly in its orientation mapping.
    """

    def __init__(
        self,
        rotations_global_to_material: ArrayLike,
        *,
        element_order: Literal["C", "F"] = "C",
    ) -> None:
        rotations = np.asarray(rotations_global_to_material, dtype=float)
        if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
            raise ValueError(
                "pixel orientations must have shape (nx, ny, 3, 3), "
                f"got {rotations.shape}"
            )
        self._pixel_shape = (int(rotations.shape[0]), int(rotations.shape[1]))
        if element_order not in {"C", "F"}:
            raise ValueError("element_order must be 'C' or 'F'")
        self._element_order = element_order
        self._rotations = validate_rotations(rotations.reshape(-1, 3, 3)).reshape(
            rotations.shape
        )

    @classmethod
    def from_euler_bunge_deg(
        cls,
        angles: ArrayLike,
        *,
        element_order: Literal["C", "F"] = "C",
    ) -> PixelOrientationProvider:
        return cls(rotations_from_euler_bunge_deg(angles), element_order=element_order)

    @property
    def pixel_shape(self) -> tuple[int, int]:
        return self._pixel_shape

    def rotations_global_to_material(self, point_count: int) -> NDArray:
        nx, ny = self.pixel_shape
        pixel_count = nx * ny
        if point_count < 1 or point_count % pixel_count != 0:
            raise ValueError(
                f"point count {point_count} is not an integer multiple of "
                f"pixel count {pixel_count}"
            )
        states_per_pixel = point_count // pixel_count
        flattened = self._rotations.reshape(pixel_count, 3, 3, order=self._element_order)
        return np.repeat(flattened, states_per_pixel, axis=0)


def orientation_provider_from_mapping(configuration: dict[str, object]) -> OrientationProvider:
    """Build a provider from the `crystal_orientation` block of a configuration.

    Only the homogeneous mode exists today. The dispatch is written as a mode
    lookup so that a per-Gauss-point or EBSD provider is a new branch here and
    not a change to any caller.
    """

    mode = configuration.get("mode", "homogeneous")
    if mode == "ebsd":
        angles = configuration.get("euler_bunge_deg")
        if angles is None:
            raise ValueError("ebsd orientation needs 'euler_bunge_deg'")
        element_order = configuration.get("element_order", "C")
        if element_order not in {"C", "F"}:
            raise ValueError("ebsd element_order must be 'C' or 'F'")
        return PixelOrientationProvider.from_euler_bunge_deg(
            cast(ArrayLike, angles), element_order=element_order
        )
    if mode != "homogeneous":
        raise ValueError(
            f"unsupported crystal_orientation mode {mode!r}; available: homogeneous, ebsd"
        )
    matrix = configuration.get("matrix")
    angles = configuration.get("euler_bunge_deg")
    if (matrix is None) == (angles is None):
        raise ValueError(
            "crystal_orientation needs exactly one of 'matrix' or 'euler_bunge_deg'"
        )
    if matrix is not None:
        return HomogeneousOrientationProvider(matrix)  # type: ignore[arg-type]
    values = np.asarray(angles, dtype=float)
    if values.shape != (3,):
        raise ValueError("euler_bunge_deg must hold exactly three angles")
    return HomogeneousOrientationProvider.from_euler_bunge_deg(*values)
