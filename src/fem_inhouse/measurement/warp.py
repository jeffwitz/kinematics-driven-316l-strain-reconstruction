"""Physically explicit forward-image warping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

WarpMode = Literal["legacy_approximate_inverse", "iterative_forward_inverse"]
ByteImage = NDArray[np.uint8]

#: Resampling used by the synthetic warp, named for the observation manifest so
#: a replay is reproducible without reading this module. cv2 is imported lazily,
#: so these are the symbolic names rather than the OpenCV enum values.
WARP_INTERPOLATION = "cv2.INTER_LINEAR"
WARP_BORDER_MODE = "cv2.BORDER_REFLECT101"


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("image warping requires the measurement dependency") from error
    return cv2


@dataclass(frozen=True, slots=True)
class WarpResult:
    """Warped image and inverse-map convergence diagnostics."""

    image: ByteImage
    mode: WarpMode
    converged: bool
    iterations: int
    residual_pixels: float
    minimum_forward_jacobian: float


def _inputs(
    reference: NDArray[np.generic],
    displacement_pixels: NDArray[np.generic],
) -> tuple[ByteImage, NDArray[np.float64]]:
    image = np.asarray(reference)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise TypeError("reference must be a two-dimensional uint8 image")
    displacement = np.asarray(displacement_pixels, dtype=np.float64)
    if displacement.shape != (*image.shape, 2):
        raise ValueError("displacement_pixels must have shape (*image.shape, 2)")
    if not np.isfinite(displacement).all():
        raise ValueError("displacement_pixels must contain finite values")
    return np.ascontiguousarray(image), np.ascontiguousarray(displacement)


def _minimum_jacobian(displacement: NDArray[np.float64]) -> float:
    du_drow, du_dcolumn = np.gradient(displacement[..., 0].astype(np.float64))
    dv_drow, dv_dcolumn = np.gradient(displacement[..., 1].astype(np.float64))
    determinant = (1.0 + du_dcolumn) * (1.0 + dv_drow) - du_drow * dv_dcolumn
    return float(np.min(determinant))


def warp_forward_displacement(
    reference: NDArray[np.generic],
    displacement_pixels: NDArray[np.generic],
    *,
    mode: WarpMode = "iterative_forward_inverse",
    inverse_tolerance_pixels: float = 1.0e-5,
    maximum_inverse_iterations: int = 50,
    minimum_jacobian: float = 1.0e-6,
) -> WarpResult:
    """Warp an image whose displacement is tabulated at source coordinates.

    The nominal mode solves ``destination = source + u(source)`` for the
    source coordinate of every destination pixel. The legacy mode evaluates
    ``destination - u(destination)`` once.
    """

    cv2 = _cv2()
    image, displacement = _inputs(reference, displacement_pixels)
    if not np.isfinite(inverse_tolerance_pixels) or inverse_tolerance_pixels <= 0.0:
        raise ValueError("inverse_tolerance_pixels must be finite and positive")
    if maximum_inverse_iterations < 1:
        raise ValueError("maximum_inverse_iterations must be positive")
    forward_jacobian = _minimum_jacobian(displacement)
    if forward_jacobian <= minimum_jacobian:
        raise ValueError(
            "forward displacement map is non-invertible: "
            f"minimum Jacobian={forward_jacobian:.6g}"
        )

    destination_row, destination_column = np.indices(image.shape, dtype=np.float64)
    if mode == "legacy_approximate_inverse":
        source_column = destination_column - displacement[..., 0]
        source_row = destination_row - displacement[..., 1]
        iterations = 1
        residual = float("nan")
        converged = True
    elif mode == "iterative_forward_inverse":
        source_column = destination_column - displacement[..., 0]
        source_row = destination_row - displacement[..., 1]
        converged = False
        residual = float("inf")
        for iteration in range(1, maximum_inverse_iterations + 1):
            coordinates = np.stack((source_row, source_column))
            sampled_u = map_coordinates(
                displacement[..., 0],
                coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            )
            sampled_v = map_coordinates(
                displacement[..., 1],
                coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            )
            next_column = destination_column - sampled_u
            next_row = destination_row - sampled_v
            residual = float(
                max(
                    np.max(np.abs(next_column - source_column)),
                    np.max(np.abs(next_row - source_row)),
                )
            )
            source_column, source_row = next_column, next_row
            iterations = iteration
            if residual <= inverse_tolerance_pixels:
                converged = True
                break
        if not converged:
            raise RuntimeError(
                "forward displacement inversion did not converge after "
                f"{maximum_inverse_iterations} iterations; residual={residual:.6g} px"
            )
    else:
        raise ValueError(f"unsupported warp mode: {mode}")

    warped = cv2.remap(
        image,
        source_column.astype(np.float32),
        source_row.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return WarpResult(
        image=np.asarray(warped, dtype=np.uint8),
        mode=mode,
        converged=converged,
        iterations=iterations,
        residual_pixels=residual,
        minimum_forward_jacobian=forward_jacobian,
    )
