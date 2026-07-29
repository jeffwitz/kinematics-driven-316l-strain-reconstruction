"""Declared OpenCV DISFlow reproduction implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

ByteImage = NDArray[np.uint8]
FloatArray = NDArray[np.float32]


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "DISFlow support requires the 'measurement' optional dependency"
        ) from error
    return cv2


@dataclass(frozen=True, slots=True)
class DISFlowConfig:
    """Fully declared settings for the reproduction implementation."""

    preset: str = "medium"
    finest_scale: int = 0
    gradient_descent_iterations: int = 30
    patch_size: int = 8
    patch_stride: int = 3
    use_mean_normalization: bool = True
    use_spatial_propagation: bool = True
    variational_refinement_alpha: float = 100.0
    variational_refinement_delta: float = 1.0
    variational_refinement_gamma: float = 0.0
    variational_refinement_epsilon: float = 0.002
    variational_refinement_iterations: int = 30

    def __post_init__(self) -> None:
        if self.preset not in {"ultrafast", "fast", "medium"}:
            raise ValueError("preset must be ultrafast, fast, or medium")
        integer_positive = (
            self.gradient_descent_iterations,
            self.patch_size,
            self.patch_stride,
            self.variational_refinement_iterations,
        )
        if any(value < 1 for value in integer_positive):
            raise ValueError("DISFlow iteration and patch values must be positive")
        if self.finest_scale < 0:
            raise ValueError("finest_scale must be nonnegative")
        weights = (
            self.variational_refinement_alpha,
            self.variational_refinement_delta,
            self.variational_refinement_gamma,
            self.variational_refinement_epsilon,
        )
        if not np.isfinite(weights).all() or any(value < 0.0 for value in weights):
            raise ValueError("variational-refinement parameters must be finite and nonnegative")

    def as_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible configuration."""

        return asdict(self)


def create_disflow(config: DISFlowConfig | None = None) -> Any:
    """Create and configure one OpenCV DISFlow object."""

    cv2 = _cv2()
    selected = DISFlowConfig() if config is None else config
    presets = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    flow = cv2.DISOpticalFlow_create(presets[selected.preset])
    flow.setFinestScale(selected.finest_scale)
    flow.setGradientDescentIterations(selected.gradient_descent_iterations)
    flow.setPatchSize(selected.patch_size)
    flow.setPatchStride(selected.patch_stride)
    flow.setUseMeanNormalization(selected.use_mean_normalization)
    flow.setUseSpatialPropagation(selected.use_spatial_propagation)
    flow.setVariationalRefinementAlpha(selected.variational_refinement_alpha)
    flow.setVariationalRefinementDelta(selected.variational_refinement_delta)
    flow.setVariationalRefinementGamma(selected.variational_refinement_gamma)
    flow.setVariationalRefinementEpsilon(selected.variational_refinement_epsilon)
    flow.setVariationalRefinementIterations(selected.variational_refinement_iterations)
    return flow


def _byte_image(values: NDArray[np.generic], *, name: str) -> ByteImage:
    image = np.asarray(values)
    if image.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional grayscale image")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8")
    return np.ascontiguousarray(image)


def run_disflow(
    reference: NDArray[np.generic],
    deformed: NDArray[np.generic],
    *,
    config: DISFlowConfig | None = None,
    initial_flow: NDArray[np.generic] | None = None,
) -> FloatArray:
    """Return image-coordinate flow ``[..., (column, row)]`` in pixels."""

    reference_image = _byte_image(reference, name="reference")
    deformed_image = _byte_image(deformed, name="deformed")
    if reference_image.shape != deformed_image.shape:
        raise ValueError("reference and deformed images must have the same shape")
    initial = None
    if initial_flow is not None:
        initial = np.asarray(initial_flow, dtype=np.float32)
        if initial.shape != (*reference_image.shape, 2):
            raise ValueError("initial_flow must have shape (*image.shape, 2)")
        if not np.isfinite(initial).all():
            raise ValueError("initial_flow must contain finite values")
        initial = np.ascontiguousarray(initial)
    result = create_disflow(config).calc(reference_image, deformed_image, initial)
    output = np.asarray(result, dtype=np.float32)
    if output.shape != (*reference_image.shape, 2) or not np.isfinite(output).all():
        raise RuntimeError("DISFlow returned an invalid flow field")
    return output


def warp_image(
    reference: NDArray[np.generic],
    displacement_pixels: NDArray[np.generic],
) -> ByteImage:
    """Warp a reference image by a known forward displacement field."""

    cv2 = _cv2()
    image = _byte_image(reference, name="reference")
    displacement = np.asarray(displacement_pixels, dtype=np.float32)
    if displacement.shape != (*image.shape, 2):
        raise ValueError("displacement_pixels must have shape (*image.shape, 2)")
    if not np.isfinite(displacement).all():
        raise ValueError("displacement_pixels must contain finite values")
    rows, columns = np.indices(image.shape, dtype=np.float32)
    map_x = columns - displacement[..., 0]
    map_y = rows - displacement[..., 1]
    return np.asarray(
        cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        ),
        dtype=np.uint8,
    )
