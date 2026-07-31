"""Declared OpenCV DISFlow reproduction implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .warp import WarpMode, warp_forward_displacement

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

    preset: str | None = "medium"
    finest_scale: int | None = 0
    gradient_descent_iterations: int | None = 30
    patch_size: int | None = 8
    patch_stride: int | None = 3
    use_mean_normalization: bool | None = True
    use_spatial_propagation: bool | None = True
    variational_refinement_alpha: float = 100.0
    variational_refinement_delta: float = 1.0
    variational_refinement_gamma: float = 0.0
    variational_refinement_epsilon: float = 0.002
    variational_refinement_iterations: int = 30

    def __post_init__(self) -> None:
        if self.preset not in {None, "ultrafast", "fast", "medium"}:
            raise ValueError("preset must be None, ultrafast, fast, or medium")
        integer_positive = (
            self.gradient_descent_iterations,
            self.patch_size,
            self.patch_stride,
            self.variational_refinement_iterations,
        )
        if any(value is not None and value < 1 for value in integer_positive):
            raise ValueError("DISFlow iteration and patch values must be positive")
        if self.finest_scale is not None and self.finest_scale < 0:
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
    flow = (
        cv2.DISOpticalFlow_create()
        if selected.preset is None
        else cv2.DISOpticalFlow_create(presets[selected.preset])
    )
    optional_setters = (
        ("setFinestScale", selected.finest_scale),
        ("setGradientDescentIterations", selected.gradient_descent_iterations),
        ("setPatchSize", selected.patch_size),
        ("setPatchStride", selected.patch_stride),
        ("setUseMeanNormalization", selected.use_mean_normalization),
        ("setUseSpatialPropagation", selected.use_spatial_propagation),
    )
    for setter, value in optional_setters:
        if value is not None:
            getattr(flow, setter)(value)
    flow.setVariationalRefinementAlpha(selected.variational_refinement_alpha)
    flow.setVariationalRefinementDelta(selected.variational_refinement_delta)
    flow.setVariationalRefinementGamma(selected.variational_refinement_gamma)
    flow.setVariationalRefinementEpsilon(selected.variational_refinement_epsilon)
    flow.setVariationalRefinementIterations(selected.variational_refinement_iterations)
    return flow


def query_disflow_configuration(config: DISFlowConfig | None = None) -> dict[str, Any]:
    """Read every supported setting back from the configured OpenCV object."""

    flow = create_disflow(config)
    getters = {
        "finest_scale": "getFinestScale",
        "gradient_descent_iterations": "getGradientDescentIterations",
        "patch_size": "getPatchSize",
        "patch_stride": "getPatchStride",
        "use_mean_normalization": "getUseMeanNormalization",
        "use_spatial_propagation": "getUseSpatialPropagation",
        "variational_refinement_alpha": "getVariationalRefinementAlpha",
        "variational_refinement_delta": "getVariationalRefinementDelta",
        "variational_refinement_gamma": "getVariationalRefinementGamma",
        "variational_refinement_epsilon": "getVariationalRefinementEpsilon",
        "variational_refinement_iterations": "getVariationalRefinementIterations",
    }
    return {name: getattr(flow, getter)() for name, getter in getters.items()}


def _byte_image(values: NDArray[np.generic], *, name: str) -> ByteImage:
    image = np.asarray(values)
    if image.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional grayscale image")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8")
    return np.ascontiguousarray(image)



def require_native_finest_scale(config: DISFlowConfig) -> None:
    """Refuse a coarse finest scale for metrological use.

    The first measurement-chain campaign ran at finest scale 1, which skips
    full-resolution variational refinement and reported an MTF-50 near 127 px
    against 49 px at native scale. That run was invalidated. Any use of this
    chain as a measurement instrument must therefore pin scale 0 explicitly.
    """

    if config.finest_scale != 0:
        raise ValueError(
            "metrological use requires finest_scale=0; "
            f"got {config.finest_scale!r}, which skips full-resolution refinement"
        )


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
    *,
    mode: WarpMode = "iterative_forward_inverse",
) -> ByteImage:
    """Warp a reference image by a known forward displacement field."""

    return warp_forward_displacement(
        reference,
        displacement_pixels,
        mode=mode,
    ).image
