"""Optional explicit pyFFTW implementation of the full-Dirichlet DST-I."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.transforms import (
    FloatArray,
    SpectralTransformConfig,
    TransformDiagnostics,
)
from fem_inhouse.spectral2d.wisdom import load_wisdom, save_wisdom


@dataclass(slots=True)
class FullDirichletDSTIFFTWPlan2D:
    grid: StructuredGrid2D
    config: SpectralTransformConfig
    _diagnostics: TransformDiagnostics = field(init=False)
    #: One plan and one buffer pair serve both directions. `RODFT00` is its own
    #: inverse up to the scaling already applied below, and the two plans this
    #: class used to build were identical in every respect a planner sees --
    #: same direction, shape, flags and thread count -- differing only in which
    #: arrays they were bound to. Measuring both doubled the planning cost, and
    #: at full field the two extra buffers were 356 MB of nothing.
    _input: np.ndarray = field(init=False, repr=False)
    _output: np.ndarray = field(init=False, repr=False)
    _forward: Any = field(init=False, repr=False)
    _scale: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import pyfftw  # type: ignore[import-not-found]
        except ImportError as error:
            raise ImportError(
                "FFTW transform backend requested, but pyFFTW is not installed. "
                "Install the project with the 'fftw' optional dependency."
            ) from error
        if self.grid.nx < 2 or self.grid.ny < 2:
            raise ValueError("DST-I full-Dirichlet plans need at least 2 pixels per axis")
        shape = (*self.grid.interior_shape, 2)
        metadata = {
            "schema_version": 1,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pyfftw_version": pyfftw.__version__,
            "fftw_version": getattr(pyfftw, "__version__", "unknown"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "dtype": "float64",
            "shape": list(shape),
            "axes": [0, 1],
            "threads": self.config.workers,
            "planner_effort": self.config.fftw_planner_effort,
        }
        wisdom_loaded = False
        if self.config.fftw_use_wisdom and self.config.fftw_wisdom_directory is not None:
            wisdom_loaded = load_wisdom(self.config.fftw_wisdom_directory, metadata, pyfftw)
        self._input = pyfftw.empty_aligned(shape, dtype="float64")
        self._output = pyfftw.empty_aligned(shape, dtype="float64")
        flags = {
            "estimate": "FFTW_ESTIMATE",
            "measure": "FFTW_MEASURE",
            "patient": "FFTW_PATIENT",
        }
        started = time.perf_counter()
        self._forward = pyfftw.FFTW(
            self._input,
            self._output,
            axes=(0, 1),
            direction=("FFTW_RODFT00", "FFTW_RODFT00"),
            flags=(flags[self.config.fftw_planner_effort],),
            threads=self.config.workers,
        )
        planning_seconds = time.perf_counter() - started
        if self.config.fftw_use_wisdom and self.config.fftw_wisdom_directory is not None:
            save_wisdom(self.config.fftw_wisdom_directory, metadata, pyfftw)
        self._scale = 1.0 / (2.0 * np.sqrt(self.grid.nx * self.grid.ny))
        self._diagnostics = TransformDiagnostics(
            backend="fftw",
            implementation="pyfftw.FFTW",
            interior_shape=self.grid.interior_shape,
            batch_components=2,
            dtype="float64",
            workers=self.config.workers,
            planner_effort=self.config.fftw_planner_effort,
            wisdom_loaded=wisdom_loaded,
            planning_seconds=planning_seconds,
        )

    @property
    def frequencies_x(self) -> FloatArray:
        return np.pi * np.arange(1, self.grid.nx, dtype=np.float64) / self.grid.nx

    @property
    def frequencies_y(self) -> FloatArray:
        return np.pi * np.arange(1, self.grid.ny, dtype=np.float64) / self.grid.ny

    @property
    def backend_name(self) -> str:
        return "fftw"

    @property
    def diagnostics(self) -> TransformDiagnostics:
        return self._diagnostics

    def _validate(self, values: ArrayLike) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        expected = (*self.grid.interior_shape, 2)
        if array.shape != expected:
            raise ValueError(f"expected displacement shape {expected}, got {array.shape}")
        return array

    def forward_into(self, source: ArrayLike, destination: FloatArray) -> None:
        values = self._validate(source)
        if destination.shape != values.shape:
            raise ValueError(f"destination shape {destination.shape} does not match {values.shape}")
        np.copyto(self._input, values)
        self._forward()
        np.multiply(self._output, self._scale, out=destination)

    def inverse_into(self, source: ArrayLike, destination: FloatArray) -> None:
        values = self._validate(source)
        if destination.shape != values.shape:
            raise ValueError(f"destination shape {destination.shape} does not match {values.shape}")
        np.copyto(self._input, values)
        self._forward()
        np.multiply(self._output, self._scale, out=destination)

    def forward_displacement(self, interior_field: ArrayLike) -> FloatArray:
        values = self._validate(interior_field)
        result = np.empty_like(values)
        self.forward_into(values, result)
        return result

    def inverse_displacement(self, transformed_field: ArrayLike) -> FloatArray:
        values = self._validate(transformed_field)
        result = np.empty_like(values)
        self.inverse_into(values, result)
        return result

    def extract_interior(self, nodal_field: ArrayLike) -> FloatArray:
        values = np.asarray(nodal_field, dtype=np.float64)
        if values.shape[:2] != self.grid.node_shape:
            raise ValueError(f"expected nodal shape {self.grid.node_shape}, got {values.shape}")
        return values[1:-1, 1:-1, ...].copy()

    def embed_interior(self, interior_field: ArrayLike) -> FloatArray:
        values = self._validate(interior_field)
        embedded = np.zeros(self.grid.node_shape + values.shape[2:], dtype=np.float64)
        embedded[1:-1, 1:-1, ...] = values
        return embedded
