"""The exact split `sigma = sigma_n + C_0 : d eps + h`, at full field.

Step one of the hyper-reduction, and deliberately not yet reduced: before any
sampling, the decomposition must reproduce the behaviour it decomposes, to
rounding. Everything later rests on that.

Why this split rather than `C_0 : eps`. The correction vanishes at the last
converged state,

```text
h(u_n) = sigma_exact(u_n) - sigma_n - C_0 : 0 = 0
```

so in the warm regime -- the one a coefficient perturbation lives in, where a
solve costs one Newton -- the quantity being sampled starts from zero and stays
small. A split around the origin would sample a large, nearly constant field
and waste its resolution on it.

`C_0` is read from the material itself at zero strain, never assumed. The J2
batches here work in the **engineering** convention, where the tangent equals
`plane_stress_elasticity` and differs from the Kelvin stiffness by `mu` on the
shear entry; the identification operator and the spectral solver are Kelvin. The
elastic lifting in a dozen P43 scripts retains 32 % of its interior residual
because those two conventions were chained without conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceSplitTrial:
    """One trial state, expressed as reference plus correction."""

    reference_stress_mpa: FloatArray
    correction_mpa: FloatArray
    tangent_mpa: FloatArray
    tangent_correction_mpa: FloatArray

    @property
    def stress_mpa(self) -> FloatArray:
        """The exact stress the behaviour returned, reassembled."""

        return self.reference_stress_mpa + self.correction_mpa


def reference_stiffness_of(material, *, time_increment: float = 1.0) -> FloatArray:
    """`C_0` from the behaviour's own tangent at zero strain.

    Taken from the material rather than rebuilt from `E` and `nu`, so the
    convention cannot drift apart from the behaviour it is meant to shadow. The
    committed state is restored afterwards, since this is a probe and not a
    step of the calculation.
    """

    zero = np.zeros((material.point_count, 3), dtype=np.float64)
    trial = material.evaluate(zero, time_increment=time_increment, consistent_tangent=True)
    tangent = np.asarray(trial.tangent_in_plane_mpa, dtype=np.float64)
    material.revert()
    if tangent.ndim != 3 or tangent.shape[1:] != (3, 3):
        raise ValueError(f"unexpected tangent shape {tangent.shape}")
    spread = float(np.abs(tangent - tangent[0]).max())
    if spread > 1e-9 * float(np.abs(tangent[0]).max()):
        raise ValueError(
            "the elastic reference is not homogeneous across points; a per-point "
            f"C_0 is not supported by this split (spread {spread:.3e})"
        )
    return np.ascontiguousarray(tangent[0])


class ConstitutiveSplit:
    """Wraps a behaviour and reports it as reference plus correction.

    Adds nothing to the physics: `reference_stress + correction` is the stress
    the behaviour returned, bit for bit up to the additions performed here. The
    committed stress and strain are the ones the split is taken around, and they
    are refreshed only on `commit`.
    """

    def __init__(self, material, *, reference_stiffness: FloatArray | None = None) -> None:
        self.material = material
        self.reference_stiffness = (
            np.ascontiguousarray(reference_stiffness, dtype=np.float64)
            if reference_stiffness is not None
            else reference_stiffness_of(material)
        )
        count = material.point_count
        self._committed_strain = np.zeros((count, 3), dtype=np.float64)
        self._committed_stress = np.zeros((count, 3), dtype=np.float64)
        self._last_strain: FloatArray | None = None

    @property
    def point_count(self) -> int:
        return int(self.material.point_count)

    @property
    def committed_stress_mpa(self) -> FloatArray:
        return self._committed_stress

    def reference_stress(self, strain: FloatArray) -> FloatArray:
        """`sigma_n + C_0 : (eps - eps_n)`, the cheap backbone."""

        increment = np.asarray(strain, dtype=np.float64).reshape(-1, 3) - self._committed_strain
        return self._committed_stress + increment @ self.reference_stiffness.T

    def evaluate(self, strain: FloatArray, *, time_increment: float = 1.0,
                 consistent_tangent: bool = True) -> ReferenceSplitTrial:
        """The exact behaviour, reported as reference plus correction."""

        values = np.ascontiguousarray(np.asarray(strain, dtype=np.float64).reshape(-1, 3))
        trial = self.material.evaluate(
            values, time_increment=time_increment, consistent_tangent=consistent_tangent
        )
        stress = np.asarray(trial.stress_in_plane_mpa, dtype=np.float64)
        reference = self.reference_stress(values)
        tangent = (
            np.asarray(trial.tangent_in_plane_mpa, dtype=np.float64)
            if trial.tangent_in_plane_mpa is not None
            else np.broadcast_to(
                self.reference_stiffness, (values.shape[0], 3, 3)
            ).copy()
        )
        self._last_strain = values
        return ReferenceSplitTrial(
            reference_stress_mpa=reference,
            correction_mpa=stress - reference,
            tangent_mpa=tangent,
            tangent_correction_mpa=tangent - self.reference_stiffness,
        )

    def commit(self) -> None:
        """Accept the last trial, and move the point the split is taken around."""

        if self._last_strain is None:
            raise RuntimeError("commit without a preceding evaluate")
        trial = self.material.evaluate(
            self._last_strain, time_increment=1.0, consistent_tangent=False
        )
        self.material.commit()
        self._committed_strain = self._last_strain.copy()
        self._committed_stress = np.asarray(
            trial.stress_in_plane_mpa, dtype=np.float64
        ).copy()
        self._last_strain = None

    def revert(self) -> None:
        self.material.revert()
        self._last_strain = None
