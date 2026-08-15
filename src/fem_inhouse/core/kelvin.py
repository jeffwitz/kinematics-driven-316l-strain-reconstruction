"""Kelvin/Mandel representation of symmetric tensors.

The repository stores symmetric tensors in engineering Voigt form: strains carry
`gamma_xy = 2 eps_xy`, stresses do not. Contractions then need a metric --
`sigma : eps = sigma^T diag(1, 1, 2)^{-1} ...` depending on which side carries
the factor -- and every norm, dissipation and inner product has to remember
which convention its operand is in. That bookkeeping is where the mistakes live.

Kelvin/Mandel removes it by scaling the off-diagonal slots of *both* stress and
strain by `sqrt(2)`:

```text
2D:  [xx, yy, sqrt2 xy]
3D:  [xx, yy, zz, sqrt2 yz, sqrt2 xz, sqrt2 xy]
```

The basis is then orthonormal, so `A : B` is the plain Euclidean dot product and
`|A|_F` the plain 2-norm. No metric matrix, no hand-inserted factor of two.

Two conversions are counter-intuitive coming from engineering Voigt, and both
are asserted in the tests rather than left to the reader:

* a `B` matrix producing the **engineering** shear `d_y u_x + d_x u_y` converts
  with its shear row **divided** by `sqrt(2)`, not multiplied -- because
  `sqrt2 eps_xy = gamma_xy / sqrt2`;
* an isotropic plane-stress stiffness has `C^K = 2G` in the shear slot, where
  the engineering form has `G`.

This module is the representation only. Migrating the mechanical core to use it
is a separate, larger piece of work: the engineering convention is currently
baked into the qualified solver chain and into the MFront/MGIS interfaces, where
it must survive as the exchange format with conversion at the boundary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_ROOT_TWO = float(np.sqrt(2.0))
#: Multipliers taking a tensor's independent components to Kelvin coordinates.
KELVIN_SCALE_2D = np.array([1.0, 1.0, _ROOT_TWO])
KELVIN_SCALE_3D = np.array([1.0, 1.0, 1.0, _ROOT_TWO, _ROOT_TWO, _ROOT_TWO])


def _scale(size: int) -> FloatArray:
    if size == 3:
        return KELVIN_SCALE_2D
    if size == 6:
        return KELVIN_SCALE_3D
    raise ValueError("symmetric tensors have three components in 2D and six in 3D")


def stress_from_voigt(values: ArrayLike) -> FloatArray:
    """Voigt stress to Kelvin: the off-diagonal slots gain `sqrt(2)`."""

    array = np.asarray(values, dtype=np.float64)
    return array * _scale(array.shape[-1])


def stress_to_voigt(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    return array / _scale(array.shape[-1])


def strain_from_engineering(values: ArrayLike) -> FloatArray:
    """Engineering strain to Kelvin: the shear slots are **divided** by `sqrt(2)`.

    The engineering slot holds `gamma = 2 eps`, and Kelvin wants `sqrt2 eps`,
    so the factor is `1/sqrt2` and not `sqrt2`. Getting this backwards leaves
    every shear four times too large in a quadratic form, which is exactly the
    kind of error a metric-free representation exists to prevent.
    """

    array = np.asarray(values, dtype=np.float64)
    return array / _scale(array.shape[-1])


def strain_to_engineering(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    return array * _scale(array.shape[-1])


def stiffness_from_engineering(matrix: ArrayLike) -> FloatArray:
    """Convert `sigma_voigt = C eps_engineering` into its Kelvin counterpart.

    With `sigma_K = S sigma_V` and `eps_eng = S eps_K`, where `S` is the scale
    vector as a diagonal, the Kelvin stiffness is `S C S`. For isotropic plane
    stress this sends the shear entry from `G` to `2G`.
    """

    array = np.asarray(matrix, dtype=np.float64)
    scale = _scale(array.shape[-1])
    return scale[:, None] * array * scale[None, :]


def strain_operator_from_engineering(matrix: ArrayLike) -> FloatArray:
    """Convert a `B` producing engineering strain into one producing Kelvin strain."""

    array = np.asarray(matrix, dtype=np.float64)
    return array / _scale(array.shape[0])[:, None]


def three_dimensional_from_plane_stress_plastic(values: ArrayLike) -> FloatArray:
    """Complete a plane-stress plastic strain to its 3D Kelvin form.

    Plastic incompressibility fixes `eps_zz = -(eps_xx + eps_yy)`. The in-plane
    triple alone is a legitimate object, but its norm is **not** the equivalent
    plastic strain: calling a two-dimensional norm `p_eq` silently drops the
    out-of-plane contribution. Anything labelled equivalent plastic strain has
    to go through here first.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError("expected a plane-stress Kelvin triple")
    out_of_plane = -(array[..., 0] + array[..., 1])
    zeros = np.zeros((*array.shape[:-1], 2), dtype=np.float64)
    return np.concatenate(
        [array[..., :2], out_of_plane[..., None], zeros, array[..., 2:3]], axis=-1
    )


def equivalent_plastic_strain(values: ArrayLike) -> FloatArray:
    """`sqrt(2/3) |dev eps_p|` from a plane-stress Kelvin plastic strain.

    The completion above makes the tensor deviatoric by construction, so the
    von Mises equivalent is the plain Kelvin norm scaled by `sqrt(2/3)` -- no
    metric, and the out-of-plane component included.
    """

    full = three_dimensional_from_plane_stress_plastic(values)
    return np.sqrt(2.0 / 3.0) * np.linalg.norm(full, axis=-1)
