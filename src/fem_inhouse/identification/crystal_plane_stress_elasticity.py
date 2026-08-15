"""Plane-stress elasticity of an FCC crystal, per point, from EBSD orientations.

The observability machinery measures the mechanical defect of a measured field
against a reference elastic model. With a homogeneous isotropic reference, any
elastic heterogeneity of the specimen lands in that defect and is
indistinguishable from an eigenstrain -- Eshelby's equivalent inclusion. Giving
the reference the real crystallographic elasticity is what removes that
confounder, and it changes nothing else in the construction.

Two things have to be done properly.

The cubic tensor must be rotated in **three dimensions and then condensed**, not
rotated as a `3 x 3` in-plane matrix. An orientation with an out-of-plane
component couples the in-plane response to the transverse one, and condensing
after the rotation is the only way to see it:

```text
C_ps = C_aa - C_ab (C_bb)^-1 C_ba,
```

with `a = (11, 22, 12)` in plane and `b = (33, 23, 13)` eliminated.

And the Voigt convention has to be fixed once. Strains carry engineering shear,
stresses do not, which is what makes `C_voigt[I, J] = C_ijkl` hold with no
factors for the shear columns: the factor two from the minor symmetry is
absorbed by `gamma = 2 eps`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg

FloatArray = NDArray[np.float64]

#: Voigt index pairs, engineering-shear ordering.
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
#: In-plane and eliminated Voigt indices for the plane-stress condensation.
_IN_PLANE = (0, 1, 5)
_TRANSVERSE = (2, 3, 4)


def cubic_stiffness_from_engineering_constants(
    young_modulus_mpa: float, poisson_ratio: float, shear_modulus_mpa: float
) -> FloatArray:
    """Voigt stiffness of a cubic crystal, from the constants the law declares.

    The repository states its FCC elasticity as `(E, nu, G)` in the crystal
    frame through the `StandardElasticity` brick, so the same three numbers are
    used here rather than a second set of `(C11, C12, C44)` that could drift
    away from the constitutive law.
    """

    compliance = np.zeros((6, 6), dtype=np.float64)
    normal = np.full((3, 3), -poisson_ratio / young_modulus_mpa)
    np.fill_diagonal(normal, 1.0 / young_modulus_mpa)
    compliance[:3, :3] = normal
    for index in range(3, 6):
        compliance[index, index] = 1.0 / shear_modulus_mpa
    return np.asarray(np.linalg.inv(compliance), dtype=np.float64)


def voigt_to_tensor(voigt: ArrayLike) -> FloatArray:
    """Expand a Voigt stiffness into the full fourth-order tensor."""

    matrix = np.asarray(voigt, dtype=np.float64)
    tensor = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for row, (i, j) in enumerate(_VOIGT):
        for column, (k, current) in enumerate(_VOIGT):
            value = matrix[row, column]
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, current), (current, k)):
                    tensor[a, b, c, d] = value
    return tensor


def tensor_to_voigt(tensor: ArrayLike) -> FloatArray:
    """Contract a fourth-order stiffness back to engineering-shear Voigt form."""

    values = np.asarray(tensor, dtype=np.float64)
    matrix = np.empty((6, 6), dtype=np.float64)
    for row, (i, j) in enumerate(_VOIGT):
        for column, (k, current) in enumerate(_VOIGT):
            matrix[row, column] = values[i, j, k, current]
    return matrix


def rotated_plane_stress_stiffness(
    crystal_voigt: ArrayLike, euler_bunge_deg: ArrayLike
) -> FloatArray:
    """Per-point plane-stress stiffness in the global frame, `(points, 3, 3)`.

    `rotation_from_euler_bunge_deg` returns `Q_global_to_material`, so the
    crystal tensor is carried to the global frame by its transpose:
    `C_global_ijkl = Q_pi Q_qj Q_rk Q_sl C_crystal_pqrs`.
    """

    crystal = voigt_to_tensor(crystal_voigt)
    rotations = np.asarray(
        rotations_from_euler_bunge_deg(np.asarray(euler_bunge_deg, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1, 3, 3)
    result = np.empty((rotations.shape[0], 3, 3), dtype=np.float64)
    for index, rotation in enumerate(rotations):
        rotated = np.einsum(
            "pi,qj,rk,sl,pqrs->ijkl", rotation, rotation, rotation, rotation, crystal
        )
        voigt = tensor_to_voigt(rotated)
        in_plane = np.ix_(_IN_PLANE, _IN_PLANE)
        coupling = np.ix_(_IN_PLANE, _TRANSVERSE)
        transverse = np.ix_(_TRANSVERSE, _TRANSVERSE)
        reverse = np.ix_(_TRANSVERSE, _IN_PLANE)
        result[index] = voigt[in_plane] - voigt[coupling] @ np.linalg.solve(
            voigt[transverse], voigt[reverse]
        )
    return result
