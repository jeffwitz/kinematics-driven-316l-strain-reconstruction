"""Observability of an effective elastic heterogeneity, in the Kelvin basis.

The EBSD surface map does not explain the early mechanical defect, but that
does not make the elastic heterogeneity of the specimen negligible: the map is
one section of a body several millimetres thick, and what surface DIC sees is
the effective operator of everything underneath. This looks for that effective
heterogeneity directly, without pretending to reconstruct orientations at
depth.

In plane stress the isotropic stiffness has only **two** distinct Kelvin
eigenvalues over three directions,

```text
k_V = (1, 1, 0)/sqrt(2),  k_D = (1, -1, 0)/sqrt(2),  k_S = (0, 0, 1),
c_V = E/(1-nu),           c_D = E/(1+nu) = 2G   (doubly degenerate).
```

A perturbation is written multiplicatively so that its coefficients are
dimensionless *relative* stiffness changes rather than megapascals:

```text
C(x) = C0^(1/2) (I + A(x)) C0^(1/2),      A symmetric, |A|_F^2 = sum a_alpha^2.
```

`A` has six components in an orthonormal basis of symmetric matrices: the
volumetric channel, the mean deviatoric channel, the two that lift the
deviatoric degeneracy, and the two volumetric/deviatoric couplings. Which of
them the experiment can see is left to the spectrum rather than decided in
advance.

To first order the perturbation acts as `K0 du = -B^T (dC eps)`, so the
observation operator is the exact elastic twin of the plastic one, sharing
`K0^-1`, `M_D` and `W_D`. One difference works in its favour: the stiffness
field is the **same for every state**, so a single unknown must explain all the
early images at once, which is far better conditioned than one plastic
increment per state.

A uniform `A = alpha I` rescales `C` uniformly, and under pure Dirichlet
conditions that leaves the displacement unchanged. This mode is therefore
exactly in the kernel, which the tests use as a built-in check that the
construction is right.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator

from fem_inhouse.identification.tensor_plastic_observability import (
    FieldOperator,
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

FloatArray = NDArray[np.float64]

#: Engineering-shear to Kelvin scaling: eps_eng = S eps_kelvin, sig_kelvin = S sig_eng.
_KELVIN_SCALE = np.array([1.0, 1.0, np.sqrt(2.0)])


def kelvin_channel_basis() -> FloatArray:
    """Six orthonormal symmetric matrices, `(6, 3, 3)`, in Kelvin coordinates.

    Ordered so the interpretation is readable: volumetric, mean deviatoric,
    degeneracy lifting, then the two volumetric/deviatoric couplings.
    """

    root = 1.0 / np.sqrt(2.0)
    directions = np.array(
        [[root, root, 0.0], [root, -root, 0.0], [0.0, 0.0, 1.0]]
    )  # k_V, k_D, k_S as rows, in Kelvin coordinates
    channels = []
    for index in (0, 1, 2):
        outer = np.outer(directions[index], directions[index])
        channels.append(outer)
    for first, second in ((1, 2), (0, 1), (0, 2)):
        outer = np.outer(directions[first], directions[second])
        channels.append((outer + outer.T) / np.sqrt(2.0))
    return np.asarray(channels, dtype=np.float64)


#: Human-readable families, in the order of :func:`kelvin_channel_basis`.
CHANNEL_FAMILIES = (
    "volumetric",
    "deviatoric-in-plane",
    "deviatoric-shear",
    "degeneracy-lifting",
    "coupling-volumetric-deviatoric",
    "coupling-volumetric-shear",
)
#: The two channels that keep the isotropic form, `A = diag(a_V, a_D, a_D)`.
ISOTROPIC_CHANNELS = (0, 1, 2)


def isotropic_kelvin_stiffness(young_modulus_mpa: float, poisson_ratio: float) -> FloatArray:
    """`C0` in Kelvin coordinates, and its square root, both `(3, 3)`."""

    volumetric = young_modulus_mpa / (1.0 - poisson_ratio)
    deviatoric = young_modulus_mpa / (1.0 + poisson_ratio)
    root = 1.0 / np.sqrt(2.0)
    vectors = np.array([[root, root, 0.0], [root, -root, 0.0], [0.0, 0.0, 1.0]]).T
    eigenvalues = np.array([volumetric, deviatoric, deviatoric])
    return np.asarray((vectors * eigenvalues) @ vectors.T, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ElasticKelvinObservabilityOperator:
    """Observability of a relative stiffness perturbation over several states."""

    mechanics: TensorPlasticObservabilityOperator
    reference_strain: FloatArray
    channels: FloatArray
    channel_indices: tuple[int, ...]
    pixels: int
    transfer: FieldOperator
    whitener: FieldOperator
    solve_stiffness: Callable[[FloatArray], FloatArray]

    @classmethod
    def build(
        cls,
        mechanics: TensorPlasticObservabilityOperator,
        reference_displacement: Sequence[ArrayLike],
        *,
        young_modulus_mpa: float,
        poisson_ratio: float,
        channel_indices: Sequence[int] | None = None,
    ) -> ElasticKelvinObservabilityOperator:
        pixels = mechanics.grid.nx
        stiffness = isotropic_kelvin_stiffness(young_modulus_mpa, poisson_ratio)
        eigenvalues, vectors = np.linalg.eigh(stiffness)
        root = (vectors * np.sqrt(eigenvalues)) @ vectors.T
        # dC in engineering coordinates, per unit channel coefficient.
        scale = np.diag(1.0 / _KELVIN_SCALE)
        channels = np.asarray(
            [scale @ (root @ channel @ root) @ scale for channel in kelvin_channel_basis()],
            dtype=np.float64,
        )
        selected = tuple(range(6)) if channel_indices is None else tuple(channel_indices)
        strains = np.asarray(
            [
                np.asarray(mechanics.kinematics.strain(field), dtype=np.float64).reshape(-1, 3)
                for field in reference_displacement
            ]
        )
        return cls(
            mechanics=mechanics,
            reference_strain=strains,
            channels=channels[list(selected)],
            channel_indices=selected,
            pixels=pixels,
            transfer=mechanics.transfer,
            whitener=mechanics.whitener,
            solve_stiffness=mechanics.solve_stiffness,
        )

    @property
    def state_count(self) -> int:
        return int(self.reference_strain.shape[0])

    @property
    def channel_count(self) -> int:
        return len(self.channel_indices)

    @property
    def coefficient_size(self) -> int:
        return self.pixels * self.pixels * self.channel_count

    @property
    def observation_size(self) -> int:
        return self.state_count * self.mechanics.observation_size

    def _per_point(self, coefficients: FloatArray) -> FloatArray:
        """Expand per-pixel coefficients onto the two sub-cells of each pixel."""

        per_pixel = coefficients.reshape(self.pixels * self.pixels, self.channel_count)
        return np.repeat(per_pixel, 2, axis=0)

    def matvec(self, values: ArrayLike) -> FloatArray:
        coefficients = self._per_point(np.asarray(values, dtype=np.float64))
        # dC(x) eps_n(x), summed over the retained channels.
        perturbation = np.einsum("pa,aij->pij", coefficients, self.channels)
        blocks = []
        for state in range(self.state_count):
            stress = np.einsum("pij,pj->pi", perturbation, self.reference_strain[state])
            forcing = -self.mechanics._strain_transpose(stress.reshape(-1))
            displacement = unpack_interior(
                self.solve_stiffness(forcing), self.mechanics.grid
            )
            observed = self.whitener.apply(self.transfer.apply(displacement))
            blocks.append(np.asarray(observed, dtype=np.float64).reshape(-1))
        return np.concatenate(blocks)

    def rmatvec(self, values: ArrayLike) -> FloatArray:
        stacked = np.asarray(values, dtype=np.float64).reshape(
            self.state_count, *self.mechanics.grid.node_shape, 2
        )
        total = np.zeros(
            (self.pixels * self.pixels * 2, self.channel_count), dtype=np.float64
        )
        for state in range(self.state_count):
            dual = self.transfer.adjoint(self.whitener.adjoint(stacked[state]))
            solved = self.solve_stiffness(pack_interior(np.asarray(dual, dtype=np.float64)))
            field = unpack_interior(solved, self.mechanics.grid)
            strain = np.asarray(
                self.mechanics.kinematics.strain(field), dtype=np.float64
            ).reshape(-1, 3)
            # Adjoint of `-B^T dC eps`: the sign follows the forward action.
            total += -np.einsum(
                "pi,aij,pj->pa", strain, self.channels, self.reference_strain[state]
            )
        per_pixel = total.reshape(self.pixels * self.pixels, 2, self.channel_count).sum(axis=1)
        return per_pixel.reshape(-1)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            (self.observation_size, self.coefficient_size),
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.float64,
        )

    def uniform_rescaling_coefficients(self) -> FloatArray:
        """`A = alpha I` everywhere: the exact kernel direction of the operator."""

        identity = np.eye(3)
        weights = np.einsum("aij,ij->a", kelvin_channel_basis(), identity)[
            list(self.channel_indices)
        ]
        field = np.tile(weights, (self.pixels * self.pixels, 1))
        return (field / np.linalg.norm(field)).reshape(-1)
