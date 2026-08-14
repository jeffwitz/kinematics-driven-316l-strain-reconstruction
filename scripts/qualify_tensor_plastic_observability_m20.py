#!/usr/bin/env python3
"""Full observability spectrum of the tensor plastic increment on P43 M20.

The unknown is the tensor increment ``z = Delta eps_p`` per material point, not
a scalar ``Delta p`` along an imposed J2 direction. The mechanics is then linear:

```text
sigma = C (B u - eps_p^n - z),      G = B^T C,      K = B^T C B,
K u = G (eps_p^n + z)  ->  the plastic forcing enters only through  f = G z.
```

Two fields differing by an element of ``ker G`` are mechanically identical, so
the object of interest is the quotient ``E_p / ker G``. Among the infinitely
many fields producing the same forcing, the canonical representative is the one
of least norm in a metric ``H``,

```text
z* = argmin 1/2 z^T H z   s.t.   G z = f,
z* = H^-1 G^T S^-1 f,     S = G H^-1 G^T.
```

`H` is a gauge, not a constitutive law. It is taken as the metric for which the
norm of an increment IS its equivalent plastic strain: with ``q^2 = s^T M s``
and the associated direction ``n = M s / q``, one has ``n^T M^-1 n = 1``, hence
``(Delta p n)^T M^-1 (Delta p n) = Delta p^2``. So ``H_loc = M^-1``, divided by
the point count so that the norm is a root-mean-square rather than a sum, which
matches the convention of the scalar observability runs. The inverse metric
``M`` weights the shear correctly; ``M`` itself would not.

Observability then reduces to one SVD. With ``S = L L^T`` and ``v = L w``,

```text
T^T T v = lambda S^-1 v   <=>   (T L)^T (T L) w = lambda w,
```

so the non-zero spectrum is the squared singular values of ``A = T L``, where
``T = W_D M_D K^-1`` is the experimental chain, and the plastic modes
``phi_j = H^-1 G^T L^-T w_j`` come out `H`-orthonormal for free.

The rank criterion is absolute, not a share of cumulated energy. `W_D` whitens
to noise units and the objective divides by the number of observed components,
so a mode excited at the physical amplitude ``p_ref`` produces

```text
SNR_j = p_ref sqrt(lambda_j / N_D)
```

in units of the DIC noise. Cumulated energy is archived as a descriptor only:
it always returns a number, including for a flat spectrum.

Nothing here is fitted and no oracle is solved. The operator is built from the
mechanics and the measurement chain alone, so this spectrum does not depend on
the 40 experimental states -- they decide later which part of the observable
subspace is actually excited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.identification.dic_whitening import DICSpectralTransfer, DICSpectralWhitener
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

ROOT = Path(__file__).resolve().parents[1]
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_OUTPUT = (
    ROOT / "validation/_generated/performance/experimental_oracle_p43_m20/tensor_observability"
)
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
#: Two amplitudes, because detectability depends on which field is being
#: identified. The first is the root-mean-square plastic increment of ONE step
#: of the archived oracle history; the second is the accumulated equivalent
#: plastic strain at the final state. The same operator answers both questions,
#: and judging an accumulated field at the increment amplitude would understate
#: observability by more than an order of magnitude.
REFERENCE_AMPLITUDES = {
    "single_increment_rms": 2.3503361528920064e-04,
    "accumulated_rms_final_state": 5.669788370458351e-03,
}


def _dominant_wavelength_mm(mode: np.ndarray, pixels: int) -> float:
    """Power-weighted mean wavelength of a plastic mode, in millimetres.

    The mode is averaged over the two sub-cells of each pixel and over its three
    tensor components, then read in the discrete Fourier plane. The zero mode is
    excluded: a uniform field has no wavelength and would otherwise dominate.
    """

    field = mode.reshape(pixels, pixels, 2, 3).mean(axis=2)
    power = np.zeros((pixels, pixels), dtype=np.float64)
    for component in range(3):
        power += np.abs(np.fft.fft2(field[:, :, component])) ** 2
    frequencies = np.fft.fftfreq(pixels, d=PIXEL_SIZE_MM)
    kx, ky = np.meshgrid(frequencies, frequencies, indexing="ij")
    magnitude = np.sqrt(kx**2 + ky**2)
    power[0, 0] = 0.0
    total = power.sum()
    if total <= 0.0:
        return float("inf")
    mean_frequency = float((power * magnitude).sum() / total)
    return float("inf") if mean_frequency <= 0.0 else 1.0 / mean_frequency


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=20)
    parser.add_argument("--modes-archived", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    pixels = arguments.pixels
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    kinematics = TwoSubcellDiagnostic2D(grid)
    point_count = kinematics.material_point_count
    plastic_size = point_count * 3
    free_size = 2 * (pixels - 1) ** 2

    elasticity = plane_stress_elasticity(YOUNG_MPA, POISSON)

    # B, assembled column by column from the qualified strain operator.
    strain_operator = np.empty((plastic_size, free_size), dtype=np.float64)
    for column in range(free_size):
        unit = np.zeros(free_size, dtype=np.float64)
        unit[column] = 1.0
        nodal = unpack_interior(unit, grid)
        strain_operator[:, column] = kinematics.strain(nodal).reshape(-1)

    # `divergence_from_sample_stress` is the internal-force adjoint, which
    # carries the opposite sign and the quadrature weight: it equals
    # `-w B^T`. Undoing both keeps K = B^T C B positive definite and leaves the
    # spectrum unchanged, since the weight cancels between K^-1 and S.
    weight = float(kinematics.sample_quadrature_weight)

    def strain_transpose(stress_flat: np.ndarray) -> np.ndarray:
        nodal = kinematics.divergence_from_sample_stress(
            stress_flat.reshape((pixels, pixels, 2, 3))
        )
        return -pack_interior(nodal) / weight

    # The adjoint identity is the premise of using B^T as the divergence: check
    # it rather than assume the two operators share a quadrature convention.
    generator = np.random.default_rng(20260815)
    probe_stress = generator.normal(size=plastic_size)
    probe_displacement = generator.normal(size=free_size)
    adjoint_error = abs(
        float(probe_stress @ (strain_operator @ probe_displacement))
        - float(strain_transpose(probe_stress) @ probe_displacement)
    ) / abs(float(probe_stress @ (strain_operator @ probe_displacement)))

    block_metric = PLANE_STRESS_VON_MISES_METRIC

    # G = B^T C, one column per plastic component.
    forcing_operator = np.empty((free_size, plastic_size), dtype=np.float64)
    elastic_stress = np.zeros(plastic_size, dtype=np.float64)
    for column in range(plastic_size):
        elastic_stress[:] = 0.0
        block = column // 3
        elastic_stress[3 * block : 3 * block + 3] = elasticity[:, column % 3]
        forcing_operator[:, column] = strain_transpose(elastic_stress)
    stiffness = forcing_operator @ strain_operator

    # H = blockdiag(M^-1) / point_count, so H^-1 = blockdiag(M) * point_count.
    inverse_gauge_blocks = block_metric * point_count
    scattered = forcing_operator.reshape(free_size, point_count, 3)
    weighted = np.einsum("fpi,ij->fpj", scattered, inverse_gauge_blocks)
    schur = np.einsum("fpi,gpi->fg", weighted, scattered)
    schur = 0.5 * (schur + schur.T)

    forcing_singular = np.linalg.svd(forcing_operator, compute_uv=False)
    forcing_rank = int(np.count_nonzero(forcing_singular > forcing_singular[0] * 1e-12))
    cholesky = np.linalg.cholesky(schur)

    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = image_flow_to_canonical(np.asarray(noise[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM)
    support = np.ones((*grid.node_shape, 2), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    whitener = DICSpectralWhitener.from_stationary_noise_field(
        canonical,
        target_shape=grid.node_shape,
        sample_count=256,
        seed=42,
        remove_spatial_mean=False,
        support_mask=support,
    )
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    observation_count = int(np.prod(grid.node_shape) * 2)

    # A = T L, one column at a time: solve, observe, whiten.
    factorised = np.linalg.cholesky(stiffness)

    def solve_stiffness(rhs: np.ndarray) -> np.ndarray:
        intermediate = np.linalg.solve(factorised, rhs)
        return np.linalg.solve(factorised.T, intermediate)

    observation = np.empty((observation_count, free_size), dtype=np.float64)
    for column in range(free_size):
        displacement = unpack_interior(solve_stiffness(cholesky[:, column]), grid)
        observation[:, column] = whitener.apply(transfer.apply(displacement)).reshape(-1)

    _, singular_values, right_transposed = np.linalg.svd(observation, full_matrices=False)
    eigenvalues = singular_values**2
    signal_to_noise = {
        name: amplitude * singular_values / np.sqrt(observation_count)
        for name, amplitude in REFERENCE_AMPLITUDES.items()
    }

    # phi_j = H^-1 G^T L^-T w_j, H-orthonormal by construction.
    right = right_transposed.T
    lifted = np.linalg.solve(cholesky.T, right[:, : arguments.modes_archived])
    dual = forcing_operator.T @ lifted
    modes = np.einsum("ij,pjm->pim", inverse_gauge_blocks, dual.reshape(point_count, 3, -1))
    modes = modes.reshape(plastic_size, -1)
    gauge_norms = np.einsum(
        "pim,ij,pjm->m",
        modes.reshape(point_count, 3, -1),
        np.linalg.inv(inverse_gauge_blocks),
        modes.reshape(point_count, 3, -1),
    )

    total = float(eigenvalues.sum())
    cumulated = np.cumsum(eigenvalues) / total
    effective_rank = float(total**2 / float((eigenvalues**2).sum()))
    archived = min(arguments.modes_archived, modes.shape[1])
    wavelengths = [_dominant_wavelength_mm(modes[:, index], pixels) for index in range(archived)]
    # Share of a mode carried by its spatial mean. A near-uniform eigenstrain is
    # the most efficiently transmitted to the interior under fixed boundaries,
    # so this distinguishes "a few long-wavelength modes" from "a few uniform
    # tensor components", which are very different statements.
    uniform_fractions = []
    for index in range(archived):
        field = modes[:, index].reshape(pixels, pixels, 2, 3).mean(axis=2)
        mean = field.mean(axis=(0, 1))
        uniform_fractions.append(
            float((mean**2).sum() * pixels * pixels / max((field**2).sum(), 1e-300))
        )

    report = {
        "schema_version": 1,
        "mesh": [pixels, pixels],
        "pixel_size_mm": PIXEL_SIZE_MM,
        "dimensions": {
            "plastic_components": plastic_size,
            "mechanical_free_dofs": free_size,
            "forcing_rank": forcing_rank,
            "self_equilibrated_kernel": plastic_size - forcing_rank,
            "observed_components": observation_count,
        },
        "gauge": {
            "local_metric": "inverse von Mises, H_loc = M^-1 / point_count",
            "reason": "n^T M^-1 n = 1, so the norm of Delta p n is Delta p",
            "maximum_relative_gauge_norm_error": float(np.max(np.abs(gauge_norms - 1.0))),
        },
        "checks": {
            "strain_divergence_adjoint_relative_error": adjoint_error,
            "schur_is_positive_definite": True,
            "stiffness_is_positive_definite": True,
        },
        "reference_amplitudes": REFERENCE_AMPLITUDES,
        "spectrum": {
            "eigenvalues": eigenvalues.tolist(),
            "square_roots": singular_values.tolist(),
            "normalised": (eigenvalues / eigenvalues[0]).tolist(),
            "cumulated_energy": cumulated.tolist(),
            "signal_to_noise": {name: value.tolist() for name, value in signal_to_noise.items()},
            "effective_rank": effective_rank,
        },
        "modes_above_noise": {
            name: {
                str(threshold): int(np.count_nonzero(value > threshold))
                for threshold in (1.0, 3.0, 10.0)
            }
            for name, value in signal_to_noise.items()
        },
        "modes_for_cumulated_energy": {
            str(share): int(np.searchsorted(cumulated, share) + 1)
            for share in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "archived_mode_dominant_wavelength_mm": wavelengths,
        "archived_mode_uniform_fraction": uniform_fractions,
    }

    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        arguments.output / "modes.npz",
        modes=modes,
        eigenvalues=eigenvalues,
        signal_to_noise=signal_to_noise,
    )

    print(f"plastic components         : {plastic_size}")
    print(f"mechanical free dofs       : {free_size}")
    print(f"rank(G)                    : {forcing_rank}")
    print(f"dim ker(G)                 : {plastic_size - forcing_rank}")
    print(f"adjoint check              : {adjoint_error:.3e}")
    print(f"mode H-orthonormality      : {float(np.max(np.abs(gauge_norms - 1.0))):.3e}")
    print(f"effective rank             : {effective_rank:.2f}")
    for name, value in signal_to_noise.items():
        counts = " ".join(
            f"{threshold:g}:{int(np.count_nonzero(value > threshold))}"
            for threshold in (1.0, 3.0, 10.0)
        )
        print(f"modes above noise ({name:28s}) : {counts}")
    print("\n  j |      sqrt(lambda) |  SNR increment |  SNR cumulated | wavelen mm | uniform")
    for index in range(min(12, len(singular_values))):
        print(
            f"{index + 1:3d} | {singular_values[index]:16.6e} | "
            f"{signal_to_noise['single_increment_rms'][index]:14.4f} | "
            f"{signal_to_noise['accumulated_rms_final_state'][index]:14.4f} | "
            f"{wavelengths[index]:10.4f} | {uniform_fractions[index]:7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
