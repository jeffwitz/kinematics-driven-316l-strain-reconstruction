#!/usr/bin/env python3
"""What does the DISFlow correlation window cost, and what is the noise floor?

Two profiles reach the same measurement. The repository default uses patch 8
with stride 3; the legacy source supplied with the experiment used 4 and 1. On
displacement they agree to 0.9999, which is why the discrepancy went unnoticed:
it lives entirely in the derivative, and the derivative is the whole subject.

Three questions, answered against the received `U_40.npy` and `V_40.npy` rather
than against each other:

1. **Which profile produced the archived fields?** Every result in this project
   descends from them, so mixing resolutions would make every comparison
   ambiguous.
2. **What does the window cost in spatial resolution?** Measured by the scale at
   which the two EVM fields converge once both are smoothed, and by the radial
   power spectrum, which localises the disagreement in frequency instead of
   asserting it.
3. **What is the noise floor at each resolution?** `000335.tif` repeats the
   final mechanical state, so the difference between it and state 40 is pure
   metrology. A finer window resolves more real structure *and* more noise, and
   only the ratio decides whether the extra resolution is worth having.

The last point is the one that matters for everything downstream: a morphology
study is only meaningful above its own noise floor, and until now this project
has divided by an unknown denominator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
RECEIVED_ROOT = ROOT / "data/raw/case_study"


def equivalent_strain(along_rows: FloatArray, along_columns: FloatArray) -> FloatArray:
    """Von Mises equivalent of the small-strain tensor, incompressible closure.

    First differences on the pixel grid, trimmed to a common shape. `eps_zz` is
    taken as `-(eps_rr + eps_cc)`, which is a display convention for a total
    strain rather than a statement about the elastic part, and it is the same
    convention on both sides of every comparison here.
    """

    err = np.diff(along_rows, axis=0)[:, :-1]
    ecc = np.diff(along_columns, axis=1)[:-1, :]
    erc = 0.5 * (
        np.diff(along_rows, axis=1)[:-1, :] + np.diff(along_columns, axis=0)[:, :-1]
    )
    ezz = -(err + ecc)
    return np.sqrt(2.0 / 3.0 * (err**2 + ecc**2 + ezz**2 + 2.0 * erc**2))


def _correlation(first: FloatArray, second: FloatArray) -> float:
    return float(np.corrcoef(first.ravel(), second.ravel())[0, 1])


def _radial_spectrum(field: FloatArray, bins: int = 60) -> tuple[FloatArray, FloatArray]:
    """Azimuthally averaged power spectrum on a decimated central window.

    A 3600x3100 transform is affordable but the comparison only needs the shape
    of the spectrum, and a central window avoids the crop edges where the
    correlation window is one-sided.
    """

    window = field[
        field.shape[0] // 2 - 1024 : field.shape[0] // 2 + 1024,
        field.shape[1] // 2 - 1024 : field.shape[1] // 2 + 1024,
    ]
    window = window - window.mean()
    power = np.abs(np.fft.rfft2(window)) ** 2
    rows = np.fft.fftfreq(window.shape[0])[:, None]
    columns = np.fft.rfftfreq(window.shape[1])[None, :]
    radius = np.sqrt(rows**2 + columns**2)
    edges = np.linspace(0.0, 0.5, bins + 1)
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, bins - 1)
    total = np.bincount(index, weights=power.ravel(), minlength=bins)
    count = np.bincount(index, minlength=bins)
    return 0.5 * (edges[:-1] + edges[1:]), total / np.maximum(count, 1)


def _load(path: Path, state: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    with h5py.File(path, "r") as handle:
        rows = np.asarray(handle["displacement_pixel/along_rows"][state], dtype=np.float64)
        columns = np.asarray(
            handle["displacement_pixel/along_columns"][state], dtype=np.float64
        )
        null_rows = np.asarray(handle["null_test_pixel/along_rows"], dtype=np.float64)
        null_columns = np.asarray(handle["null_test_pixel/along_columns"], dtype=np.float64)
    return rows, columns, null_rows, null_columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--state", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    received_rows = np.asarray(np.load(RECEIVED_ROOT / "V_40.npy"), dtype=np.float64)
    received_columns = np.asarray(np.load(RECEIVED_ROOT / "U_40.npy"), dtype=np.float64)
    received = equivalent_strain(received_rows, received_columns)

    profiles: dict[str, dict[str, object]] = {}
    fields: dict[str, FloatArray] = {}
    candidates = (
        ("coarse_patch8_stride3", arguments.coarse),
        ("fine_patch4_stride1", arguments.fine),
    )
    for label, path in candidates:
        rows, columns, null_rows, null_columns = _load(path, arguments.state)
        signal = equivalent_strain(rows, columns)
        # Same mechanical state twice, so the difference is metrology alone.
        noise = equivalent_strain(null_rows - rows, null_columns - columns)
        displacement_noise = float(
            np.sqrt(np.mean((null_rows - rows) ** 2 + (null_columns - columns) ** 2))
        )
        fields[label] = signal
        profiles[label] = {
            "displacement_correlation_with_received": {
                "along_rows_vs_V_40": _correlation(rows, received_rows),
                "along_columns_vs_U_40": _correlation(columns, received_columns),
            },
            "evm_correlation_with_received": _correlation(signal, received),
            "evm_rms": float(np.sqrt((signal**2).mean())),
            "evm_rms_ratio_to_received": float(
                np.sqrt((signal**2).mean()) / np.sqrt((received**2).mean())
            ),
            "displacement_noise_rms_pixel": displacement_noise,
            "evm_noise_rms": float(np.sqrt((noise**2).mean())),
            "noise_to_signal": float(
                np.sqrt((noise**2).mean()) / np.sqrt((signal**2).mean())
            ),
        }
        print(
            f"{label}: EVM corr with received "
            f"{profiles[label]['evm_correlation_with_received']:.4f}, amplitude ratio "
            f"{profiles[label]['evm_rms_ratio_to_received']:.4f}, "
            f"noise/signal {profiles[label]['noise_to_signal']:.4f}",
            flush=True,
        )

    # At what scale do the two profiles agree? Smoothing both and watching the
    # correlation climb says where the disagreement lives, rather than asserting
    # that it is "fine structure".
    convergence = []
    for sigma in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        coarse = fields["coarse_patch8_stride3"]
        fine = fields["fine_patch4_stride1"]
        if sigma > 0.0:
            coarse = gaussian_filter(coarse, sigma)
            fine = gaussian_filter(fine, sigma)
        convergence.append(
            {
                "sigma_pixel": sigma,
                "correlation": _correlation(coarse, fine),
                "rms_ratio_coarse_over_fine": float(
                    np.sqrt((coarse**2).mean()) / np.sqrt((fine**2).mean())
                ),
            }
        )
        print(
            f"  sigma {sigma:4.0f} px: profiles agree at "
            f"{convergence[-1]['correlation']:.4f}",
            flush=True,
        )

    frequency, coarse_power = _radial_spectrum(fields["coarse_patch8_stride3"])
    _, fine_power = _radial_spectrum(fields["fine_patch4_stride1"])
    _, received_power = _radial_spectrum(received)

    report = {
        "schema_version": 1,
        "status": "completed_disflow_profile_comparison",
        "state": arguments.state,
        "profiles": profiles,
        "scale_convergence": convergence,
        "radial_spectrum": {
            "cycles_per_pixel": frequency.tolist(),
            "coarse_patch8_stride3": coarse_power.tolist(),
            "fine_patch4_stride1": fine_power.tolist(),
            "received": received_power.tolist(),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
