#!/usr/bin/env python3
"""Which variational smoothing reproduces the received field's spectrum?

At alpha 100 the DISFlow field carries no structure below roughly twenty
pixels, while the received field is dense with it -- thirty to fifty times more
power below a twelve-pixel wavelength. Patch size and stride were shown not to
be responsible: two profiles that differ by a factor of two in window size are
visually indistinguishable. That leaves the variational smoothing weight.

State 40 only, one image pair, so an alpha can be tried in half a minute
instead of twenty. The iteration count stays at the repository value of thirty:
the refinement is iterative and its effective smoothing therefore depends on
both alpha and the iteration budget, so this fixes the budget and varies the
weight alone. Any alpha found here is the answer *at thirty iterations*, which
is what the comparison needs, not a converged fixed point.

The criterion is spectral rather than a single correlation. Two fields can
share a correlation and disagree about where their power sits, and the whole
question is where the power sits. The score is the mean absolute log ratio of
the radial power spectra over the resolvable band, so a value of zero means the
two spectra coincide at every scale and a value of one means they differ by a
factor of e on average.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from compare_disflow_profiles_p43 import (  # type: ignore[import-not-found]
    _radial_spectrum,
    equivalent_strain,
)
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
RECEIVED_ROOT = ROOT / "data/raw/case_study"
IMAGE_ROOT = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/DIC_images")
CROP_ROWS = (400, 4000)
CROP_COLUMNS = (1211, 4311)


def _flow_field(
    reference: FloatArray,
    target: FloatArray,
    *,
    alpha: float,
    iterations: int,
    patch_size: int,
    patch_stride: int,
    epsilon: float = 0.002,
) -> tuple[FloatArray, FloatArray]:
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow.setFinestScale(0)
    flow.setGradientDescentIterations(30)
    flow.setPatchSize(patch_size)
    flow.setPatchStride(patch_stride)
    flow.setVariationalRefinementAlpha(alpha)
    flow.setVariationalRefinementDelta(1.0)
    flow.setVariationalRefinementGamma(0.0)
    flow.setVariationalRefinementEpsilon(epsilon)
    flow.setVariationalRefinementIterations(iterations)
    field = flow.calc(reference, target, None)
    rows = slice(*CROP_ROWS)
    columns = slice(*CROP_COLUMNS)
    return (
        np.asarray(field[rows, columns, 1], dtype=np.float64),
        np.asarray(field[rows, columns, 0], dtype=np.float64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=float, default=[10.0, 3.0, 1.0, 0.3])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--patch-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    received = equivalent_strain(
        np.asarray(np.load(RECEIVED_ROOT / "V_40.npy"), dtype=np.float64),
        np.asarray(np.load(RECEIVED_ROOT / "U_40.npy"), dtype=np.float64),
    )
    frequency, received_power = _radial_spectrum(received)
    received_rms = float(np.sqrt((received**2).mean()))
    # Below two pixels the transform is dominated by the differencing stencil
    # rather than by the measurement, and the first bin carries the removed
    # mean, so the score is taken over the band in between.
    band = (frequency > 0.01) & (frequency < 0.45)

    images = sorted(IMAGE_ROOT.glob("*.tif"))
    reference = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(images[40]), cv2.IMREAD_GRAYSCALE)
    print(f"reference {images[0].name}, state 40 {images[40].name}", flush=True)
    print(
        f"received: EVM rms {received_rms:.4e}, "
        f"patch {arguments.patch_size}/{arguments.patch_stride}, "
        f"{arguments.iterations} refinement iterations held fixed\n",
        flush=True,
    )

    results = []
    for alpha in arguments.alphas:
        started = time.perf_counter()
        rows, columns = _flow_field(
            reference,
            target,
            alpha=alpha,
            iterations=arguments.iterations,
            patch_size=arguments.patch_size,
            patch_stride=arguments.patch_stride,
        )
        candidate = equivalent_strain(rows, columns)
        _, power = _radial_spectrum(candidate)
        ratio = np.log(
            np.maximum(power[band], 1.0e-300) / np.maximum(received_power[band], 1.0e-300)
        )
        entry = {
            "alpha": alpha,
            "evm_rms": float(np.sqrt((candidate**2).mean())),
            "evm_rms_ratio": float(np.sqrt((candidate**2).mean()) / received_rms),
            "evm_correlation": float(np.corrcoef(candidate.ravel(), received.ravel())[0, 1]),
            "spectral_score": float(np.mean(np.abs(ratio))),
            "spectral_bias": float(np.mean(ratio)),
            "power_ratio_at_12px": float(
                power[np.argmin(np.abs(frequency - 1.0 / 12.0))]
                / received_power[np.argmin(np.abs(frequency - 1.0 / 12.0))]
            ),
            "power_ratio_at_4px": float(
                power[np.argmin(np.abs(frequency - 0.25))]
                / received_power[np.argmin(np.abs(frequency - 0.25))]
            ),
            "seconds": time.perf_counter() - started,
        }
        results.append(entry)
        print(
            f"alpha {alpha:7.3f}: score {entry['spectral_score']:.3f}  "
            f"corr {entry['evm_correlation']:.4f}  rms ratio {entry['evm_rms_ratio']:.3f}  "
            f"power@12px {entry['power_ratio_at_12px']:7.3f}  "
            f"power@4px {entry['power_ratio_at_4px']:7.3f}  ({entry['seconds']:.0f} s)",
            flush=True,
        )

    best = min(results, key=lambda entry: entry["spectral_score"])
    report = {
        "schema_version": 1,
        "status": "completed_disflow_alpha_tuning",
        "state": 40,
        "refinement_iterations": arguments.iterations,
        "patch_size": arguments.patch_size,
        "patch_stride": arguments.patch_stride,
        "received_evm_rms": received_rms,
        "criterion": "mean absolute log ratio of radial EVM power spectra",
        "results": results,
        "best_alpha": best["alpha"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nbest alpha {best['alpha']} at score {best['spectral_score']:.3f}")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
