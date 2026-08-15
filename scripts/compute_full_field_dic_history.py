#!/usr/bin/env python3
"""Full-field DIC displacement history for the whole 42-image sequence.

The repository holds one measured displacement state, the final one, and a
41-state history on a 661x611 computational crop. Neither is the sequence a
convolutional study of the strain morphology needs. The raw material for it has
been there all along -- forty-two 5400x4400 speckle images -- and this turns
them into the missing artefact.

Every field is referred to `000294.tif`, the undeformed reference, and never to
the preceding frame: an incremental chain accumulates its own registration
error, and the quantity wanted at state n is the displacement since the
beginning, not since state n-1.

DISFlow runs on the **whole** image and the crop is taken afterwards. Cropping
first would put the correlation window against an artificial border and
contaminate exactly the region of interest.

## Two things this produces for free

`000335.tif` repeats the final state. Its field is therefore pure metrology --
two acquisitions of one mechanical state -- and it is the **noise floor on the
full field**, the denominator that every error ratio in this project has so far
lacked. It is stored alongside the history rather than mixed into it.

The reference against itself is computed rather than assumed to be zero, so the
algorithm's own fixed-point error is on the record.

## Component convention is measured, not asserted

The experiment-specific mapping is `U = u_y` along the traction axis and
`V = u_x` transverse, and the inventory warns that generic `x`/`y` attributes
found in one HDF5 export must not override it. Rather than trust a reading of
that note, the arrays are stored under names that cannot be misread --
displacement along image rows and along image columns -- and the state-40 field
is correlated against the received `U_40.npy` and `V_40.npy` to determine which
is which, with the outcome written into the file's attributes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/DIC_images")
RECEIVED_ROOT = ROOT / "data/raw/case_study"
#: `rows[400:4000], columns[1211:4311]` of the 4400x5400 raw image is the
#: 3600x3100 support of every prepared field in the repository.
CROP_ROWS = (400, 4000)
CROP_COLUMNS = (1211, 4311)
PIXEL_SIZE_UM = 1.84

#: Repository defaults, `src/fem_inhouse/measurement/disflow.py`. `finest_scale`
#: must stay at zero: one is what produced the archived measurement-chain
#: artefact, because it skips full-resolution variational refinement.
DISFLOW_SETTINGS = {
    "preset": "medium",
    "finest_scale": 0,
    "gradient_descent_iterations": 30,
    "patch_size": 8,
    "patch_stride": 3,
    "variational_refinement_alpha": 100.0,
    "variational_refinement_delta": 1.0,
    "variational_refinement_gamma": 0.0,
    "variational_refinement_epsilon": 0.002,
    "variational_refinement_iterations": 30,
}


def _build_flow() -> cv2.DISOpticalFlow:
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow.setFinestScale(DISFLOW_SETTINGS["finest_scale"])
    flow.setGradientDescentIterations(DISFLOW_SETTINGS["gradient_descent_iterations"])
    flow.setPatchSize(DISFLOW_SETTINGS["patch_size"])
    flow.setPatchStride(DISFLOW_SETTINGS["patch_stride"])
    flow.setVariationalRefinementAlpha(DISFLOW_SETTINGS["variational_refinement_alpha"])
    flow.setVariationalRefinementDelta(DISFLOW_SETTINGS["variational_refinement_delta"])
    flow.setVariationalRefinementGamma(DISFLOW_SETTINGS["variational_refinement_gamma"])
    flow.setVariationalRefinementEpsilon(DISFLOW_SETTINGS["variational_refinement_epsilon"])
    flow.setVariationalRefinementIterations(
        DISFLOW_SETTINGS["variational_refinement_iterations"]
    )
    return flow


def _read_back(flow: cv2.DISOpticalFlow) -> dict[str, float]:
    """What OpenCV actually applied, not what was requested.

    A `set*` call that a build silently ignores is the kind of failure that
    survives into every downstream result, so the applied values travel with
    the data.
    """

    return {
        "finest_scale": float(flow.getFinestScale()),
        "gradient_descent_iterations": float(flow.getGradientDescentIterations()),
        "patch_size": float(flow.getPatchSize()),
        "patch_stride": float(flow.getPatchStride()),
        "variational_refinement_alpha": float(flow.getVariationalRefinementAlpha()),
        "variational_refinement_delta": float(flow.getVariationalRefinementDelta()),
        "variational_refinement_gamma": float(flow.getVariationalRefinementGamma()),
        "variational_refinement_epsilon": float(flow.getVariationalRefinementEpsilon()),
        "variational_refinement_iterations": float(
            flow.getVariationalRefinementIterations()
        ),
    }


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _identify_components(
    rows: FloatArray, columns: FloatArray
) -> dict[str, object]:
    """Correlate the computed state-40 crop against the received arrays."""

    received = {}
    for name in ("U_40", "V_40"):
        path = RECEIVED_ROOT / f"{name}.npy"
        if path.exists():
            received[name] = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
    if not received:
        return {"status": "the received U_40/V_40 arrays are absent, mapping unverified"}
    outcome: dict[str, object] = {"status": "correlated against the received arrays"}
    for name, reference in received.items():
        for label, candidate in (("along_rows", rows), ("along_columns", columns)):
            if reference.shape != candidate.shape:
                outcome[f"{name}_vs_{label}"] = f"shape mismatch {reference.shape}"
                continue
            correlation = float(
                np.corrcoef(reference.ravel(), candidate.ravel())[0, 1]
            )
            scale = float(
                np.dot(reference.ravel(), candidate.ravel())
                / max(float(np.dot(candidate.ravel(), candidate.ravel())), 1.0e-30)
            )
            outcome[f"{name}_vs_{label}"] = {
                "correlation": correlation,
                "least_squares_scale": scale,
            }
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compression-level", type=int, default=4)
    # The repository default is patch 8 / stride 3; the legacy source supplied
    # with the experiment used 4 / 1. Displacement agrees between them to
    # 0.9999, but EVM only to 0.72 -- the pair sets the spatial resolution, and
    # the derivative is where that shows. Both are reachable from here so the
    # price of resolution stays measurable.
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    # The tuned settings. Alpha 100 leaves no structure below twenty pixels;
    # thirty iterations leave the matching grid the refinement exists to erase;
    # epsilon at 1e-3 sits close enough to total variation to produce staircase
    # bands that a derivative turns into spurious localisation. None of these
    # was chosen to resemble the received fields, which are themselves
    # unconverged -- they are the settings that leave no identifiable artefact.
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--refinement-iterations", type=int, default=None)
    arguments = parser.parse_args()

    images = sorted(arguments.image_root.glob("*.tif"))
    if len(images) != 42:
        raise SystemExit(f"expected 42 images, found {len(images)} in {arguments.image_root}")
    reference_path = images[0]
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise SystemExit(f"could not read {reference_path}")
    print(f"reference {reference_path.name}: {reference.shape} {reference.dtype}", flush=True)

    if arguments.patch_size is not None:
        DISFLOW_SETTINGS["patch_size"] = arguments.patch_size
    if arguments.patch_stride is not None:
        DISFLOW_SETTINGS["patch_stride"] = arguments.patch_stride
    if arguments.alpha is not None:
        DISFLOW_SETTINGS["variational_refinement_alpha"] = arguments.alpha
    if arguments.epsilon is not None:
        DISFLOW_SETTINGS["variational_refinement_epsilon"] = arguments.epsilon
    if arguments.refinement_iterations is not None:
        DISFLOW_SETTINGS["variational_refinement_iterations"] = arguments.refinement_iterations

    row_slice = slice(*CROP_ROWS)
    column_slice = slice(*CROP_COLUMNS)
    crop_shape = (CROP_ROWS[1] - CROP_ROWS[0], CROP_COLUMNS[1] - CROP_COLUMNS[0])
    flow = _build_flow()
    applied = _read_back(flow)
    print(f"applied DISFlow settings: {applied}", flush=True)

    # Frames 0..40 are the reference and the forty monotonic steps; frame 41 is
    # the repeat of the final state and is kept apart as the null test.
    state_count = 41
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    chunks = (1, min(512, crop_shape[0]), min(512, crop_shape[1]))
    with h5py.File(arguments.output, "w") as handle:
        group = handle.create_group("displacement_pixel")
        datasets = {
            name: group.create_dataset(
                name,
                shape=(state_count, *crop_shape),
                dtype=np.float32,
                chunks=chunks,
                compression="gzip",
                compression_opts=arguments.compression_level,
            )
            for name in ("along_rows", "along_columns")
        }
        null_group = handle.create_group("null_test_pixel")
        null_datasets = {
            name: null_group.create_dataset(
                name,
                shape=crop_shape,
                dtype=np.float32,
                chunks=chunks[1:],
                compression="gzip",
                compression_opts=arguments.compression_level,
            )
            for name in ("along_rows", "along_columns")
        }

        statistics = []
        state_forty: dict[str, FloatArray] = {}
        for index, path in enumerate(images):
            started = time.perf_counter()
            target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if target is None or target.shape != reference.shape:
                raise SystemExit(f"unreadable or mis-shaped image {path}")
            field = flow.calc(reference, target, None)
            # OpenCV orders the flow (x, y) with x along columns.
            along_columns = field[row_slice, column_slice, 0]
            along_rows = field[row_slice, column_slice, 1]
            if index < state_count:
                datasets["along_rows"][index] = along_rows
                datasets["along_columns"][index] = along_columns
            else:
                null_datasets["along_rows"][...] = along_rows
                null_datasets["along_columns"][...] = along_columns
            if index == state_count - 1:
                state_forty = {
                    "along_rows": np.asarray(along_rows, dtype=np.float64),
                    "along_columns": np.asarray(along_columns, dtype=np.float64),
                }
            entry = {
                "index": index,
                "file": path.name,
                "role": (
                    "reference"
                    if index == 0
                    else "null_test_repeat"
                    if index >= state_count
                    else f"step_{index}"
                ),
                "rms_along_rows_pixel": float(np.sqrt(np.mean(along_rows.astype(np.float64) ** 2))),
                "rms_along_columns_pixel": float(
                    np.sqrt(np.mean(along_columns.astype(np.float64) ** 2))
                ),
                "seconds": time.perf_counter() - started,
            }
            statistics.append(entry)
            print(
                f"  {entry['role']:<18} {path.name}  rms rows "
                f"{entry['rms_along_rows_pixel']:.4f} px  columns "
                f"{entry['rms_along_columns_pixel']:.4f} px  "
                f"{entry['seconds']:.1f} s",
                flush=True,
            )

        mapping = _identify_components(
            state_forty["along_rows"], state_forty["along_columns"]
        )
        handle.attrs["description"] = (
            "DISFlow displacement history of the full 42-image P43 sequence, every "
            "field referred to the undeformed reference 000294.tif, computed on the "
            "whole 4400x5400 image and cropped afterwards"
        )
        handle.attrs["units"] = "pixel"
        handle.attrs["pixel_size_um"] = PIXEL_SIZE_UM
        handle.attrs["crop_rows"] = CROP_ROWS
        handle.attrs["crop_columns"] = CROP_COLUMNS
        handle.attrs["raw_image_shape"] = reference.shape
        handle.attrs["reference_image"] = reference_path.name
        handle.attrs["reference_sha256"] = _digest(reference_path)
        handle.attrs["opencv_version"] = cv2.__version__
        handle.attrs["disflow_requested"] = json.dumps(DISFLOW_SETTINGS)
        handle.attrs["disflow_applied"] = json.dumps(applied)
        handle.attrs["component_mapping"] = json.dumps(mapping, indent=1)
        handle.attrs["component_note"] = (
            "arrays are named by image axis, not by mechanical axis; the "
            "experiment-specific convention is U = u_y along traction and V = u_x "
            "transverse, and component_mapping records the correlation of the "
            "state-40 field against the received U_40.npy and V_40.npy"
        )
        handle.attrs["frames"] = json.dumps(statistics, indent=1)
    print(f"\ncomponent mapping: {json.dumps(mapping, indent=1)}")
    print(f"wrote {arguments.output} ({arguments.output.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
