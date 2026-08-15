#!/usr/bin/env python3
"""The noise floor at the tuned settings, from the repeated final state.

`000335.tif` is a second acquisition of the mechanical state `000334.tif`
already holds, so a field correlated from one to the other through the same
reference differs from the state-40 field by metrology alone -- no mechanics,
no model, no assumption about what the specimen did.

This runs only that pair, so the number is available in three minutes instead of
after the full forty-two-image history. It is the number every error ratio in
this project has been divided by without knowing it, and the one measured
earlier, 0.161 on the equivalent strain, does not transfer: it was obtained at
alpha 100 where the fields carry no structure below twenty pixels, and a
setting that resolves fine structure resolves fine noise with it. Whether the
recovered granular texture is physics or metrology is exactly what this
decides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from compare_disflow_profiles_p43 import equivalent_strain  # type: ignore[import-not-found]
from tune_disflow_alpha_against_received import _flow_field  # type: ignore[import-not-found]

IMAGE_ROOT = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/DIC_images")
PIXEL_SIZE_UM = 1.84


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--patch-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    images = sorted(IMAGE_ROOT.glob("*.tif"))
    reference = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    settings = dict(
        alpha=arguments.alpha,
        iterations=arguments.iterations,
        patch_size=arguments.patch_size,
        patch_stride=arguments.patch_stride,
        epsilon=arguments.epsilon,
    )
    fields = {}
    for label, index in (("state_40", 40), ("repeat", 41)):
        fields[label] = _flow_field(
            reference, cv2.imread(str(images[index]), cv2.IMREAD_GRAYSCALE), **settings
        )
        print(f"  {label} ({images[index].name}) done", flush=True)

    rows_noise = fields["repeat"][0] - fields["state_40"][0]
    columns_noise = fields["repeat"][1] - fields["state_40"][1]
    displacement_noise = float(np.sqrt(np.mean(rows_noise**2 + columns_noise**2)))

    signal = equivalent_strain(*fields["state_40"])
    noise = equivalent_strain(rows_noise, columns_noise)

    def components(along_rows, along_columns):
        err = np.diff(along_rows, axis=0)[:, :-1]
        ecc = np.diff(along_columns, axis=1)[:-1, :]
        erc = 0.5 * (
            np.diff(along_rows, axis=1)[:-1, :] + np.diff(along_columns, axis=0)[:, :-1]
        )
        return err, ecc, erc

    signal_components = components(*fields["state_40"])
    noise_components = components(rows_noise, columns_noise)
    per_component = {
        name: {
            "noise_rms": float(np.sqrt((bad**2).mean())),
            "signal_rms": float(np.sqrt((good**2).mean())),
            "ratio": float(np.sqrt((bad**2).mean()) / np.sqrt((good**2).mean())),
        }
        for name, good, bad in zip(
            ("xx", "yy", "xy"), signal_components, noise_components, strict=True
        )
    }

    report = {
        "schema_version": 1,
        "status": "completed_null_test_noise_floor",
        "settings": {**settings, "finest_scale": 0},
        "pair": [images[40].name, images[41].name],
        "displacement_noise_rms_pixel": displacement_noise,
        "displacement_noise_rms_um": displacement_noise * PIXEL_SIZE_UM,
        "equivalent_strain": {
            "noise_rms": float(np.sqrt((noise**2).mean())),
            "signal_rms": float(np.sqrt((signal**2).mean())),
            "noise_to_signal": float(
                np.sqrt((noise**2).mean()) / np.sqrt((signal**2).mean())
            ),
        },
        "per_component": per_component,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(
        f"\ndisplacement noise {displacement_noise:.4f} px "
        f"({displacement_noise * PIXEL_SIZE_UM:.3f} um)"
    )
    for name, entry in per_component.items():
        print(f"  {name}: noise {entry['noise_rms']:.3e}  ratio {entry['ratio']:.3f}")
    print(f"  EVM noise/signal {report['equivalent_strain']['noise_to_signal']:.3f}")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
