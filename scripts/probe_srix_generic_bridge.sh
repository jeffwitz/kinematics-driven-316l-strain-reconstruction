#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/srix-generic-bridge.XXXXXX")
trap 'rm -rf "$work"' EXIT

set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u

"$root/.venv/bin/python" "$root/scripts/generate_srix_generic_3d.py" \
  "$root/mfront/Fcc316LForestRubinSrix.mfront" \
  "$work/Fcc316LForestRubinSrixGeneric3D.mfront"
(cd "$work" && mfront --obuild --interface=generic Fcc316LForestRubinSrixGeneric3D.mfront >/dev/null)

LIBRARY="$work/src/libBehaviour.so" "$root/.venv/bin/python" - <<'PY'
import os

import numpy as np

from fem_inhouse.core.mfront import SrixGeneric3DMaterialPointBatch
from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg

material = SrixGeneric3DMaterialPointBatch(
    os.environ["LIBRARY"],
    point_count=2,
    micromorphic_coupling_modulus_mpa=100.0,
)
strain = np.array(
    [
        [2.0e-3, -6.0e-4, 1.0e-4, 3.0e-4, 0.0, 0.0],
        [1.5e-3, -4.0e-4, 8.0e-5, -2.0e-4, 0.0, 0.0],
    ]
)
trial = material.evaluate(strain, np.array([2.0e-4, 1.0e-4]), time_increment=1.0)
assert trial.stress_kelvin_mpa.shape == (2, 6)
assert trial.accumulated_slip.shape == (2,)
assert trial.stress_strain_tangent.shape == (2, 6, 6)
assert trial.stress_chi_tangent.shape == (2, 6, 1)
assert trial.accumulated_slip_strain_tangent.shape == (2, 1, 6)
assert trial.accumulated_slip_chi_tangent.shape == (2, 1, 1)
assert np.isfinite(trial.stress_kelvin_mpa).all()
material.revert()
trial = material.evaluate(strain, np.array([2.0e-4, 1.0e-4]), time_increment=1.0)
material.commit()
rotated = SrixGeneric3DMaterialPointBatch(
    os.environ["LIBRARY"],
    point_count=2,
    micromorphic_coupling_modulus_mpa=100.0,
    rotation_global_to_material=np.stack(
        [np.eye(3), rotation_from_euler_bunge_deg(17.0, 31.0, 43.0)]
    ),
)
rotated_trial = rotated.evaluate(
    strain, np.array([2.0e-4, 1.0e-4]), time_increment=1.0
)
assert np.isfinite(rotated_trial.stress_kelvin_mpa).all()
assert np.isfinite(rotated_trial.stress_strain_tangent).all()
print("SRIX Generic 3-D bridge: passed")
print(f"stress_norm={np.linalg.norm(trial.stress_kelvin_mpa):.6e}")
print(f"source_norm={np.linalg.norm(trial.accumulated_slip):.6e}")
print(f"rotated_stress_norm={np.linalg.norm(rotated_trial.stress_kelvin_mpa):.6e}")
PY
