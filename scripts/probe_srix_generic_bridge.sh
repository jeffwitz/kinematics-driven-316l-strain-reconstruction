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

from fem_inhouse.core.mfront import (
    SrixGeneric3DCondensedPlaneStressBatch,
    SrixGeneric3DMaterialPointBatch,
)
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
condensed_bridge = SrixGeneric3DMaterialPointBatch(
    os.environ["LIBRARY"],
    point_count=2,
    micromorphic_coupling_modulus_mpa=100.0,
)
condensed = SrixGeneric3DCondensedPlaneStressBatch(condensed_bridge)
in_plane = 0.1 * strain[:, [0, 1, 3]]
plane_trial = condensed.evaluate(in_plane, np.array([2.0e-4, 1.0e-4]), time_increment=1.0)
assert plane_trial.tangent_in_plane_mpa.shape == (2, 3, 3)
assert np.max(np.abs(plane_trial.transverse_stress_mpa)) < 1.0e-8
condensed.commit()
chi_value = np.array([2.0e-4, 1.0e-4])
protocol_bridge = SrixGeneric3DMaterialPointBatch(
    os.environ["LIBRARY"],
    point_count=2,
    micromorphic_coupling_modulus_mpa=100.0,
)
protocol_adapter = SrixGeneric3DCondensedPlaneStressBatch(protocol_bridge)
protocol_adapter.set_nonlocal_equivalent_plastic_strain(chi_value)
protocol_trial = protocol_adapter.evaluate_in_plane(
    in_plane,
    time_increment=1.0,
    consistent_tangent=True,
)
assert protocol_trial.tangent_in_plane_mpa is not None
assert "accumulated_slip" in protocol_trial.observables
protocol_complete = protocol_adapter.complete_trial(protocol_trial)
assert protocol_complete.full_stress_tensor_mpa.shape == (2, 3, 3)
assert protocol_adapter.statistics.maximum_local_plane_stress_iterations >= 1
assert protocol_adapter.timing_statistics.evaluate_calls >= 1
protocol_adapter.commit()


def response(in_plane_value, chi_value):
    bridge = SrixGeneric3DMaterialPointBatch(
        os.environ["LIBRARY"],
        point_count=2,
        micromorphic_coupling_modulus_mpa=100.0,
    )
    adapter = SrixGeneric3DCondensedPlaneStressBatch(bridge)
    result = adapter.evaluate(in_plane_value, chi_value, time_increment=1.0)
    return result


fd_mechanical = np.zeros((2, 3, 3))
fd_stress_chi = np.zeros((2, 3, 1))
fd_source_strain = np.zeros((2, 1, 3))
fd_source_chi = np.zeros((2, 1, 1))
for column in range(4):
    h = 1.0e-7
    plus_in_plane = in_plane.copy()
    minus_in_plane = in_plane.copy()
    plus_chi = chi_value.copy()
    minus_chi = chi_value.copy()
    if column < 3:
        plus_in_plane[:, column] += h
        minus_in_plane[:, column] -= h
    else:
        plus_chi += h
        minus_chi -= h
    plus = response(plus_in_plane, plus_chi)
    minus = response(minus_in_plane, minus_chi)
    if column < 3:
        fd_mechanical[:, :, column] = (plus.stress_in_plane_mpa - minus.stress_in_plane_mpa) / (2.0 * h)
        fd_source_strain[:, :, column] = (plus.accumulated_slip - minus.accumulated_slip)[:, None] / (2.0 * h)
    else:
        fd_stress_chi[:, :, 0] = (plus.stress_in_plane_mpa - minus.stress_in_plane_mpa) / (2.0 * h)
        fd_source_chi[:, :, 0] = (plus.accumulated_slip - minus.accumulated_slip)[:, None] / (2.0 * h)

def relative_error(reference, value):
    return np.linalg.norm(reference - value) / max(np.linalg.norm(reference), 1.0e-30)

errors = [
    relative_error(fd_mechanical, plane_trial.tangent_in_plane_mpa),
    relative_error(fd_stress_chi, plane_trial.stress_chi_tangent_mpa),
    relative_error(fd_source_strain, plane_trial.accumulated_slip_strain_tangent),
    relative_error(fd_source_chi, plane_trial.accumulated_slip_chi_tangent),
]
assert max(errors) < 1.0e-5, errors
print("SRIX Generic 3-D bridge: passed")
print(f"stress_norm={np.linalg.norm(trial.stress_kelvin_mpa):.6e}")
print(f"source_norm={np.linalg.norm(trial.accumulated_slip):.6e}")
print(f"rotated_stress_norm={np.linalg.norm(rotated_trial.stress_kelvin_mpa):.6e}")
print(f"plane_stress_residual={np.max(np.abs(plane_trial.transverse_stress_mpa)):.6e}")
print(f"plane_stress_four_block_fd_error={max(errors):.6e}")
PY
