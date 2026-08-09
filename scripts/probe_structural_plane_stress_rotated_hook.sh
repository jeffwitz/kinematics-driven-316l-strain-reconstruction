#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
probe_source="$repo_root/validation/mfront/StructuralPlaneStressRotatedElasticProbe.mfront"
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/structural-plane-stress-rotated.XXXXXX")
set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u
cd "$probe_dir"
mfront --obuild --interface=generic "$probe_source" >/dev/null

python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
LIBRARY="$probe_dir/src/libBehaviour.so" "$python_bin" - <<'PY'
import os

import mgis.behaviour as mgis
import numpy as np

behaviour = mgis.load(
    os.environ["LIBRARY"],
    "StructuralPlaneStressRotatedElasticProbe",
    mgis.Hypothesis.Tridimensional,
)
data = mgis.MaterialDataManager(behaviour, 1)
for state in (data.s0, data.s1):
    mgis.setExternalStateVariable(state, "Temperature", 293.15)
data.s1.gradients[0] = np.array([1.0e-3, -2.0e-4, 0.0, 3.0e-4, 0.0, 0.0])
mgis.integrate(
    data,
    mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
    1.0,
    0,
    1,
)
values = np.asarray(data.s1.thermodynamic_forces[0])
stress = np.array(
    [
        [values[0], values[3] / np.sqrt(2), values[4] / np.sqrt(2)],
        [values[3] / np.sqrt(2), values[1], values[5] / np.sqrt(2)],
        [values[4] / np.sqrt(2), values[5] / np.sqrt(2), values[2]],
    ]
)
q_global_to_material = np.array(
    [
        [0.6517403912340062, 0.7532585459971657, 0.08852132690137686],
        [-0.7326322075147665, 0.5950699920075869, 0.33036608954935215],
        [0.19617469496901108, -0.2801664995932355, 0.9396926207859084],
    ]
)
assert np.max(np.abs(q_global_to_material @ q_global_to_material.T - np.eye(3))) < 1.0e-12
assert abs(np.linalg.det(q_global_to_material) - 1.0) < 1.0e-12
material_to_global = q_global_to_material.T
global_stress = material_to_global @ stress @ material_to_global.T
transverse = np.abs(
    [global_stress[2, 2], global_stress[1, 2], global_stress[0, 2]]
)
if float(np.max(transverse)) > 1.0e-7:
    raise SystemExit(f"rotated closure probe failed: transverse stress={transverse}")
internal = np.asarray(data.s1.internal_state_variables[0])
structural_total_strain = internal[6:12]
imposed = np.asarray(data.s1.gradients[0])
in_plane_error = np.max(np.abs(structural_total_strain[[0, 1, 3]] - imposed[[0, 1, 3]]))
if float(in_plane_error) > 1.0e-12:
    raise SystemExit(f"rotated kinematics probe failed: error={in_plane_error}")
print(
    "rotated structural closure probe: passed "
    f"(max transverse stress={np.max(transverse):.3e}, "
    f"max in-plane strain error={in_plane_error:.3e})"
)
PY
