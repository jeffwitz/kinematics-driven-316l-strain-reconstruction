#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
probe_source="$repo_root/validation/mfront/StructuralPlaneStressJ2Probe.mfront"
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/structural-plane-stress-j2.XXXXXX")
set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u

if rg -n "dg|Nss|ForestRubin|Meric|PlasticSlip" "$probe_source"; then
  echo "J2 probe contains a law-specific closure symbol" >&2
  exit 1
fi

cd "$probe_dir"
mfront --obuild --interface=generic "$probe_source" >/dev/null
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
LIBRARY="$probe_dir/src/libBehaviour.so" "$python_bin" - <<'PY'
import os

import mgis.behaviour as mgis
import numpy as np

behaviour = mgis.load(
    os.environ["LIBRARY"],
    "StructuralPlaneStressJ2Probe",
    mgis.Hypothesis.Tridimensional,
)
q_global_to_material = np.array(
    [
        [0.6517403912340062, 0.7532585459971657, 0.08852132690137686],
        [-0.7326322075147665, 0.5950699920075869, 0.33036608954935215],
        [0.19617469496901108, -0.2801664995932355, 0.9396926207859084],
    ]
)
assert np.max(np.abs(q_global_to_material @ q_global_to_material.T - np.eye(3))) < 1e-12
material_to_global = q_global_to_material.T


def to_kelvin(values: np.ndarray) -> np.ndarray:
    material_tensor = np.array(
        [
            [values[0], values[3] / np.sqrt(2), values[4] / np.sqrt(2)],
            [values[3] / np.sqrt(2), values[1], values[5] / np.sqrt(2)],
            [values[4] / np.sqrt(2), values[5] / np.sqrt(2), values[2]],
        ]
    )
    global_tensor = material_to_global @ material_tensor @ material_to_global.T
    return np.array(
        [
            global_tensor[0, 0],
            global_tensor[1, 1],
            global_tensor[2, 2],
            np.sqrt(2) * global_tensor[0, 1],
            np.sqrt(2) * global_tensor[0, 2],
            np.sqrt(2) * global_tensor[1, 2],
        ]
    )


def integrate(strain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mgis.MaterialDataManager(behaviour, 1)
    for state in (data.s0, data.s1):
        mgis.setExternalStateVariable(state, "Temperature", 293.15)
    data.s1.gradients[0] = strain
    result = mgis.integrate(
        data,
        mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
        1.0,
        0,
        1,
    )
    if result != 1:
        raise SystemExit(f"J2 integration failed: result={result}")
    return (
        to_kelvin(np.asarray(data.s1.thermodynamic_forces[0])),
        np.asarray(data.K[0]).copy(),
        np.asarray(data.s1.internal_state_variables[0]).copy(),
    )


base = np.array([1.0e-2, -2.0e-3, 0.0, 3.0e-3, 0.0, 0.0])
stress, tangent, internal = integrate(base)
transverse = np.abs(stress[[2, 4, 5]])
if float(np.max(transverse)) > 1.0e-8:
    raise SystemExit(f"J2 plane-stress closure failed: {transverse}")

in_plane = np.array([0, 1, 3])
fd_errors = []
for step in (1.0e-5, 1.0e-6, 1.0e-7):
    fd = np.zeros((3, 3))
    for column, component in enumerate(in_plane):
        plus = base.copy()
        minus = base.copy()
        plus[component] += step
        minus[component] -= step
        plus_stress, _, _ = integrate(plus)
        minus_stress, _, _ = integrate(minus)
        fd[:, column] = (plus_stress[in_plane] - minus_stress[in_plane]) / (2.0 * step)
    error = np.max(np.abs(fd - tangent[np.ix_(in_plane, in_plane)])) / np.max(
        np.abs(tangent[np.ix_(in_plane, in_plane)])
    )
    fd_errors.append(float(error))
    if error > 1.0e-5:
        raise SystemExit(f"J2 tangent failed for h={step}: relative error={error}")

inactive_columns = np.delete(np.arange(6), in_plane)
inactive_error = float(np.max(np.abs(tangent[:, inactive_columns])))
print(
    "generic J2 structural plane-stress probe: passed "
    f"(max transverse stress={np.max(transverse):.3e}, "
    f"FD errors={fd_errors}, max inactive-column={inactive_error:.3e}, "
    f"internal-state-size={internal.size})"
)
PY
