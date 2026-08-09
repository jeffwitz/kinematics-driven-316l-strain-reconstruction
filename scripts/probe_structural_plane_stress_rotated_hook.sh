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


def kelvin_operator(rotation: np.ndarray) -> np.ndarray:
    result = np.zeros((6, 6))
    for column in range(6):
        basis = np.zeros((3, 3))
        if column < 3:
            basis[column, column] = 1.0
        elif column == 3:
            basis[0, 1] = basis[1, 0] = 1.0 / np.sqrt(2.0)
        elif column == 4:
            basis[0, 2] = basis[2, 0] = 1.0 / np.sqrt(2.0)
        else:
            basis[1, 2] = basis[2, 1] = 1.0 / np.sqrt(2.0)
        transformed = rotation @ basis @ rotation.T
        result[:, column] = [
            transformed[0, 0],
            transformed[1, 1],
            transformed[2, 2],
            np.sqrt(2.0) * transformed[0, 1],
            np.sqrt(2.0) * transformed[0, 2],
            np.sqrt(2.0) * transformed[1, 2],
        ]
    return result


def integrate(strain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mgis.MaterialDataManager(behaviour, 1)
    for state in (data.s0, data.s1):
        mgis.setExternalStateVariable(state, "Temperature", 293.15)
    data.s1.gradients[0] = strain
    mgis.integrate(
        data,
        mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
        1.0,
        0,
        1,
    )
    return (
        np.asarray(data.s1.thermodynamic_forces[0]).copy(),
        np.asarray(data.K[0]).copy(),
        np.asarray(data.s1.internal_state_variables[0]).copy(),
    )


base_strain = np.array([1.0e-3, -2.0e-4, 0.0, 3.0e-4, 0.0, 0.0])
values, tangent, internal = integrate(base_strain)
stress = np.array(
    [
        [values[0], values[3] / np.sqrt(2), values[4] / np.sqrt(2)],
        [values[3] / np.sqrt(2), values[1], values[5] / np.sqrt(2)],
        [values[4] / np.sqrt(2), values[5] / np.sqrt(2), values[2]],
    ]
)
global_stress = material_to_global @ stress @ material_to_global.T
transverse = np.abs(
    [global_stress[2, 2], global_stress[1, 2], global_stress[0, 2]]
)
if float(np.max(transverse)) > 1.0e-7:
    raise SystemExit(f"rotated closure probe failed: transverse stress={transverse}")
offset = 0
structural_indices = []
for variable in behaviour.internal_state_variables:
    width = 6 if variable.type == mgis.VariableType.Stensor else 1
    if variable.name.startswith("StructuralTotalStrain"):
        structural_indices.extend(range(offset, offset + width))
    offset += width
if len(structural_indices) != 6:
    raise SystemExit(f"could not locate StructuralTotalStrain metadata: {structural_indices}")
structural_total_strain = internal[structural_indices]
imposed = base_strain
in_plane_error = np.max(np.abs(structural_total_strain[[0, 1, 3]] - imposed[[0, 1, 3]]))
if float(in_plane_error) > 1.0e-12:
    raise SystemExit(f"rotated kinematics probe failed: error={in_plane_error}")
print(
    "rotated structural closure probe: passed "
    f"(max transverse stress={np.max(transverse):.3e}, "
    f"max in-plane strain error={in_plane_error:.3e})"
)

# Independent anisotropic-elastic Schur oracle in Kelvin storage.
young = np.array([210000.0, 190000.0, 180000.0])
compliance = np.array(
    [
        [1.0 / young[0], -0.25 / young[0], -0.23 / young[0]],
        [-0.25 / young[0], 1.0 / young[1], -0.27 / young[1]],
        [-0.23 / young[0], -0.27 / young[1], 1.0 / young[2]],
    ]
)
stiffness_normal = np.linalg.inv(compliance)
material_stiffness = np.zeros((6, 6))
material_stiffness[:3, :3] = stiffness_normal
material_stiffness[3, 3] = 2.0 * 80000.0
material_stiffness[4, 4] = 2.0 * 75000.0
material_stiffness[5, 5] = 2.0 * 70000.0
material_to_global_kelvin = kelvin_operator(material_to_global)
global_stiffness = material_to_global_kelvin @ material_stiffness @ material_to_global_kelvin.T
in_plane = np.array([0, 1, 3])
transverse_indices = np.array([2, 4, 5])
schur = (
    global_stiffness[np.ix_(in_plane, in_plane)]
    - global_stiffness[np.ix_(in_plane, transverse_indices)]
    @ np.linalg.solve(
        global_stiffness[np.ix_(transverse_indices, transverse_indices)],
        global_stiffness[np.ix_(transverse_indices, in_plane)],
    )
)
gps_in_plane = tangent[np.ix_(in_plane, in_plane)]
gps_schur_error = np.max(np.abs(gps_in_plane - schur)) / np.max(np.abs(schur))
if float(gps_schur_error) > 1.0e-10:
    raise SystemExit(f"GPS tangent vs Schur failed: relative error={gps_schur_error}")
inactive_columns = np.delete(np.arange(6), in_plane)
inactive_column_error = np.max(np.abs(tangent[:, inactive_columns]))
transverse_row_error = np.max(np.abs(tangent[np.ix_(transverse_indices, in_plane)]))

fd_errors = []
for step in (1.0e-5, 1.0e-6, 1.0e-7):
    fd = np.zeros((3, 3))
    for column, component in enumerate(in_plane):
        plus = base_strain.copy()
        minus = base_strain.copy()
        plus[component] += step
        minus[component] -= step
        plus_values, _, _ = integrate(plus)
        minus_values, _, _ = integrate(minus)
        plus_kelvin = material_to_global_kelvin @ plus_values
        minus_kelvin = material_to_global_kelvin @ minus_values
        fd[:, column] = (plus_kelvin[in_plane] - minus_kelvin[in_plane]) / (2.0 * step)
    error = np.max(np.abs(gps_in_plane - fd)) / np.max(np.abs(schur))
    fd_errors.append(float(error))
    if error > 1.0e-7:
        raise SystemExit(
            f"GPS tangent vs FD failed for h={step}: relative error={error}; "
            f"gps={gps_in_plane.tolist()}; fd={fd.tolist()}; schur={schur.tolist()}"
        )

print(
    "rotated tangent probe: passed "
    f"(GPS/Schur={gps_schur_error:.3e}, "
    f"GPS/FD={fd_errors}, "
    f"max inactive-column={inactive_column_error:.3e}, "
    f"max transverse-row={transverse_row_error:.3e})"
)
PY
