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
rotation = np.array(
    [
        [0.788071, -0.482929, 0.382683],
        [0.548985, 0.425669, -0.719846],
        [0.280166, 0.765908, 0.579228],
    ]
)
global_stress = rotation @ stress @ rotation.T
transverse = np.abs(
    [global_stress[2, 2], global_stress[1, 2], global_stress[0, 2]]
)
if float(np.max(transverse)) > 1.0e-7:
    raise SystemExit(f"rotated closure probe failed: transverse stress={transverse}")
print(f"rotated structural closure probe: passed (max transverse stress={np.max(transverse):.3e})")
PY
