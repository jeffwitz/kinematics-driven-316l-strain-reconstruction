#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
probe_source="$repo_root/validation/mfront/ImplicitGenericTwoFieldProbe.mfront"
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/implicit-generic-two-field.XXXXXX")

set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u
cd "$probe_dir"
mfront --obuild --interface=generic "$probe_source"

library="$probe_dir/src/libBehaviour.so"
PYTHON_BIN="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
LIBRARY="$library" "$PYTHON_BIN" - <<'PY'
import os

import mgis.behaviour as mgis
import numpy as np

library = os.environ["LIBRARY"]
behaviour = mgis.load(
    library,
    "ImplicitGenericTwoFieldProbe",
    mgis.Hypothesis.Tridimensional,
)
blocks = list(behaviour.tangent_operator_blocks)
expected = ["dsig_ddeto", "dsig_ddphi", "dpre_ddeto", "dpre_ddphi"]
block_names = [(left.name, right.name) for left, right in blocks]
if block_names != [("Stress", "Strain"), ("Stress", "CoupledStrain"),
                   ("CoupledStress", "Strain"), ("CoupledStress", "CoupledStrain")]:
    raise SystemExit(f"unexpected tangent blocks: {block_names!r}")

data = mgis.MaterialDataManager(behaviour, 1)
for state in (data.s0, data.s1):
    mgis.setMaterialProperty(state, "lambda", 100.0)
    mgis.setMaterialProperty(state, "mu", 50.0)
    mgis.setMaterialProperty(state, "coupling", 7.0)
    mgis.setExternalStateVariable(state, "Temperature", 293.15)
data.s1.gradients[0] = np.array([1.0e-3, -2.0e-4, 3.0e-4, 4.0e-4, -5.0e-4, 6.0e-4, 2.0e-3])
mgis.integrate(
    data,
    mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
    1.0,
    0,
    1,
)

values = np.asarray(data.K)
if values.size != 7 * 7:
    raise SystemExit(f"unexpected tangent storage size: {values.size}")
matrix = values.reshape(7, 7)
block_sizes = [matrix[:6, :6].size, matrix[:6, 6:].size,
               matrix[6:, :6].size, matrix[6:, 6:].size]
if block_sizes != [36, 6, 6, 1]:
    raise SystemExit(f"unexpected block sizes: {block_sizes}")
print(f"ImplicitGenericBehaviour probe: passed (blocks={block_names!r})")
print(f"block_sizes={block_sizes}")
print(f"K_shape={values.shape}")
PY
