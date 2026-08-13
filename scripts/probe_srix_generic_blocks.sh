#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/srix-generic-blocks.XXXXXX")
trap 'rm -rf "$work"' EXIT

set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u

"$root/.venv/bin/python" "$root/scripts/generate_srix_generic_3d.py" \
  "$root/mfront/Fcc316LForestRubinSrix.mfront" \
  "$work/Fcc316LForestRubinSrixGeneric3D.mfront"
(cd "$work" && mfront --obuild --interface=generic Fcc316LForestRubinSrixGeneric3D.mfront)

LIBRARY="$work/src/libBehaviour.so" "$root/.venv/bin/python" - <<'PY'
import os

import mgis.behaviour as mgis
import numpy as np

library = os.environ["LIBRARY"]
behaviour = mgis.load(
    library,
    "Fcc316LForestRubinSrixGeneric3D",
    mgis.Hypothesis.Tridimensional,
)
blocks = [(left.name, right.name) for left, right in behaviour.tangent_operator_blocks]
expected = [
    ("Stress", "Strain"),
    ("Stress", "NonlocalEquivalentPlasticStrain"),
    ("AccumulatedSlipOutput", "Strain"),
    ("AccumulatedSlipOutput", "NonlocalEquivalentPlasticStrain"),
]
if blocks != expected:
    raise SystemExit(f"unexpected SRIX tangent blocks: {blocks!r}")

data = mgis.MaterialDataManager(behaviour, 1)
for state in (data.s0, data.s1):
    mgis.setMaterialProperty(state, "MicromorphicCouplingModulus", 100.0)
    mgis.setExternalStateVariable(state, "Temperature", 293.15)
data.s1.gradients[0] = np.array([1.0e-5, -3.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0])
status = mgis.integrate(
    data,
    mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
    1.0,
    0,
    1,
)
if status != 1:
    raise SystemExit(f"SRIX generic elastic integration failed: status={status}")
if np.asarray(data.K).shape != (1, 49):
    raise SystemExit(f"unexpected SRIX generic tangent storage: {np.asarray(data.K).shape}")
print(f"SRIX generic 3-D tangent-block probe: passed (blocks={blocks!r})")
print("The probe validates the generic block interface and elastic integration;")
print("plastic equivalence against the historical SRIX behaviour remains required.")
PY
