#!/usr/bin/env bash
# Compile the same micromorphic generic-block probe under PlaneStress.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=${1:-${TMPDIR:-/tmp}/micromorphic_generic_plane_stress}
mkdir -p "$work"
set +u
source "${TFEL_ENV:-$HOME/.local/share/tfel/env/env.sh}"
set -u

sed 's/@ModellingHypothesis Tridimensional;/@ModellingHypothesis PlaneStress;/' \
    "$root/validation/mfront/MicromorphicJ2GenericBlocksProbe.mfront" \
    > "$work/MicromorphicJ2GenericBlocksPlaneStressProbe.mfront"
cd "$work"
mfront --obuild --interface=generic \
    "$work/MicromorphicJ2GenericBlocksPlaneStressProbe.mfront"

LIBRARY="$work/src/libBehaviour.so" "$root/.venv/bin/python" - <<'PY'
import os

import mgis.behaviour as mgis
import numpy as np

behaviour = mgis.load(
    os.environ["LIBRARY"],
    "MicromorphicJ2GenericBlocksProbe",
    mgis.Hypothesis.PlaneStress,
)
names = [(left.name, right.name) for left, right in behaviour.tangent_operator_blocks]
expected = [
    ("Stress", "Strain"),
    ("Stress", "NonlocalEquivalentPlasticStrain"),
    ("EquivalentPlasticStrainOutput", "Strain"),
    ("EquivalentPlasticStrainOutput", "NonlocalEquivalentPlasticStrain"),
]
if names != expected:
    raise SystemExit(f"unexpected tangent blocks: {names!r}")
data = mgis.MaterialDataManager(behaviour, 1)
for state in (data.s0, data.s1):
    mgis.setMaterialProperty(state, "YoungModulus", 205000.0)
    mgis.setMaterialProperty(state, "PoissonRatio", 0.3)
    mgis.setMaterialProperty(state, "InitialYieldStress", 250.0)
    mgis.setMaterialProperty(state, "HardeningCoefficient", 500.0)
    mgis.setMaterialProperty(state, "HardeningExponent", 0.245)
    mgis.setMaterialProperty(state, "MicromorphicCouplingModulus", 3000.0)
    mgis.setExternalStateVariable(state, "Temperature", 293.15)
data.s1.gradients[0] = np.array([1.0e-3, -4.0e-4, 2.0e-4, 0.0, 1.0e-3])
if mgis.integrate(
    data, mgis.IntegrationType.IntegrationWithConsistentTangentOperator, 1.0, 0, 1
) != 1:
    raise SystemExit("plane-stress generic micromorphic integration failed")
if np.asarray(data.K).shape != (1, 25):
    raise SystemExit(f"unexpected plane-stress tangent shape: {np.asarray(data.K).shape}")
print(f"plane-stress micromorphic generic probe: passed (blocks={names!r})")
PY
