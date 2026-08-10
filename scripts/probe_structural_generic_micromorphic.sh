#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=${1:-${TMPDIR:-/tmp}/structural_generic_micromorphic}
mkdir -p "$work"
set +u
source "${TFEL_ENV:-$HOME/.local/share/tfel/env/env.sh}"
set -u

"$root/.venv/bin/python" "$root/scripts/generate_structural_generic_micromorphic.py" \
    "$root/validation/mfront/MicromorphicJ2GenericBlocksProbe.mfront" \
    "$work/MicromorphicJ2GenericStructuralPlaneStressProbe.mfront"
cd "$work"
mfront --obuild --interface=generic \
    "$work/MicromorphicJ2GenericStructuralPlaneStressProbe.mfront"

LIBRARY="$work/src/libBehaviour.so" "$root/.venv/bin/python" - <<'PY'
import os

import mgis.behaviour as mgis
import numpy as np

behaviour = mgis.load(
    os.environ["LIBRARY"],
    "MicromorphicJ2GenericStructuralPlaneStressProbe",
    mgis.Hypothesis.Tridimensional,
)
data = mgis.MaterialDataManager(behaviour, 1)
for state in (data.s0, data.s1):
    for name, value in {
        "YoungModulus": 205000.0,
        "PoissonRatio": 0.3,
        "InitialYieldStress": 250.0,
        "HardeningCoefficient": 500.0,
        "HardeningExponent": 0.245,
        "MicromorphicCouplingModulus": 3000.0,
    }.items():
        mgis.setMaterialProperty(state, name, value)
    mgis.setExternalStateVariable(state, "Temperature", 293.15)
data.s1.gradients[0] = np.array([1.0e-2, -4.0e-3, 0.0, 2.0e-3, 0.0, 0.0, 1.0e-2])
if mgis.integrate(
    data, mgis.IntegrationType.IntegrationWithConsistentTangentOperator, 1.0, 0, 1
) != 1:
    raise SystemExit("structural generic integration failed")
stress = np.asarray(data.s1.thermodynamic_forces[0, :6])
if float(np.max(np.abs(stress[[2, 4, 5]]))) > 1.0e-8:
    raise SystemExit(f"structural closure failed: {stress[[2, 4, 5]]}")
if np.asarray(data.K).shape != (1, 49):
    raise SystemExit(f"unexpected tangent shape: {np.asarray(data.K).shape}")
print(
    "structural generic micromorphic probe: passed "
    f"(max transverse stress={np.max(np.abs(stress[[2, 4, 5]])):.3e})"
)
PY
