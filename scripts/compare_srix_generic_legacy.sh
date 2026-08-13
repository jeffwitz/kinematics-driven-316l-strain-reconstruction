#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/srix-generic-compare.XXXXXX")
trap 'rm -rf "$work"' EXIT
set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u

# HCHI activates the scalar micromorphic term. CHISCALE drives a non-zero
# field; for legacy equivalence chi is held fixed within each constitutive
# increment so both formulations evaluate the same local law.
"$root/.venv/bin/python" "$root/scripts/generate_srix_generic_3d.py" \
  "$root/mfront/Fcc316LForestRubinSrix.mfront" \
  "$work/Fcc316LForestRubinSrixGeneric3D.mfront"
(cd "$work" && mfront --obuild --interface=generic Fcc316LForestRubinSrixGeneric3D.mfront >/dev/null)

LEGACY="$root/build/mfront/src/libBehaviour.so" \
GENERIC="$work/src/libBehaviour.so" \
HCHI="${HCHI:-0}" \
CHISCALE="${CHISCALE:-2e-4}" \
"$root/.venv/bin/python" - <<'PY'
import os

import mgis.behaviour as mgis
import numpy as np

legacy = mgis.load(os.environ["LEGACY"], "Fcc316LForestRubinSrix", mgis.Hypothesis.Tridimensional)
generic = mgis.load(
    os.environ["GENERIC"], "Fcc316LForestRubinSrixGeneric3D", mgis.Hypothesis.Tridimensional
)
coupling = float(os.environ["HCHI"])
chi_scale = float(os.environ["CHISCALE"])
ld = mgis.MaterialDataManager(legacy, 1)
gd = mgis.MaterialDataManager(generic, 1)
for data, behaviour in ((ld, legacy), (gd, generic)):
    for state in (data.s0, data.s1):
        mgis.setMaterialProperty(state, "MicromorphicCouplingModulus", coupling)
        for variable in behaviour.external_state_variables:
            mgis.setExternalStateVariable(
                state, variable.name, 293.15 if variable.name == "Temperature" else 0.0
            )

print("step,legacy_status,generic_status,relative_stress_error,max_stress_error_mpa")
for step in range(1, 7):
    strain = np.array([0.003 * step / 8, -0.0009 * step / 8, 0, 0, 0, 0.0])
    chi = chi_scale * step / 8
    mgis.setExternalStateVariable(ld.s0, "NonlocalEquivalentPlasticStrain", chi)
    mgis.setExternalStateVariable(ld.s1, "NonlocalEquivalentPlasticStrain", chi)
    ld.s1.gradients[0] = strain
    gd.s0.gradients[0, 6] = chi
    gd.s1.gradients[0] = np.concatenate((strain, [chi]))
    ls = mgis.integrate(
        ld, mgis.IntegrationType.IntegrationWithConsistentTangentOperator, 1 / 8, 0, 1
    )
    gs = mgis.integrate(
        gd, mgis.IntegrationType.IntegrationWithConsistentTangentOperator, 1 / 8, 0, 1
    )
    legacy_stress = np.asarray(ld.s1.thermodynamic_forces)[0]
    generic_stress = np.asarray(gd.s1.thermodynamic_forces)[0, :6]
    error = np.linalg.norm(generic_stress - legacy_stress) / max(
        np.linalg.norm(legacy_stress), 1e-30
    )
    maximum = np.max(np.abs(generic_stress - legacy_stress))
    print(f"{step},{ls},{gs},{error:.16e},{maximum:.16e}")
    if ls != 1 or gs != 1:
        break
    mgis.update(ld)
    mgis.update(gd)
PY
