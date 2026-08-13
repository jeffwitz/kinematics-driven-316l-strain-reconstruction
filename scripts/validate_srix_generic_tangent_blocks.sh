#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/srix-generic-tangent.XXXXXX")
trap 'rm -rf "$work"' EXIT

set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u

"$root/.venv/bin/python" "$root/scripts/generate_srix_generic_3d.py" \
  "$root/mfront/Fcc316LForestRubinSrix.mfront" \
  "$work/Fcc316LForestRubinSrixGeneric3D.mfront"
(cd "$work" && mfront --obuild --interface=generic Fcc316LForestRubinSrixGeneric3D.mfront >/dev/null)

LIBRARY="$work/src/libBehaviour.so" "$root/.venv/bin/python" - <<'PY'
import gc
import os

import mgis.behaviour as mgis
import numpy as np

library = os.environ["LIBRARY"]
behaviour = mgis.load(
    library,
    "Fcc316LForestRubinSrixGeneric3D",
    mgis.Hypothesis.Tridimensional,
)
coupling = 100.0
steps = 6
target_strain = np.array([0.003, -0.0009, 0.0, 0.0, 0.0, 0.0])
target_chi = 2.0e-4
strain_step = 1.0e-7
chi_step = 1.0e-7


def configure(data):
    for state in (data.s0, data.s1):
        mgis.setMaterialProperty(state, "MicromorphicCouplingModulus", coupling)
        mgis.setExternalStateVariable(state, "Temperature", 293.15)


def integrate_to(final_strain, final_chi, with_tangent=True):
    data = mgis.MaterialDataManager(behaviour, 1)
    configure(data)
    for step in range(1, steps + 1):
        fraction = step / steps
        strain = target_strain * fraction
        chi = target_chi * fraction
        if step == steps:
            strain = final_strain
            chi = final_chi
        data.s1.gradients[0] = np.concatenate((strain, [chi]))
        mode = (
            mgis.IntegrationType.IntegrationWithConsistentTangentOperator
            if with_tangent
            else mgis.IntegrationType.IntegrationWithoutTangentOperator
        )
        status = mgis.integrate(data, mode, 1.0 / steps, 0, 1)
        if status != 1:
            raise RuntimeError(f"Generic integration failed at step {step}: {status}")
        if step != steps:
            mgis.update(data)
    return data


base = integrate_to(target_strain, target_chi)
flat_tangent = np.asarray(base.K)[0].copy()
if flat_tangent.shape != (49,):
    raise RuntimeError(f"unexpected tangent shape: {flat_tangent.shape}")

# MGIS stores the requested blocks consecutively for this 6+1 by 6+1 system.
analytical = np.zeros((7, 7))
analytical[:6, :6] = flat_tangent[:36].reshape(6, 6)
analytical[:6, 6:] = flat_tangent[36:42].reshape(6, 1)
analytical[6:, :6] = flat_tangent[42:48].reshape(1, 6)
analytical[6:, 6:] = flat_tangent[48:].reshape(1, 1)
del base
gc.collect()

fd = np.empty((7, 7))
for column in range(7):
    h = chi_step if column == 6 else strain_step
    plus_strain = target_strain.copy()
    minus_strain = target_strain.copy()
    plus_chi = target_chi
    minus_chi = target_chi
    if column == 6:
        plus_chi += h
        minus_chi -= h
    else:
        plus_strain[column] += h
        minus_strain[column] -= h
    plus = integrate_to(plus_strain, plus_chi, with_tangent=False)
    plus_forces = np.asarray(plus.s1.thermodynamic_forces)[0].copy()
    del plus
    gc.collect()
    minus = integrate_to(minus_strain, minus_chi, with_tangent=False)
    minus_forces = np.asarray(minus.s1.thermodynamic_forces)[0].copy()
    del minus
    gc.collect()
    fd[:, column] = (plus_forces - minus_forces) / (2.0 * h)

print("block,column,relative_error,cosine")
labels = [
    ("dsigma_depsilon", slice(0, 6), slice(0, 6)),
    ("dsigma_dchi", slice(0, 6), slice(6, 7)),
    ("dGamma_depsilon", slice(6, 7), slice(0, 6)),
    ("dGamma_dchi", slice(6, 7), slice(6, 7)),
]
for name, rows, columns in labels:
    expected = analytical[rows, columns]
    measured = fd[rows, columns]
    block_scale = max(np.linalg.norm(expected), np.linalg.norm(measured), 1.0e-30)
    for local_column in range(measured.shape[1]):
        a = expected[:, local_column]
        b = measured[:, local_column]
        error = np.linalg.norm(a - b) / block_scale
        cosine = float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1.0e-30)
        print(f"{name},{local_column},{error:.16e},{cosine:.16e}")

max_error = 0.0
for rows, columns in ((slice(0, 6), slice(0, 6)), (slice(0, 6), slice(6, 7)),
                      (slice(6, 7), slice(0, 6)), (slice(6, 7), slice(6, 7))):
    a = analytical[rows, columns]
    b = fd[rows, columns]
    max_error = max(max_error, np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))
if max_error > 1.0e-5:
    raise SystemExit(f"SRIX Generic tangent block check failed: max error={max_error:.3e}")
print(f"SRIX Generic 3-D tangent blocks: passed (max relative error={max_error:.3e})")
PY
