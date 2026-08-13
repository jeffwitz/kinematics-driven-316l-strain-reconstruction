#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/srix-generic-plane-stress.XXXXXX")
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

behaviour = mgis.load(
    os.environ["LIBRARY"],
    "Fcc316LForestRubinSrixGeneric3D",
    mgis.Hypothesis.Tridimensional,
)
coupling = 100.0
in_plane = np.array([0.003, -0.0009, 0.0003])
chi = 2.0e-4
plane = np.array([0, 1, 3])
transverse = np.array([2, 4, 5])


def evaluate(total_strain, chi_value, tangent=True):
    data = mgis.MaterialDataManager(behaviour, 1)
    for state in (data.s0, data.s1):
        mgis.setMaterialProperty(state, "MicromorphicCouplingModulus", coupling)
        mgis.setExternalStateVariable(state, "Temperature", 293.15)
    data.s1.gradients[0] = np.concatenate((total_strain, [chi_value]))
    mode = (
        mgis.IntegrationType.IntegrationWithConsistentTangentOperator
        if tangent
        else mgis.IntegrationType.IntegrationWithoutTangentOperator
    )
    status = mgis.integrate(data, mode, 1.0, 0, 1)
    if status != 1:
        raise RuntimeError(f"3-D Generic integration failed: {status}")
    forces = np.asarray(data.s1.thermodynamic_forces)[0].copy()
    matrix = np.asarray(data.K)[0].copy() if tangent else None
    del data
    gc.collect()
    return forces, matrix


def blocks(flat):
    return (
        flat[:36].reshape(6, 6),
        flat[36:42].reshape(6, 1),
        flat[42:48].reshape(1, 6),
        flat[48:].reshape(1, 1),
    )


def closed_response(in_plane_value, chi_value, initial=None):
    total = np.zeros(6)
    total[plane] = in_plane_value
    total[transverse] = np.zeros(3) if initial is None else initial
    for iteration in range(15):
        forces, tangent = evaluate(total, chi_value)
        c_ee, _, _, _ = blocks(tangent)
        residual = forces[transverse]
        if np.max(np.abs(residual)) < 1.0e-8:
            break
        cbb = c_ee[np.ix_(transverse, transverse)]
        cbb_inv = np.linalg.inv(cbb)
        total[transverse] -= cbb_inv @ residual
    else:
        raise RuntimeError("3-D Generic plane-stress closure did not converge")
    forces, tangent = evaluate(total, chi_value)
    c_ee, s_chi, gamma_eps, gamma_chi = blocks(tangent)
    cbb = c_ee[np.ix_(transverse, transverse)]
    cbb_inv = np.linalg.inv(cbb)
    cab = c_ee[np.ix_(plane, transverse)]
    cba = c_ee[np.ix_(transverse, plane)]
    c_ps = c_ee[np.ix_(plane, plane)] - cab @ cbb_inv @ cba
    s_chi_ps = s_chi[plane] - cab @ cbb_inv @ s_chi[transverse]
    gamma_eps_ps = gamma_eps[:, plane] - gamma_eps[:, transverse] @ cbb_inv @ cba
    gamma_chi_ps = gamma_chi - gamma_eps[:, transverse] @ cbb_inv @ s_chi[transverse]
    return (
        forces[plane].copy(),
        forces[6].copy(),
        total[transverse].copy(),
        forces[transverse].copy(),
        np.block([[c_ps, s_chi_ps], [gamma_eps_ps, gamma_chi_ps]]),
    )


base_stress, base_gamma, base_transverse, base_residual, condensed = closed_response(in_plane, chi)
fd = np.zeros((4, 4))
for column in range(4):
    h = 1.0e-7
    plus_in_plane = in_plane.copy()
    minus_in_plane = in_plane.copy()
    plus_chi = chi
    minus_chi = chi
    if column < 3:
        plus_in_plane[column] += h
        minus_in_plane[column] -= h
    else:
        plus_chi += h
        minus_chi -= h
    plus = closed_response(plus_in_plane, plus_chi, base_transverse)
    minus = closed_response(minus_in_plane, minus_chi, base_transverse)
    fd[:, column] = np.concatenate((plus[0], [plus[1]]))
    fd[:, column] -= np.concatenate((minus[0], [minus[1]]))
    fd[:, column] /= 2.0 * h

error = np.linalg.norm(fd - condensed) / max(np.linalg.norm(fd), 1.0e-30)
print(f"plane-stress residual max={np.max(np.abs(base_residual)):.3e}")
print("condensed_block,finite_difference")
print(condensed)
print(fd)
if error > 1.0e-5:
    raise SystemExit(f"SRIX Generic plane-stress Schur check failed: {error:.3e}")
print(f"SRIX Generic plane-stress four-block Schur check: passed (relative error={error:.3e})")
PY
