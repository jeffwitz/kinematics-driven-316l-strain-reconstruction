#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
probe_source="$repo_root/validation/mfront/StructuralPlaneStressIntegratorHookProbe.mfront"
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/structural-plane-stress-hook.XXXXXX")

if [[ ! -f "$probe_source" ]]; then
  echo "missing probe source: $probe_source" >&2
  exit 2
fi

set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u
cd "$probe_dir"
mfront --obuild --interface=generic "$probe_source"

header="include/TFEL/Material/StructuralPlaneStressIntegratorHookProbe.hxx"
if [[ ! -f "$header" ]]; then
  echo "generated header not found: $header" >&2
  exit 2
fi

grep -Fq 'this->fzeros(i) = this->sig(i) / Gref;' "$header"
grep -Fq 'this->jacobian(i, j) = this->D_tdt(i, j) / Gref;' "$header"

compute_line=$(grep -n 'bool computeFdF' "$header" | head -1 | cut -d: -f1)
probe_line=$(grep -n 'this->fzeros(i) = this->sig(i) / Gref' "$header" | head -1 | cut -d: -f1)
if (( probe_line <= compute_line )); then
  echo "probe code was not emitted inside computeFdF after its entry" >&2
  exit 1
fi

echo "StructuralPlaneStress3D local hook probe: passed"
echo "generated_header=$probe_dir/$header"
echo "computeFdF_line=$compute_line"
echo "row_replacement_line=$probe_line"

python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "runtime probe requires an executable Python with MGIS: $python_bin" >&2
  exit 2
fi

LIBRARY="$probe_dir/src/libBehaviour.so" "$python_bin" - <<'PY'
import os

import mgis.behaviour as mgis
import numpy as np

library = os.environ["LIBRARY"]
behaviour = mgis.load(
    library,
    "StructuralPlaneStressIntegratorHookProbe",
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
stress = np.asarray(data.s1.thermodynamic_forces[0])
transverse = np.abs(stress[[2, 4, 5]])
if float(np.max(transverse)) > 1.0e-8:
    raise SystemExit(f"plane-stress probe failed: transverse stress={transverse}")
print(f"runtime closure probe: passed (max transverse stress={np.max(transverse):.3e})")
PY
