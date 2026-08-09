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
