#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tfel_prefix=${TFEL_PREFIX:-/home/jeff/.local}
build_dir=${1:-"${repo_root}/build/structural-plane-stress-brick-probe"}

mkdir -p "${build_dir}"

g++ -std=c++20 -fPIC -shared \
  -I"${tfel_prefix}/include" \
  "${repo_root}/validation/mfront/structural_plane_stress_brick_probe.cxx" \
  -L"${tfel_prefix}/lib" -Wl,-rpath,"${tfel_prefix}/lib" \
  -lTFELMFront -lTFELMaterial -lTFELMath -lTFELSystem \
  -lTFELUtilities -lTFELGlossary -lTFELException \
  -o "${build_dir}/libStructuralPlaneStress3DProbe.so"

export LD_LIBRARY_PATH="${tfel_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MFRONT_ADDITIONAL_LIBRARIES="${build_dir}/libStructuralPlaneStress3DProbe.so"

echo "Registered bricks containing StructuralPlaneStress3DProbe:"
set +e
probe_output=$(mfront --list-behaviour-bricks 2>&1)
probe_status=$?
set -e

if grep -Fq "undefined symbol" <<<"${probe_output}"; then
  echo "external module loading reached the brick ABI and failed as expected:"
  grep -F "undefined symbol" <<<"${probe_output}"
  exit 0
fi

if grep -Fxq StructuralPlaneStress3DProbe <<<"${probe_output}"; then
  echo "external Behaviour Brick registration succeeded"
  exit 0
fi

printf '%s\n' "${probe_output}"
exit "${probe_status:-1}"
