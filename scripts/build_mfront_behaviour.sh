#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd -- "${script_dir}/.." && pwd)"
tfel_env_file="${TFEL_ENV_FILE:-/home/jeff/.local/share/tfel/env/env.sh}"
build_dir="${MFRONT_BUILD_DIR:-${repository_dir}/build/mfront}"
behaviour_files=(
  "${repository_dir}/mfront/PixelLudwikJ2Plasticity.mfront"
  "${repository_dir}/mfront/PixelLudwikJ2Plasticity3D.mfront"
  "${repository_dir}/mfront/PixelMicromorphicLudwikJ2Plasticity.mfront"
  "${repository_dir}/mfront/PixelMicromorphicLudwikJ2Plasticity3D.mfront"
  "${repository_dir}/mfront/Fcc316LMericCailletaud.mfront"
  "${repository_dir}/mfront/Fcc316LForestRubinSrix.mfront"
)

if [[ ! -f "${tfel_env_file}" ]]; then
  echo "TFEL environment file not found: ${tfel_env_file}" >&2
  echo "Set TFEL_ENV_FILE to the installed TFEL env.sh." >&2
  exit 2
fi

# shellcheck disable=SC1090
set +u
source "${tfel_env_file}"
set -u

mkdir -p "${build_dir}"
cd "${build_dir}"
mfront --obuild --interface=generic "${behaviour_files[@]}"

library_path="${build_dir}/src/libBehaviour.so"
if [[ ! -f "${library_path}" ]]; then
  echo "MFront did not produce the expected library: ${library_path}" >&2
  exit 3
fi

printf '%s\n' "${library_path}"
