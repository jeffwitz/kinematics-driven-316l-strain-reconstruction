#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_dir=$(cd -- "${script_dir}/.." && pwd)
tfel_env_file="${TFEL_ENV_FILE:-/home/jeff/.local/share/tfel/env/env.sh}"
build_dir="${MERIC_GENERIC_BUILD_DIR:-${repository_dir}/build/meric-generic}"
source_file="${build_dir}/Fcc316LMericCailletaudGeneric3D.mfront"

if [[ ! -f "${tfel_env_file}" ]]; then
  echo "TFEL environment file not found: ${tfel_env_file}" >&2
  exit 2
fi

set +u
source "${tfel_env_file}"
set -u

mkdir -p "${build_dir}"
python3 "${repository_dir}/scripts/generate_meric_generic_3d.py" \
  "${repository_dir}/mfront/Fcc316LMericCailletaud.mfront" \
  "${source_file}"
(cd "${build_dir}" && mfront --obuild --interface=generic "${source_file}")

library_path="${build_dir}/src/libBehaviour.so"
if [[ ! -f "${library_path}" ]]; then
  echo "MFront did not produce the expected library: ${library_path}" >&2
  exit 3
fi
printf '%s\n' "${library_path}"
