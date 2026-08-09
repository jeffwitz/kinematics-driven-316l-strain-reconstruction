#!/usr/bin/env bash
# Compatibility entry point retained for historical probe commands.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/generate_structural_plane_stress.sh" "$@"
