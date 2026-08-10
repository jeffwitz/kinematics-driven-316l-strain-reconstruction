#!/usr/bin/env bash
# Build and run the micromorphic generic-tangent-block feasibility probe.
#
# The probe is deliberately kept out of the production behaviour library: it is
# tridimensional, hand-written, and answers one question -- whether MFront can
# return the four coupled tangent blocks the micromorphic Newton needs, instead
# of the host computing three of them by finite differences.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${1:-${TMPDIR:-/tmp}/micromorphic_generic_blocks}"
mkdir -p "$work"

# shellcheck disable=SC1090
source "${TFEL_ENV:-$HOME/.local/share/tfel/env/env.sh}"

cd "$work"
mfront --obuild --interface=generic \
    "$root/validation/mfront/MicromorphicJ2GenericBlocksProbe.mfront"

cd "$root"
PYTHONPATH="${PYTHONPATH:-}:$HOME/.local/lib/python3.12/site-packages" \
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.local/lib" \
    .venv/bin/python scripts/probe_micromorphic_generic_blocks.py \
    --library "$work/src/libBehaviour.so" \
    --fd-step 1e-6 --fd-step 1e-7 --fd-step 1e-8 \
    --output validation/_generated/performance/micromorphic_generic_tangent_blocks.json
