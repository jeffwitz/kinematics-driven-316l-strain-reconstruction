#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cli="${repository_root}/.venv/bin/fem-inhouse"
raw_directory="${repository_root}/data/raw/case_study"
input_directory="${repository_root}/data/processed/case-study-10x10"
output_directory="${repository_root}/results/dic-smoke-10x10"

if [[ ! -x "${cli}" ]]; then
  echo "Missing ${cli}; install the locked environment first." >&2
  exit 2
fi

"${cli}" prepare-case \
  --raw "${raw_directory}" \
  --output "${input_directory}" \
  --crop-nx 10 \
  --crop-ny 10

"${cli}" partition \
  --input "${input_directory}" \
  --output "${output_directory}" \
  --count 25 \
  --padding 0 \
  --increments 10 \
  --solve-pending

for field in U S E PEEQ; do
  "${cli}" partition \
    --input "${input_directory}" \
    --output "${output_directory}" \
    --count 25 \
    --padding 0 \
    --increments 10 \
    --stitch "${field}"
done

echo "DIC smoke reconstruction complete: ${output_directory}/global"
