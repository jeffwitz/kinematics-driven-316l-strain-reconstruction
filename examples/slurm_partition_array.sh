#!/usr/bin/env bash
# Submit with:
#   sbatch --array=0-24 examples/slurm_partition_array.sh
# or, for the 100-partition layout:
#   FEM_PARTITION_COUNT=100 sbatch --array=0-99 examples/slurm_partition_array.sh

#SBATCH --job-name=fem-316l
#SBATCH --output=logs/fem-316l-%A_%a.out
#SBATCH --error=logs/fem-316l-%A_%a.err

set -euo pipefail

: "${FEM_INPUT_DIR:?Set FEM_INPUT_DIR to the four case-study .npy fields}"
: "${FEM_OUTPUT_DIR:?Set FEM_OUTPUT_DIR to the reconstruction directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

FEM_PARTITION_COUNT="${FEM_PARTITION_COUNT:-25}"
FEM_PADDING="${FEM_PADDING:-150}"

fem-inhouse partition \
  --input "${FEM_INPUT_DIR}" \
  --output "${FEM_OUTPUT_DIR}" \
  --count "${FEM_PARTITION_COUNT}" \
  --padding "${FEM_PADDING}" \
  --partition-id "${SLURM_ARRAY_TASK_ID}"
