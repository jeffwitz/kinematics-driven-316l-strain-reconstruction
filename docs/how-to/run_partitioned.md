# Run and resume a partitioned calculation

This guide executes the complete ROI as independent tasks. It assumes that
`data/processed/case-study` satisfies the canonical input contract.

## Prepare the campaign manifest

```bash
.venv/bin/fem-inhouse partition \
  --input data/processed/case-study \
  --output results/reconstruction-100 \
  --count 100 \
  --padding 150 \
  --increments 20 \
  --mfront-threads 8 \
  --list-pending
```

The first call writes `manifest.json` before any solve. A later command with
incompatible settings is rejected.

:::{warning}
Do not run the complete ROI as one monolithic problem. A padded interior
partition is larger than the validated `510 × 460` corner partition. Allocate
job time and memory accordingly.
:::

## Solve one local partition

```bash
.venv/bin/fem-inhouse --verbose partition \
  --input data/processed/case-study \
  --output results/reconstruction-100 \
  --count 100 \
  --padding 150 \
  --increments 20 \
  --mfront-threads 8 \
  --partition-id 0
```

Partition 0 solves `510 × 460` elements: a `360 × 310` core plus 150 overlap
elements towards the domain interior.

## Distribute with Slurm

```bash
export FEM_INPUT_DIR="$PWD/data/processed/case-study"
export FEM_OUTPUT_DIR="$PWD/results/reconstruction-100"
export FEM_PARTITION_COUNT=100
export FEM_PADDING=150
mkdir -p logs
sbatch --array=0-99 examples/slurm_partition_array.sh
```

Each task writes only under `partitions/NNNN`. Execution order does not affect
the result.

To match the MGIS pool to the Slurm allocation, add
`--mfront-threads "${SLURM_CPUS_PER_TASK:-1}"` to the template command and
request the same number of CPUs per task.

## Resume after interruption

```bash
.venv/bin/fem-inhouse partition \
  --input data/processed/case-study \
  --output results/reconstruction-100 \
  --count 100 \
  --padding 150 \
  --list-pending
```

A partition is complete only if:

- `status.json` contains `complete: true`;
- its manifest hash matches the campaign;
- all six historical and five complete-tensor files are present;
- every SHA-256 matches the status.

A missing or corrupted file automatically returns the partition to the pending
list. Do not delete valid calculations when resuming a campaign.

## Solve remaining tasks on one machine

For a small campaign or diagnostic:

```bash
.venv/bin/fem-inhouse partition \
  --input data/processed/case-study \
  --output results/reconstruction-100 \
  --count 100 \
  --padding 150 \
  --mfront-threads 8 \
  --solve-pending
```

Prefer a job array for the complete ROI because `--solve-pending` is
sequential.

## Stitch after validation

```bash
for field in U S E PE PEEQ RF; do
  .venv/bin/fem-inhouse partition \
    --input data/processed/case-study \
    --output results/reconstruction-100 \
    --count 100 \
    --padding 150 \
    --stitch "$field" \
    --field-output "results/reconstruction-100/global/${field}.npy"
done
```

Stitching opens local files through memory mapping and does not load the full
ROI into RAM. It retains each core; padding supports the solve and is not
averaged into the final field.

## Preserve resource measurements

For a representative partition:

```bash
/usr/bin/time -v -o resource-usage.txt \
  env MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  .venv/bin/fem-inhouse partition \
  --input data/processed/case-study \
  --output results/reconstruction-100 \
  --count 100 \
  --padding 150 \
  --mfront-threads 8 \
  --partition-id 55
```

Keep this file with the campaign. Internal solver time and complete process wall
time do not cover exactly the same phases.
