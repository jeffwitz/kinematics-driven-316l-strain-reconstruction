# First DIC-driven reconstruction

This tutorial walks through the complete pipeline on a **real 10 × 10-pixel
crop**. The goal is to obtain a result in seconds and understand how DIC data
becomes a mechanical problem.

The tutorial deliberately presents one path and few options. For a detailed
installation procedure, see {doc}`../how-to/install`.

## What you will produce

At the end, `results/tutorial-dic-10x10` will contain:

- a manifest describing inputs, solver settings, and 25 partitions;
- all six fields for every partition;
- stitched global fields;
- convergence diagnostics and SHA-256 fingerprints.

:::{admonition} Prerequisites
:class: important

- the repository was cloned with Git LFS;
- `.venv` contains the Python dependencies;
- TFEL/MFront and MGIS are installed;
- `build/mfront/src/libBehaviour.so` was compiled;
- PyPardiso/MKL is available.
:::

## 1. Activate the scientific environment

From the repository root:

```bash
source .venv/bin/activate
source /home/jeff/.local/share/tfel/env/env.sh
export PYTHONPATH=/home/jeff/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
```

The last two lines match the documented local installation. Change only the
prefix if TFEL/MGIS is installed elsewhere.

Check the sparse solver:

```bash
fem-inhouse backend
```

The output must identify **PyPardiso/MKL**. Production runs do not silently
fall back to SciPy.

## 2. Prepare a real DIC crop

The raw arrays use the historical names `U_40` and `V_40`, store displacement
in pixels, and do not yet contain the last nodal row and column. The following
command applies each transformation explicitly:

```bash
fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/tutorial-dic-10x10 \
  --crop-nx 10 \
  --crop-ny 10 \
  --hardening-scale-mpa 380 \
  --nonfinite-policy nearest \
  --nodal-completion edge-pad-upper
```

The crop is taken from the centre of the ROI. Four canonical arrays are
created:

| Array | Location | Shape |
|---|---|---:|
| `displacement_x_mm.npy` | nodes | `11 × 11` |
| `displacement_y_mm.npy` | nodes | `11 × 11` |
| `yield_stress_mpa.npy` | elements | `10 × 10` |
| `hardening_coefficient_mpa.npy` | elements | `10 × 10` |

`manifest.json` records the crop bounds, units, transformations, and hashes. It
is part of the scientific result.

## 3. Prepare the partitions

Split the small domain into 25 cores without padding:

```bash
fem-inhouse partition \
  --input data/processed/tutorial-dic-10x10 \
  --output results/tutorial-dic-10x10 \
  --count 25 \
  --padding 0 \
  --increments 10 \
  --list-pending
```

The first call writes the immutable calculation manifest. `--list-pending`
prints partition identifiers `0` through `24`.

The article ROI uses the same principle, but each core is surrounded by an
overlap. {doc}`../explanation/partitioning` explains why.

## 4. Solve with MFront

Run the 25 small calculations sequentially:

```bash
fem-inhouse partition \
  --input data/processed/tutorial-dic-10x10 \
  --output results/tutorial-dic-10x10 \
  --count 25 \
  --padding 0 \
  --increments 10 \
  --mfront-threads 2 \
  --solve-pending
```

For every partition, the solver:

1. imposes the current pseudo-time fraction of the DIC boundary displacement;
2. computes strain at the four CPS4 Gauss points;
3. asks MFront for stress, PEEQ, and the consistent tangent;
4. assembles the residual and tangent matrix;
5. solves the correction with PyPardiso;
6. commits the MFront state only after increment convergence.

A complete partition whose hashes still match is not recomputed.

## 5. Stitch the fields

```bash
for field in U S E PE PEEQ RF; do
  fem-inhouse partition \
    --input data/processed/tutorial-dic-10x10 \
    --output results/tutorial-dic-10x10 \
    --count 25 \
    --padding 0 \
    --increments 10 \
    --stitch "$field"
done
```

Stitching retains only each partition core. It does not average overlaps and
writes the global array through memory mapping.

## 6. Read the result

```python
from pathlib import Path

import numpy as np

result_directory = Path("results/tutorial-dic-10x10/global")
displacement = np.load(result_directory / "U.npy")
stress = np.load(result_directory / "S.npy")
peeq = np.load(result_directory / "PEEQ.npy")

print("U:", displacement.shape, displacement.min(), displacement.max())
print("S:", stress.shape, stress.min(), stress.max())
print("PEEQ:", peeq.shape, peeq.min(), peeq.max())
```

`U` is nodal; `S` and `PEEQ` are element fields. Component conventions are
listed in the {doc}`../reference/output_contract`.

## What this tutorial demonstrated

You did not prescribe the DIC strain throughout the domain. Only boundary
displacements were imposed. The interior field follows from equilibrium,
kinematic compatibility, and the local constitutive law. This distinction
separates numerical differentiation of DIC from a mechanically admissible
reconstruction.

## Next steps

- {doc}`../explanation/scientific_goal` for scientific interpretation;
- {doc}`../explanation/material_law` for the J2/Ludwik law;
- {doc}`../how-to/run_partitioned` for the complete ROI;
- {doc}`../how-to/inspect_results` to audit a campaign;
- {doc}`../reference/results` for measured performance and current evidence.
