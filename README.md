# Kinematics-Driven 316L Strain Reconstruction

Research software supporting the case study described in
`ArticleSource/ArticleAdil.pdf`: reconstruction of mechanically admissible
microscale strain-localization fields from DIC kinematics in 316L stainless
steel.

This project is deliberately **not** a general replacement for Abaqus. Its
supported scope is a regular CPS4 plane-stress mesh, J2 plasticity with
Ludwik-Hollomon hardening, DIC-prescribed boundary displacements, and
partitioned reconstruction of the article's pixel-resolved region of interest.

## Current status

The numerical prototype is being converted into tested, reproducible research
software. The live implementation plan and validation register are maintained
in [`Claude.md`](Claude.md). Scientific conventions are specified in
[`docs/scientific_contract.md`](docs/scientific_contract.md). The supported
numerical model and partition layout are documented in
[`docs/numerical_model.md`](docs/numerical_model.md) and
[`docs/partitioning.md`](docs/partitioning.md).

An article-sized corner partition (`510×460`, 234,600 elements) has now been
run directly from the versioned DIC inputs with the article's 100-partition,
150-pixel-padding layout. It converged in 20 increments without cutback; all
six raw fields, logs, hashes, resource measurements and derived comparison
maps are preserved under
[`validation/reference_data/article_100p_pad150_p0000`](validation/reference_data/article_100p_pad150_p0000).

An optional MFront 5.1.0/MGIS 3.1 constitutive backend is also implemented and
compiled for the same plane-stress J2/Ludwik material. Its first saved
material-point comparison passes the declared stress and PEEQ thresholds on
uniaxial, equibiaxial, and shear paths. Installation, tensor conventions, exact
metrics, and reproduction commands are documented in
[`docs/mfront.md`](docs/mfront.md).

Known limitations at this stage:

- the complete 11.16-million-element ROI has not yet been executed and
  stitched; the largest interior padded partitions are also larger than the
  completed corner partition;
- only DIC step 40 is available; the baseline steps 1–5 are not versioned;
- Abaqus parity is not yet established from the original `.inp` and ODB
  extraction scripts and is intentionally deferred until the DIC-first
  workflow is stable.
- MFront is validated at material points but is not yet connected to the
  finite-element Newton loop or used for the article-sized partition.

## Reproduce from the versioned DIC data

The four raw scientific arrays are versioned with Git LFS under
[`data/raw/case_study`](data/raw/case_study). From a fresh clone:

```bash
git lfs install
git lfs pull
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps

.venv/bin/fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study \
  --nonfinite-policy nearest
```

This command verifies every raw SHA-256, maps `V → u_x` and `U → u_y`,
converts pixels to millimetres, applies the article's nominal `K=380 MPa`,
repairs the nine declared non-finite hardening multipliers, completes the nodal
grid and writes a manifest for all generated arrays.

A fast smoke calculation uses a real central `10×10` crop:

```bash
bash examples/run_dic_smoke.sh
```

The script is equivalent to:

```bash
.venv/bin/fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study-10x10 \
  --crop-nx 10 \
  --crop-ny 10

.venv/bin/fem-inhouse partition \
  --input data/processed/case-study-10x10 \
  --output results/dic-smoke-10x10 \
  --count 25 \
  --padding 0 \
  --increments 10 \
  --solve-pending
```

Every partition preserves all final solver fields (`U`, `S`, `E`, `PE`,
`PEEQ`, `RF`) together with convergence diagnostics and output hashes. The full
production run uses the same prepared contract, with padding and partition
execution distributed according to the available memory. See
[`docs/from_dic_to_reconstruction.md`](docs/from_dic_to_reconstruction.md).

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy src/fem_inhouse
```

PyPardiso/MKL is a required runtime dependency for production solves.
MFront/MGIS is currently an optional source-built backend; see
[`docs/mfront.md`](docs/mfront.md) for the pinned installation and comparison.

The installed CLI provides the routine entry points:

```bash
fem-inhouse backend
fem-inhouse validate --nx 10 --ny 10
fem-inhouse example --nx 10 --ny 10 --output results/reduced
fem-inhouse prepare-case --help
fem-inhouse partition --help
fem-inhouse compare-fields --help
```

See [`docs/reduced_example.md`](docs/reduced_example.md) for the interpretation
of the example and its declared thresholds.

The historical Abaqus generators are kept byte-for-byte under
[`references/legacy_abaqus`](references/legacy_abaqus) solely for scientific
provenance. The production package neither imports nor executes them.
Initial PyPardiso timing and memory measurements are recorded in
[`docs/performance.md`](docs/performance.md).
The resumable CLI and Slurm-array workflow are documented in
[`docs/partitioning.md`](docs/partitioning.md).
Architecture decisions are recorded in [`docs/adr`](docs/adr), and numerical
review requirements are defined in [`CONTRIBUTING.md`](CONTRIBUTING.md).
The raw-to-canonical choices are recorded specifically in
[`ADR 0004`](docs/adr/0004-dic-input-preparation.md).

## Typed solver API

```python
import numpy as np

from fem_inhouse import CaseStudyConfig, MeshConfig, run_case_study

mesh = MeshConfig(nx=20, ny=20)
config = CaseStudyConfig(mesh)
shape_nodes = (mesh.nx + 1, mesh.ny + 1)
shape_elements = (mesh.nx, mesh.ny)

result = run_case_study(
    config,
    displacement_x_mm=np.zeros(shape_nodes),
    displacement_y_mm=np.zeros(shape_nodes),
    yield_stress_mpa=np.full(shape_elements, 250.0),
    hardening_coefficient_mpa=np.full(shape_elements, 500.0),
)
print(result.equivalent_plastic_strain.max())
print(result.diagnostics)
```

`result.diagnostics` trace le backend, la durée, les incréments convergés, les
cutbacks, les itérations de Newton et le critère de convergence final.

The top-level `fem_pixel.py` file remains only as a compatibility entry point
for existing case-study scripts.

## Citation and licensing

Citation metadata and the associated article authors are recorded in
[`CITATION.cff`](CITATION.cff). The repository does not yet declare a software
license: that legal choice remains an explicit project-owner decision and must
be resolved before a public release.
