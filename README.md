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

Known limitations at this stage:

- the 11.16-million-element workflow is implemented but not yet benchmarked or
  executed at production scale;
- the historical plotting scripts still depend on external case data;
- Abaqus parity is not yet established from the original `.inp` and ODB
  extraction scripts.

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

The installed CLI provides the three routine entry points:

```bash
fem-inhouse backend
fem-inhouse validate --nx 10 --ny 10
fem-inhouse example --nx 10 --ny 10 --output results/reduced
fem-inhouse partition --help
```

See [`docs/reduced_example.md`](docs/reduced_example.md) for the interpretation
of the example and its declared thresholds.

The retained article-migration scripts no longer contain personal paths. Their
explicit `.npy` input and environment-variable contract is documented in
[`docs/legacy_data_contract.md`](docs/legacy_data_contract.md).
Initial PyPardiso timing and memory measurements are recorded in
[`docs/performance.md`](docs/performance.md).
The resumable CLI and Slurm-array workflow are documented in
[`docs/partitioning.md`](docs/partitioning.md).

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
