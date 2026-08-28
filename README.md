# Kinematics-Driven 316L Strain Reconstruction

Research software combining DIC kinematics and EBSD crystal orientations in a
mechanically admissible 316L workflow. Three-dimensional constitutive laws are
evaluated under structural plane stress by a spectral/matrix-free FEM solver;
MFront and native SRIX share the material contract.

The code imposes measured displacement on the boundary of a structured
plane-stress domain. Interior displacement, strain, stress and crystal
plasticity then follow from compatibility and mechanical equilibrium. J2/Ludwik
is retained as a baseline; SRIX/MFront/native are the crystal-plasticity paths.
This is deliberately not a general Abaqus replacement.

## Current scientific conclusion

The repository records qualification evidence for DIC/EBSD preparation,
structural plane stress, SRIX against MFront, spectral mechanics and the native
NumPy/Numba implementation. FEMU/SVD and REGM remain explicit identification
and screening workflows with documented observability limits; historical J2 and
micromorphic results are retained as baselines and scientific branches.

## Install

TFEL/MFront, MGIS and PyPardiso/MKL are required for the nominal workflow.
After installing TFEL/MGIS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,docs]'
source /home/jeff/.local/share/tfel/env/env.sh
cmake -S . -B build/mfront
cmake --build build/mfront
fem-inhouse backend
```

See the [installation guide](docs/how-to/installation/install.md) for supported
versions and platform details.

## First reconstruction

```bash
fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/tutorial-10x10 \
  --crop-nx 10 --crop-ny 10 \
  --hardening-scale-mpa 380 \
  --nonfinite-policy nearest \
  --nodal-completion edge-pad-upper

fem-inhouse partition \
  --input data/processed/tutorial-10x10 \
  --output results/tutorial-local \
  --parts-x 1 --parts-y 1 \
  --padding 0 --increments 10 \
  --constitutive-backend mfront-native-plane-stress \
  --partition-id 0
```

The [tutorial](docs/tutorials/first_reconstruction.md) explains what is
measured, imposed and reconstructed. See the [How-to guides](docs/how-to/index.md)
for backend and qualification workflows.

## Documentation and quality

The English documentation follows Diátaxis:
[Tutorials](docs/tutorials/first_reconstruction.md),
[How-to](docs/how-to/index.md), [Reference](docs/reference/index.md), and
[Explanation](docs/explanation/index.md). Run the checks below with the venv
active; `latexpdf` needs `lualatex` and `latexmk` (Debian/Ubuntu:
`texlive-luatex`, `texlive-latex-extra`, `latexmk`).

```bash
ruff check .
mypy src tests
pytest
make -C docs html latexpdf
```

Evidence and claim boundaries come from
`validation/documentation_evidence_registry.json`.

## Citation and licence
Citation metadata are in [`CITATION.cff`](CITATION.cff). No software licence
has been declared; reuse outside the private project requires permission.
