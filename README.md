# Kinematics-Driven 316L Strain Reconstruction

Research software for reconstructing mechanically admissible microscale fields
from digital image correlation (DIC) kinematics in the 316L case study supplied
with the repository.

The code imposes measured displacement on the boundary of a structured
plane-stress finite-element domain. Interior displacement, strain, stress and
plasticity then follow from compatibility, a J2/Ludwik constitutive model and
mechanical equilibrium. This is deliberately not a general Abaqus replacement.

## Current scientific conclusion

The local baseline is numerically verified across its independent tabulated and
analytical MFront implementations, but it produces localization that is too
narrow and concentrated. A coupled micromorphic model introduces a coupling
modulus and a spatial length and demonstrably redistributes plasticity.
Reduced-fidelity evidence distinguishes a length effect from a
coupling-strength effect. No unique transferable material internal length is
currently claimed.

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

See the [installation guide](docs/how-to/install.md) for supported versions and
platform details.

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
measured, imposed and reconstructed.

## Documentation and quality

The English documentation follows Diátaxis:
[Tutorials](docs/tutorials/first_reconstruction.md),
[How-to](docs/how-to/index.md), [Reference](docs/reference/index.md), and
[Explanation](docs/explanation/index.md).

```bash
ruff check .
mypy src tests
pytest
make -C docs html latexpdf
```

The evidence and claim boundaries are generated from
`validation/documentation_evidence_registry.json`; detailed machine-readable
reports remain under `validation/`.

## Citation and licence

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). No software
licence has yet been declared; reuse outside the private project therefore
requires permission from the authors.
