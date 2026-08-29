# Install the project

**Mode:** how-to  
**Domain:** software

## Goal

Create a usable Python environment from a clean POSIX/Linux checkout. The base
installation includes the runtime packages used by the numerical code,
including NumPy, SciPy and `threadpoolctl`. Numba is an optional accelerator;
TFEL/MFront is a separate native dependency.

## Prerequisites

Use Python 3.11 or newer and run the commands from the repository root. A
working compiler is needed only for packages that are not already available as
wheels. Do not install TFEL as a Python extra: its environment is configured
separately in {doc}`build_mfront`.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the native NumPy/Numba path, add the optional accelerator explicitly:

```bash
python -m pip install -e '.[performance]'
```

Install development and documentation tools only when needed:

```bash
python -m pip install -e '.[dev,docs]'
```

## Verify

Check the runtime imports before preparing a case:

```bash
python - <<'PY'
import fem_inhouse
import numpy
import scipy
import threadpoolctl
print("python environment: OK")
PY
```

Then run the smallest dependency-free unit slice:

```bash
python -m pytest -q tests/unit/test_results.py
```

The full unit suite is `python -m pytest tests/unit -q`; it may include tests
that require optional native libraries. Keep compiler/TFEL paths in the case
manifest; platform-specific constitutive setup is covered by
{doc}`build_mfront`.

## What this establishes

The package and its Python runtime dependencies are installed, and the basic
result contracts can be imported and tested.

## What this does **not** establish

It does not build or validate an MFront behaviour, nor does it qualify a
constitutive backend. Those require the separate TFEL/MFront step.
