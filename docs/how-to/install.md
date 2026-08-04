# Install Python, TFEL/MFront, and MGIS

This guide reproduces the environment validated on Ubuntu. It installs
TFEL/MFront 5.1.0 and MGIS 3.1 under `${HOME}/.local`, then installs the Python
package in `.venv`.

## Install system prerequisites

```bash
sudo apt update
sudo apt install \
  build-essential \
  cmake \
  gfortran \
  git \
  git-lfs \
  python3-dev \
  python3-numpy \
  python3-venv \
  pybind11-dev
```

To build the PDF documentation as well:

```bash
sudo apt install \
  fonts-freefont-otf \
  graphviz \
  inkscape \
  latexmk \
  texlive-fonts-recommended \
  texlive-latex-extra \
  texlive-latex-recommended \
  texlive-luatex
```

## Build TFEL/MFront 5.1.0

```bash
mkdir -p "${HOME}/.local/src"
git clone https://github.com/thelfer/tfel.git \
  "${HOME}/.local/src/tfel-5.1.0"
git -C "${HOME}/.local/src/tfel-5.1.0" \
  checkout deee4cd77e1f36efd3715f7aad9a673de9c92880

cmake \
  -S "${HOME}/.local/src/tfel-5.1.0" \
  -B "${HOME}/.local/src/tfel-5.1.0-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${HOME}/.local" \
  -DTFEL_APPEND_VERSION=OFF \
  -Denable-numpy-support=ON \
  -Denable-python=ON \
  -Denable-python-bindings=ON
cmake --build "${HOME}/.local/src/tfel-5.1.0-build" --parallel
cmake --install "${HOME}/.local/src/tfel-5.1.0-build"
```

`TFEL_APPEND_VERSION=OFF` is the only deviation from the upstream quick
procedure. On the reference host, the enabled suffix produced an invalid
Python module name with TFEL 5.1.0. No TFEL source file is patched.

## Build MGIS 3.1

```bash
git clone https://github.com/thelfer/MFrontGenericInterfaceSupport.git \
  "${HOME}/.local/src/mgis-3.1"
git -C "${HOME}/.local/src/mgis-3.1" \
  checkout 38dd3082f745f736abbb6629d82e829b91132514

cmake \
  -S "${HOME}/.local/src/mgis-3.1" \
  -B "${HOME}/.local/src/mgis-3.1-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${HOME}/.local" \
  -DTFEL_DIR="${HOME}/.local/share/tfel/cmake" \
  -Denable-c-bindings=ON \
  -Denable-python-bindings=ON
cmake --build "${HOME}/.local/src/mgis-3.1-build" --parallel
cmake --install "${HOME}/.local/src/mgis-3.1-build"
```

## Install the Python project

```bash
git lfs install
git lfs pull
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
```

## Activate and verify

```bash
source "${HOME}/.local/share/tfel/env/env.sh"
export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

mfront --version
tfel-config --version
.venv/bin/python -c \
  "import tfel, mgis.behaviour; print(tfel.getTFELVersion())"
.venv/bin/fem-inhouse backend
```

The expected TFEL version is `5.1.0`. The last command must identify
PyPardiso/MKL.

## Compile the repository behaviour

```bash
bash scripts/build_mfront_behaviour.sh
test -f build/mfront/src/libBehaviour.so
```

For TFEL installed under another prefix:

```bash
TFEL_ENV_FILE=/opt/tfel/share/tfel/env/env.sh \
  bash scripts/build_mfront_behaviour.sh
```

## Run the checks

```bash
source "${HOME}/.local/share/tfel/env/env.sh"
export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy src/fem_inhouse
```

Tests marked `mfront` use the compiled library when it is available.

## Install and build the documentation

```bash
.venv/bin/python -m pip install -r requirements-docs.txt
PYTHONPATH=src .venv/bin/python scripts/build_documentation_figures.py
PYTHONPATH=src .venv/bin/python scripts/measure_spectral_dtt_contract.py
PYTHONPATH=src .venv/bin/python scripts/build_spectral_documentation_figures.py
PATH="$PWD/.venv/bin:$PATH" make -C docs html
PATH="$PWD/.venv/bin:$PATH" make -C docs latexpdf
```

Outputs are written under `docs/_build/html` and `docs/_build/latex`.

## Diagnose common failures

`libBehaviour.so` not found
: Run `scripts/build_mfront_behaviour.sh` from the repository root or pass
  `--mfront-library` to the solver.

`ModuleNotFoundError: mgis`
: Check the `PYTHONPATH` for the MGIS prefix and the Python version used by
  `.venv`.

PyPardiso missing
: Reinstall `requirements-lock.txt`. Production mode does not silently switch
  to SuperLU.

Undefined TFEL shared-library symbol
: Source `share/tfel/env/env.sh` in the shell that launches Python, not only in
  the shell that compiled the behaviour.
