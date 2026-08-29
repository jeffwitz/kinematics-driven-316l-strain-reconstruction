# Build the MFront reference behaviour

**Mode:** how-to  
**Domain:** crystal-plasticity

## Goal

Build the generic-interface MFront library used as the constitutive reference
by the registered SRIX and Méric workflows.

## Prerequisites

Install TFEL, MFront and MGIS using the method appropriate for the host. The
repository script accepts the location of the installed TFEL environment via
`TFEL_ENV_FILE`; no user-specific path is assumed.

## Run

```bash
TFEL_ENV_FILE=/path/to/tfel/env.sh \
./scripts/build_mfront_behaviour.sh
export MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so"
```

`TFEL_ENV_FILE` must name an environment file that exports the `mfront` and
MGIS runtime tools. The script generates the structural SRIX/Méric wrappers,
builds the generic behaviours and stops on any failed compilation.

## Expected outputs

The expected library is:

```text
build/mfront/src/libBehaviour.so
```

The source set includes `Fcc316LForestRubinSrix`,
`Fcc316LMericCailletaud` and their registered structural plane-stress wrappers.
The SRIX behaviour identifier used by the qualification scripts is
`fcc_forest_rubin_srix`.

## Verify

```bash
test -s build/mfront/src/libBehaviour.so && \
  echo "MFront library: OK"
```

Record the library path, source commit and behaviour identifier in the run
manifest. Select the backend using
{doc}`../../reference/software/configuration` and run the constitutive case
with {doc}`../crystal-plasticity/run_316l_crystal_plasticity`.

## What this establishes

The generic MFront constitutive oracle required by the registered MFront and
spectral qualification drivers is available.

## What this does **not** establish

Compilation alone does not qualify SRIX, structural plane stress or the native
NumPy backend. Those are checked by their respective qualification workflows.
