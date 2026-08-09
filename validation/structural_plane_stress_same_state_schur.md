# Generic structural plane-stress tangent: same-state Schur qualification

This qualification compares the generic `StructuralPlaneStress3D` shell with
the independent three-dimensional law at the same committed state. For each
increment, the complete generic snapshot is transplanted into the raw bridge
by declared internal-variable name and by explicit global/material-frame
conversion. Both routes then receive the same in-plane target and time
increment. The raw route solves its three transverse closure equations and
forms

\[
C_{\mathrm{Schur}}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.
\]

The live report is:

```text
validation/_generated/performance/structural_plane_stress_same_state_schur.json
```

The test uses a general Bunge orientation `(35°, 20°, 15°)` and six successive
states, from the elastic regime into the tested plastic regime. The maximum
relative discrepancies are:

| law | stress, generic vs raw | tangent, generic vs raw Schur |
| --- | ---: | ---: |
| SRIX | `7.21e-13` | `6.23e-13` |
| Méric--Cailletaud | `1.39e-11` | `1.66e-11` |

The qualification is deliberately a material-point test. It does not include
host substepping or the composite finite-difference tangent: those are
derivatives of the driver-level composed map, not of the one-step constitutive
behaviour. Méric's structural shell is generated in a temporary build for the
qualification; the committed full-field generic backend remains SRIX-specific
until the source-generation and transport buffer are industrialised.

Run it with:

```bash
set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u
.venv/bin/python scripts/qualify_structural_plane_stress_same_state_schur.py
```

The permanent integration test is
`tests/integration/test_structural_plane_stress_same_state_schur.py`.
