# Python API

**Category: Reference.**

The command line is the stable user interface. Reusable scientific contracts
are grouped as follows:

| Module | Public role |
|---|---|
| `fem_inhouse.config` | typed mesh, material, solver and nonlocal configuration |
| `fem_inhouse.results` | result fields and solver diagnostics |
| `fem_inhouse.core.tensor_reconstruction` | engineering, Kelvin and complete-tensor conversion |
| `fem_inhouse.core.mfront` | compatibility façade for MFront/MGIS material bridges |
| `fem_inhouse.core.plane_stress_material` | common constitutive protocol and backend factory |
| `fem_inhouse.postprocessing.helmholtz` | element-centred Helmholtz filter |
| `fem_inhouse.postprocessing.metrics` | field, overlap and diffusivity metrics |
| `fem_inhouse.workflows.nonlocality_diagnostic` | output-only spatial-width campaign |
| `fem_inhouse.workflows.joint_nonlocal_identification` | F0/F1/F2 identification workflow |

Docstrings and type annotations in the installed revision are authoritative for
individual signatures.

Application code should normally select a registered backend through the
configuration and factory layer rather than instantiate low-level MFront
adapters directly.

## Stability

Public dataclasses and result field names are compatibility-sensitive.
Workflow helpers may evolve while the research design is active. Functions
whose name begins with an underscore are not public API.
