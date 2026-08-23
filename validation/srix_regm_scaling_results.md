# SRIX-REGM scaling benchmark

Date: 2026-08-23  
Primary evidence: `reference_data/srix_regm_scaling_v1/report.json`  
Source commit: `a402a5aea920de041f75e147673bef0ec1189652` (clean)

This is a performance benchmark on a fixed 32-state affine history. It is not
an equilibrated twin and its residual value has no scientific interpretation.

| Mesh | Points | `K0` build/factor | One REGM evaluation | Material | Weak residual | `K0^-1` | Observation |
|---|---:|---:|---:|---:|---:|---:|---:|
| M20 | 800 | 0.200 s | 1.270 s | 1.208 s | 0.0099 s | 0.0134 s | 0.000057 s |
| M100 | 20,000 | 4.397 s | 19.708 s | 18.693 s | 0.0391 s | 0.447 s | 0.000035 s |

`K0` is built and factorised once, outside parameter evaluation. On M100,
material replay accounts for 94.9 % of evaluation time; the reconditioner is
2.3 %. A new FFT inverse is therefore not justified by this profile.

The qualified external 3D plane-stress condensation still requests local
constitutive tangents internally to close the transverse stresses, even though
REGM requests no returned global tangent. The next performance comparison must
therefore include the already-qualified native/generalised plane-stress SRIX
path before selecting the production replay backend.

