# J2/Ludwik as the historical baseline

**Mode:** explanation  
**Domain:** constitutive

J2/Ludwik is an isotropic baseline used to qualify the reconstruction pipeline
and expose the limits of a point-local model. Let
$\boldsymbol{s}=\boldsymbol{\sigma}-\tfrac13\operatorname{tr}(\boldsymbol{\sigma})\boldsymbol{I}$.
The von Mises equivalent stress is

$$
\sigma_{eq}=\sqrt{\frac32\,\boldsymbol{s}:\boldsymbol{s}},
$$

and the yield function is

$$
f=\sigma_{eq}-R(p),\qquad R(p)=\sigma_y+Kp^n.
$$

The flow is associative and deviatoric, for example
$\dot{\boldsymbol{\varepsilon}}^p=\dot\lambda\,3\boldsymbol{s}/(2\sigma_{eq})$.
The scalar $p$ is accumulated equivalent plastic strain: it carries isotropic
hardening memory, but no slip-system or directional memory. The maintained
implementation regularises the origin of the Ludwik power as specified in the
Reference page.

The historical Abaqus input used a tabulated Ludwik curve; the maintained
analytical MFront law continues the same curve beyond the last tabulated point.
Matching the constitutive definition is not the same as proving FE parity with
the unavailable original `.inp`, ODB and extraction procedure.

J2 is useful because it is simple and robust: it checks boundary data,
equilibrium and the constitutive plumbing before crystal-specific effects are
introduced. Its defining limitation is that the same response is available at
a point regardless of its EBSD orientation. It cannot resolve crystal slip
systems, latent interactions or grain-scale anisotropy, so it is a baseline
rather than a substitute for SRIX. The registered P43 baseline also shows that
a mechanically valid isotropic reconstruction need not reproduce the observed
localisation morphology.

## Status boundary

* **Law/formulation:** verified isotropic associative J2 with regularised
  Ludwik hardening.
* **Implementation:** table-to-analytic material-point agreement and the
  registered FE checks are separate claims (`E-TABLE-001`, `E-LOCAL-001`,
  `E-LOCAL-002`).
* **Material calibration:** the baseline parameters are provenance data; this
  page does not claim an experimental crystal-plasticity calibration.

See {doc}`../../reference/scientific/constitutive_models` for the model
contract, {doc}`../../adr/0005-mfront-analytical-default` for implementation
provenance, and {doc}`../../evidence/index` for the claim boundaries.
