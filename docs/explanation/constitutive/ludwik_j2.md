# J2/Ludwik as the historical baseline

**Mode:** explanation  
**Domain:** constitutive

J2/Ludwik is an isotropic baseline used to qualify the reconstruction pipeline
and expose the limits of a point-local model. In the small-strain plane-stress
setting its yield function is

$$f=\sigma_{eq}-R(p),\qquad R(p)=\sigma_y+Kp^n,$$

with associative deviatoric flow. The historical Abaqus input used a tabulated
Ludwik curve; the maintained analytical MFront law continues the same curve
beyond the last tabulated point. Matching the constitutive definition is not
the same as proving FE parity with the unavailable original `.inp`, ODB and
extraction procedure.

The registered qualification separates table-versus-analytic material-point
agreement (`E-TABLE-001`) from the finite-element reconstruction checks
(`E-LOCAL-001`, `E-LOCAL-002`). J2 reproduces the mechanics chain and provides
a diagnostic baseline, but its isotropic local response cannot resolve crystal
slip systems, EBSD orientation or spatial interactions. It is therefore a
baseline, not a substitute for SRIX.

See {doc}`../../reference/scientific/constitutive_models` for the model
contract and {doc}`../../evidence/index` for the claim boundaries.
