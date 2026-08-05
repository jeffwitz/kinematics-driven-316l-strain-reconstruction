# Conclusions and scope

The evidence separates three requirements:

* DTT and Newton-GMRES provide a convergent global solve;
* TET2 removes the one-point spatial near-null mode;
* two SRIX histories preserve the local plastic evolution.

The qualified numerical oracle is traditional TET2 with two states per pixel.
The one-state Hookean EBI-TET formulation is convergent and inexpensive, but
is `experimental_falsified_for_registered_SRIX_case` because its accumulated
slip differs materially from TET2 at identical kinematics.

The next one-state method must change the constitutive representation or the
spatial sampling strategy. More Anderson tuning, FFT optimization or tighter
Newton tolerances cannot recover information removed by state sharing.

The production-law decision is separate from the EBI falsification. SRIX is
the qualified law for the registered P43 quasi-static reconstruction because
its response does not depend on an undocumented elapsed time. Méric-Cailletaud
is retained as a viscoplastic comparison law: eight increments fail local
plane-stress condensation, while 16 increments converge numerically at a much
higher cost, without establishing temporal convergence of the fields. See
{doc}`srix_production_choice` for the complete evidence and scope boundary.

```{figure} ../../_static/spectral_mechanics/runtime_comparison.png
:alt: Indicative 24 by 24 runtime comparison.
:name: spectral-runtime-comparison

The timings are single-run evidence, not a repeat-qualified performance claim.
```
