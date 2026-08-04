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
