# Nested and coupled plane-stress closures

**Mode:** explanation  
**Domain:** plane-stress

The nested method converges SRIX for a trial transverse strain and then
corrects that strain. The coupled method solves the twelve slip equations and
the three zero-traction equations in one local Newton system with block
Jacobian \([A\ B; C\ D]\). A local Schur complement avoids a dense 15-by-15
solve while retaining the same equations.

The nested path remains the reference and is required for generic MFront. The
coupled path is a validated high-performance native option; it is not a
different constitutive law.
