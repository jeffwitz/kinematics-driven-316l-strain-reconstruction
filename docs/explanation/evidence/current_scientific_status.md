# Current scientific status

**Mode:** explanation  
**Domain:** evidence

The project currently combines measured DIC kinematics and EBSD orientations
with three-dimensional constitutive laws under structural plane stress. MFront
is the qualified generic CPU reference; native SRIX is a matching, faster
architecture for explicit coupled local algebra and future GPU work.

The evidence portal distinguishes verified numerical equivalence, supported
scientific interpretation, falsified candidate methods and open questions.
Numerical qualification is not parameter identification: the native SRIX
backend is equivalent to the registered reference on its recorded cases, but
the registered ``R`` is an analytical transposition and experimental P43
identification remains open. Likewise, the existing FEMU smoke driver is a
full-field-Dirichlet negative control; no boundary-only production workflow is
currently claimed. No performance benchmark alone upgrades a scientific
status.
