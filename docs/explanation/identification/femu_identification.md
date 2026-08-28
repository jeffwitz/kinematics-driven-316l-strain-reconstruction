# FEMU and full-field identification

**Mode:** explanation  
**Domain:** identification

FEMU embeds the constitutive law in the equilibrium forward problem and
compares predicted displacement or strain fields with observations. Each
parameter perturbation may require a complete forward solve, so the cost of
the material and global Newton/Krylov layers directly controls the practical
identification budget.

The native SRIX architecture reduces this cost while preserving an MFront
oracle; it does not make an unobservable parameter identifiable.
