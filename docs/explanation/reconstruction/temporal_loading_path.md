# Why the temporal loading path matters

**Mode:** explanation  
**Domain:** reconstruction

The DIC sequence defines a loading path. For rate-independent SRIX, changing
the numerical time increment should not change the constitutive result, while
the Méric--Cailletaud comparison can depend on temporal refinement because it
contains viscous evolution. This is a model distinction, not a reason to
discard frames or interpolate internal states.
