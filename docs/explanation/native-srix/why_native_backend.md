# Why implement a native SRIX backend?

**Mode:** explanation  
**Domain:** crystal-plasticity

MFront is already an efficient and qualified CPU backend. The reason to add a
Python/NumPy/Numba implementation is architectural: it makes the local slip
residual, its Jacobian and the coupled plane-stress equations explicit. Those
objects are hidden behind the generic MFront response interface and are
needed for a future accelerator implementation.

The native backend is accepted only after comparison with MFront on the same
parameters, orientations, histories and fields. It is therefore a second
implementation of the qualified behaviour, not a new material calibration.
