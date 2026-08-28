# From performant CPU Python to a future GPU backend

**Mode:** explanation  
**Domain:** architecture

The CPU backend is deliberately organised as independent point-local
operations with explicit committed and trial state. This permits a future
NumPy-to-CuPy or equivalent array-backend change without re-deriving the
mechanics. The coupled closure is particularly suitable because its local
unknowns and small block solves are explicit.

CuPy, Dask-CUDA, MPI and distributed FFTs are not part of the current
qualification. The present result is a CPU architecture that is scientifically
checked against MFront and technically prepared for that later port.
