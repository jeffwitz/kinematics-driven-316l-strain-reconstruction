# Qualify the native SRIX backend

**Mode:** how-to  
**Domain:** crystal-plasticity

Run the material-point and P43 M20 comparisons with identical parameters,
orientations, histories and increments for MFront and native SRIX. Compare
stress, elastic strain, slips, hardening, accumulated slip, plane-stress
residual and condensed tangent after every increment. Then compare nested and
coupled native closures before attempting a larger batch.

Archive the command, commit, options, timing, tolerances and field-difference
metrics in a JSON manifest. Keep full fields only for designated golden
references; benchmark runs retain metrics and provenance.

The scientific rationale and optimisation history are in
{doc}`../../explanation/native-srix/optimization_strategy` and the stable
options in {doc}`../../reference/numerics/native_srix_backend`.
