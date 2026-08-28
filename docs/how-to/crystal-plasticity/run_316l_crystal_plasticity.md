# Run 316L crystal plasticity

**Mode:** how-to  
**Domain:** crystal-plasticity

Build or select the MFront reference first, verify the EBSD orientation
provenance, then select `mfront` or `numpy-srix` explicitly in the case
configuration. For native SRIX, also choose `nested` or `coupled` plane stress
and record the option in the manifest. Use the parameter and orientation
contracts in {doc}`../../reference/scientific/srix_parameter_sets` and
{doc}`../../reference/scientific/ebsd_orientation_contract`, then qualify the
result with {doc}`qualify_native_srix_backend`.
