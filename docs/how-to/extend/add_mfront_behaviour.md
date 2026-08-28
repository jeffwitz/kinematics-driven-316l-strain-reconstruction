# Add an MFront behaviour

**Mode:** how-to  
**Domain:** software

Implement the behaviour in the repository's MFront validation area, expose
the required state and tangent metadata, and build the shared library with
the documented TFEL toolchain.  Register the behaviour identifier in the
runtime factory only after a material-point comparison against the canonical
reference.

Run the focused MFront tests, record the source and library hashes in the
campaign manifest, and keep the default backend unchanged until qualification
is complete.  See {doc}`../../reference/software/extension_interfaces` for the
adapter contract.
