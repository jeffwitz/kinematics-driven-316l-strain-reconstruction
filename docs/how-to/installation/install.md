# Install the project

**Mode:** how-to  
**Domain:** software

Create the project environment, install the package and run the quick unit
tests before preparing a scientific case. From the repository root, use the
documented environment files or install the package in editable mode, then
run `python -m pytest tests/unit -q`. Keep compiler/TFEL paths in the case
manifest; platform-specific constitutive setup is covered by
{doc}`build_mfront`.
