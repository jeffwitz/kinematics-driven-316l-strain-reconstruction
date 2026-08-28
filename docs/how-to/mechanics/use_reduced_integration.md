# Qualify reduced integration

**Mode:** how-to  
**Domain:** constitutive

Run the reduced-integration candidate with the same loading, material and
boundary data as the CPS4 reference:

```bash
MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so" \
python scripts/qualify_reduced_integration.py \
  --mesh 32 --crystal-mesh 8 --repeats 5 \
  --output validation/_generated/cps4r_qualification
```

Compare field errors, equilibrium, hourglass energy and convergence; the
energy ratio alone cannot qualify a plastic full-field run. Use the formulation in
{doc}`../../reference/numerics/cps4r_hourglass` and the negative qualification
criteria in {doc}`../../explanation/constitutive/reduced_integration_hourglass`.

The expected report is the JSON/CSV output under the declared output directory.
The registered campaign found no beta satisfying the plastic accuracy bound;
CPS4 remains the reference. Do not treat a low hourglass ratio as a pass.
