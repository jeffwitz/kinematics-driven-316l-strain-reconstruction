# Run REGM screening

**Mode:** how-to  
**Domain:** identification

Run the pre-registered weak-equilibrium-gap screening on the digital twin,
then pass the candidates through the real DIC observation operator. Report the
two rankings separately: a successful digital-twin ranking does not validate
transfer to measured DIC.
Run the registered digital-twin and DIC-transfer stages from the campaign
configuration, then archive the exact observation operator, masks, thresholds
and ranking metrics. Treat the exact-mechanical ranking and the DIC-transferred
ranking as separate outputs; the current registered DIC transfer is a negative
result and must not be used as a FEMU replacement.

See {doc}`../../explanation/identification/regm_screening` for the equations
and gates, and {doc}`../../reference/evidence/regm_qualification` for the
claim boundary.
