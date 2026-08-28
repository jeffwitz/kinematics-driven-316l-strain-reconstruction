# From DIC to a mechanical reconstruction

**Mode:** explanation  
**Domain:** reconstruction

DIC supplies measured boundary kinematics, not a complete equilibrated
mechanical field. The reconstruction therefore combines measured displacement
data, a constitutive law and an equilibrium solver. The observation operator,
axis conventions and admissible masks are contracts, not tuning parameters;
see {doc}`../../reference/scientific/observation_operator` and
{doc}`../../reference/dic_axis_conventions`.

The central distinction is between reproducing the measured boundary motion
and predicting an interior field. A local constitutive response can fit the
boundary while missing morphology inside the specimen. This is why full-field
validation and the distinction between reconstruction, identification and
prediction matter.

For the runnable workflow, use {doc}`../../how-to/prepare_data` and
{doc}`../../how-to/run_local_reconstruction`.
