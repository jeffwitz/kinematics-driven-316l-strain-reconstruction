# Characterise a DIC measurement chain

**Mode:** how-to  
**Domain:** dic

Before a reconstruction, record the image sequence, crop, pixel size, mask,
component mapping and temporal correspondence in the case manifest.  Run the
synthetic and repeated-frame checks when their inputs are available, then
report displacement noise and the valid support used by the observation
operator.

Keep measurement uncertainty separate from the reconstructed mechanical
field.  The stable input and observation contracts are in
{doc}`../../reference/scientific/observation_operator` and
{doc}`../../reference/data/dic_axis_conventions`.
