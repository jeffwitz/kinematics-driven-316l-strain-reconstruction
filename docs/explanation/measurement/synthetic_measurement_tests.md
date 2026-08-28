# What synthetic DIC tests tell us

**Mode:** explanation  
**Domain:** measurement

Synthetic tests separate measurement-chain resolution from mechanical-model
error. A known displacement field is passed through the image/measurement
pipeline and compared with the recovered field using pre-declared masks and
metrics. A failure here limits the interpretation of any later FEMU result;
success does not validate a constitutive law.

The existing detailed results remain in
{doc}`../dic_synthetic_measurement_tests` and the metric definitions are in
{doc}`../../reference/evidence/validation_metrics`.
