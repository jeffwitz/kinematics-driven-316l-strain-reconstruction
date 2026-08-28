# Validation metrics

**Mode:** reference  
**Domain:** evidence

Field comparisons use the same valid mask and report RMSE, MAE, signed mean
error, maximum absolute error, relative L2 error and Pearson correlation.
Localisation comparisons use quantile-selected masks and report Jaccard/IoU,
Dice, precision and recall. Shapes, masks and physical registration must be
compatible before a metric is computed; no metric performs an implicit
recalibration.

The executable definitions and thresholded command remain documented in the
legacy validation guide and the evidence registry.
