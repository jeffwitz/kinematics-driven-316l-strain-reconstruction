# Experimental-data inventory

**Mode:** reference  
**Domain:** data

This contract distinguishes received bytes, prepared data and unavailable
metadata.  The maintained P43 inventory records the 42 DIC images, the
prepared `U_40`/`V_40` arrays, the crop and pixel size, EBSD-derived fields,
and the provisional image-to-load correspondence.  Missing load-cell time
series, acquisition timestamps, native EBSD step size and the historical mask
remain explicitly unavailable rather than inferred.

The detailed inventory and hashes live in the validation manifests.  A
calculation must record input hashes, preparation transformations, masks,
units and the observation operator used.
