# P43-EXP-001 — pré-enregistrement M20 expérimental rang 3

Cette campagne est la première identification sur l'histoire DIC P43 réelle.
Elle utilise le crop M20 `[1610:1630,1075:1095]`, l'EBSD co-enregistré, les
états DIC réparés 0,5,...,40 et 32 incréments fixes (4 par segment macro).

Prior : `(tau0,R,Q,b)=(40,18.7819100705,10,3)` MPa/-, avec `C=40000 MPa` et
`d=1500` inchangés. Le résidu compare les déplacements FEMU aux déplacements
DIC préparés, sans transfert spatial additionnel ; chaque composante est
whitened par `9.40e-5 mm`, l'incertitude DIC enregistrée. La Jacobienne directe
emploie `shadow_fd_step=0.0015`.

Au prior, la Jacobienne est décomposée par SVD et son rang est fixé à 3 avant
l'optimisation. Pour chaque contrôle multi-départ, `z4` est conservé à sa
valeur initiale et seuls `z1,z2,z3` sont optimisés. La base ne sera pas
réajustée a posteriori ; sa rotation à la solution est seulement diagnostique.

Le M100 est interdit tant que le rapport M20 n'indique pas explicitement le
passage du gate : convergence numérique, diminution du mismatch, solutions
cohérentes dans le sous-espace observable et absence de dérive pathologique.

