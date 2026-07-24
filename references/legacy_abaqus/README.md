# Scripts Abaqus historiques

Ces deux scripts sont des copies byte-à-byte des sources reçues avec les
données. Ils sont conservés comme références de provenance, et non comme
composants exécutables du paquet.

Ils contiennent encore les chemins Windows d'origine. Le pipeline autonome ne
les importe pas et ne dépend pas de JAX ni d'Abaqus. Ils servent notamment à
établir les conventions suivantes :

- `U_40.npy` alimente le déplacement `u_y` ;
- `V_40.npy` alimente le déplacement `u_x` ;
- un pixel DIC correspond à `0,00184 mm` ;
- la carte historique d'écrouissage est multipliée par `396 MPa` ;
- les conditions aux limites sont imposées sur les quatre côtés du sous-domaine.
