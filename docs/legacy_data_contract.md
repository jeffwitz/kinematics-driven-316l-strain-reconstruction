# Contrat de données des scripts historiques

> Ce document décrit uniquement la couche de compatibilité. Pour le workflow
> principal autonome et les données désormais versionnées, utiliser
> `docs/from_dic_to_reconstruction.md`.

Les trois scripts de comparaison conservés à la racine provenaient d'une
arborescence externe non livrée et contenaient des chemins Windows personnels.
Ils utilisent maintenant `fem_inhouse.legacy_config`.

Par défaut, le chargeur cherche dans `./data/case_study` :

- `displacement_x_mm.npy`, champ nodal `(nx + 1, ny + 1)` ;
- `displacement_y_mm.npy`, champ nodal `(nx + 1, ny + 1)` ;
- `yield_stress_mpa.npy`, champ élémentaire `(nx, ny)` strictement positif ;
- `hardening_coefficient_mpa.npy`, champ élémentaire `(nx, ny)` positif ou nul.

Les champs plus grands sont centrés sur la fenêtre configurée. Les champs plus
petits, tridimensionnels ou non finis sont refusés explicitement.

Les emplacements sont modifiables sans éditer le code :

```bash
export FEM_INHOUSE_INPUT_DIR=/chemin/vers/les/quatre/champs
export FEM_INHOUSE_DIC_DIR=/chemin/vers/les/U_k_et_V_k
export FEM_INHOUSE_MACRO_FILE=/chemin/vers/stress_strain.npy
export FEM_INHOUSE_RESULTS_DIR=/chemin/vers/les/resultats
```

Les tailles historiques se configurent avec
`FEM_INHOUSE_LEGACY_NX` et `FEM_INHOUSE_LEGACY_NY`. Les valeurs par défaut
10×10 ne constituent pas une référence scientifique : elles permettent
seulement de conserver un point d'entrée explicite pendant la migration.
