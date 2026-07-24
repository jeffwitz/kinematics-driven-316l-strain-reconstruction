# Exemple réduit reproductible

Cet exemple vérifie le chemin complet configuration → solveur PyPardiso →
résultats typés sur un cas homogène possédant une solution analytique.

## Exécution

Après l'installation verrouillée décrite dans le `README` :

```bash
fem-inhouse backend
fem-inhouse validate --nx 10 --ny 10
fem-inhouse example --nx 10 --ny 10 --output results/reduced
```

La première commande doit annoncer `pypardiso (MKL, multithreaded)`. La
validation applique une traction équibiaxiale homogène visant 400 MPa avec :

- limite d'élasticité : 250 MPa ;
- coefficient de Ludwik : 500 MPa ;
- exposant : 0,245 ;
- loi tabulée nominale à 1000 points.

Le rapport JSON échoue si l'erreur relative sur la contrainte ou PEEQ dépasse
0,5 %, si l'erreur relative du déplacement dépasse `1e-8`, ou si le déséquilibre
relatif de la résultante des réactions dépasse `1e-10`.

La commande `example` produit :

- `displacement_mm.npy` ;
- `stress_mpa.npy` ;
- `equivalent_plastic_strain.npy` ;
- `report.json`, contenant toute la configuration et les erreurs mesurées.

Ces fichiers illustrent le contrat de sortie, mais ne remplacent pas la future
référence Abaqus. Le cas est homogène et ne valide ni les interfaces de
partitions ni les champs hétérogènes de l'article.

## Partitionnement de production

Les manifestes des deux grilles décrites dans l'article peuvent être générés
sans charger les données :

```bash
fem-inhouse layout --count 25 --padding 150 --output results/layout-25.json
fem-inhouse layout --count 100 --padding 150 --output results/layout-100.json
```
