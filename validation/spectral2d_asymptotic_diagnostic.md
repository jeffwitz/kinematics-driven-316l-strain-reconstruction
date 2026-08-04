# Diagnostic asymptotique TRI2

Cas SRIX enregistré, grille 4x4, huit incréments, `B0`, mémoire Anderson 4.

| Variante | Tolérance | Itérations | Résidu final | rho médian récent |
|---|---:|---:|---:|---:|
| déplacement | 1e-6 | 668 | 9.96e-7 | 0.986 |
| polarisation | 1e-6 | 314 | 9.98e-7 | 0.874 |
| polarisation | 1e-8 | 524 | 8.44e-9 | 0.879 |

Les traces JSONL sont produites par le script de qualification, avec flush à
chaque itération, et restent disponibles en cas d’échec ou de timeout.

Conclusion limitée : l’Anderson sur la polarisation améliore nettement la
convergence et atteint 1e-8 sur ce cas réduit. Cela ne constitue pas encore une
qualification sur la grille 12x12 ni une comparaison complète des champs.

## Réplication ACV_P corrigée

Avec une seule polarisation par itération, cadence 2, aucun Armijo et une
référence projetée multipliée par 0,5, le schéma one-point 4x4 n'atteint pas
1e-5 : résidu final observé 2,40e-5 après 3313 itérations et cutbacks. La part
d'énergie dans les 10 % de modes les plus élevés atteint 5,65 %, contre 0,14 %
pour le comparateur à deux sous-cellules dans la même exécution interrompue.

Ce résultat confirme que le changement de nom n'a pas supprimé le mécanisme
proche-hourglass. Une campagne 4/8/12/24 reste nécessaire avant falsification
formelle, mais le one-point n'est pas qualifié.
