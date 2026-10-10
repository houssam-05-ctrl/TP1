# Jeux de donnees

## titanic.csv

Jeu Titanic reel (891 passagers), recupere via la bibliotheque seaborn et
realigne sur le schema du dataset Kaggle (colonnes `Survived`, `Pclass`, `Sex`,
`Age`, `SibSp`, `Parch`, `Fare`, `Embarked`).

## uber.csv

Jeu synthetique (50 000 courses) qui reproduit exactement le schema du dataset
"Uber Fares" de Kaggle : coordonnees GPS de depart et d'arrivee, horodatage,
nombre de passagers et montant de la course. Il est genere par
`scripts/prepare_data.py` avec une graine fixe, ce qui rend le notebook
reproductible sans compte Kaggle.

Des defauts realistes y sont injectes volontairement pour illustrer l'etape de
nettoyage : valeurs manquantes sur la destination, tarifs negatifs et tarifs
aberrants.

Pour travailler sur les vraies donnees, il suffit de telecharger le fichier
depuis Kaggle (https://www.kaggle.com/datasets/yasserh/uber-fares-dataset) et de
remplacer `uber.csv` : le notebook calcule la distance a partir des coordonnees,
le schema est donc compatible.

Regeneration des deux fichiers :

```bash
python scripts/prepare_data.py
```
