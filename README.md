# TP1 - Apprentissage supervise avec Scikit-learn

Module : Technologies IA - ENSA Fes - ILIA2

Ce depot contient le rendu du TP1 sur l'apprentissage supervise. Il reprend les
etapes classiques d'un projet de machine learning : chargement, nettoyage,
preparation des donnees, entrainement, evaluation et interpretation des modeles.

## Contenu

| Fichier | Description |
|---|---|
| `exercice1_regression_uber.ipynb` | Regression : prediction du tarif d'un trajet Uber (regression lineaire vs foret aleatoire). |
| `exercice2_classification_titanic.ipynb` | Classification : prediction de la survie sur le Titanic (regression logistique vs foret aleatoire). |
| `miniprojet_breast_cancer.ipynb` | Partie 2 : comparaison rigoureuse de trois modeles sur le diagnostic du cancer du sein, avec conclusion critique. |
| `scripts/prepare_data.py` | Prepare les jeux de donnees dans `data/`. |
| `scripts/build_notebooks.py` | Reconstruit les notebooks a partir du code source. |
| `data/` | Jeux de donnees utilises (voir `data/README.md`). |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

Les notebooks sont deja executes (les sorties et les graphiques sont visibles
directement sur GitHub). Pour les relancer :

```bash
python scripts/prepare_data.py          # regenere les donnees si besoin
jupyter notebook                        # puis ouvrir un des .ipynb
```

## Resultats en bref

- Exercice 1 (Uber) : la distance est la variable dominante. La regression
  lineaire explique environ 82 % de la variance ; la foret aleatoire n'apporte
  pas de gain notable sur ce jeu de donnees.
- Exercice 2 (Titanic) : le sexe et la classe sont les facteurs determinants.
  La foret aleatoire atteint environ 82 % d'exactitude, legerement au-dessus de
  la regression logistique.
- Mini-projet (cancer du sein) : les trois modeles sont performants (AUC > 0.99).
  La regression logistique est retenue pour son interpretabilite a performance
  equivalente au SVM.
