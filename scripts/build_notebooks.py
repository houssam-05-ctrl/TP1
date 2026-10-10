"""
Construit les trois notebooks du TP a partir de cellules definies ici.
On passe par nbformat pour garder le code et les commentaires lisibles.
Lancer :  python scripts/build_notebooks.py
Puis executer :  jupyter nbconvert --to notebook --execute --inplace *.ipynb
"""

import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(path, cells):
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(os.path.join(HERE, path), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("ecrit :", path)


def md(text):
    return new_markdown_cell(text)


def code(src):
    return new_code_cell(src)


# =====================================================================
#  Exercice 1 : regression (tarifs Uber)
# =====================================================================
ex1 = [
    md("# Exercice 1 - Regression : prediction du tarif d'un trajet Uber\n"
       "\n"
       "Objectif : predire le montant d'une course Uber a partir de la distance, "
       "de l'heure et du jour de la semaine.\n"
       "\n"
       "Jeu de donnees : *Uber Fares Dataset* (Kaggle). Le fichier `data/uber.csv` "
       "reprend le schema du dataset Kaggle (coordonnees GPS, horodatage, montant). "
       "Il suffit de le remplacer par le fichier Kaggle pour refaire l'analyse sur "
       "les donnees reelles."),

    md("## Etape 1 - Chargement et exploration"),
    code("import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "\n"
         "pd.set_option('display.max_columns', None)\n"
         "\n"
         "data = pd.read_csv('data/uber.csv')\n"
         "data.head()"),
    code("# Types des colonnes et nombre de valeurs non nulles\n"
         "data.info()"),
    code("# Comptage des valeurs manquantes par colonne\n"
         "data.isnull().sum()"),
    code("# Statistiques descriptives : on repere deja des tarifs negatifs et\n"
         "# des valeurs maximales aberrantes sur fare_amount.\n"
         "data.describe()"),

    md("## Etape 2 - Nettoyage et preparation\n"
       "\n"
       "On enleve les lignes incompletes, les tarifs incoherents (negatifs ou nuls) "
       "et les valeurs extremes. On calcule ensuite la distance du trajet a partir "
       "des coordonnees GPS (formule de haversine) et on derive l'heure et le jour "
       "de la semaine depuis l'horodatage."),
    code("# Suppression des valeurs manquantes (destination absente notamment)\n"
         "data = data.dropna()\n"
         "\n"
         "# Un tarif doit etre strictement positif\n"
         "data = data[data['fare_amount'] > 0]\n"
         "\n"
         "print('Lignes restantes apres premier nettoyage :', len(data))"),
    code("def haversine(lon1, lat1, lon2, lat2):\n"
         "    \"\"\"Distance en km entre deux points GPS.\"\"\"\n"
         "    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])\n"
         "    dlon = lon2 - lon1\n"
         "    dlat = lat2 - lat1\n"
         "    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2\n"
         "    return 6371.0 * 2 * np.arcsin(np.sqrt(a))\n"
         "\n"
         "data['distance'] = haversine(data['pickup_longitude'], data['pickup_latitude'],\n"
         "                             data['dropoff_longitude'], data['dropoff_latitude'])"),
    code("# Variables temporelles derivees de l'horodatage\n"
         "data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])\n"
         "data['hour'] = data['pickup_datetime'].dt.hour\n"
         "data['day_of_week'] = data['pickup_datetime'].dt.dayofweek"),
    code("# Retrait des valeurs aberrantes : on borne le tarif et la distance\n"
         "# en coupant les 1 % extremes, et on garde des trajets plausibles.\n"
         "q_low, q_high = data['fare_amount'].quantile([0.01, 0.99])\n"
         "data = data[(data['fare_amount'] >= q_low) & (data['fare_amount'] <= q_high)]\n"
         "data = data[(data['distance'] > 0) & (data['distance'] < 50)]\n"
         "\n"
         "print('Lignes conservees pour la modelisation :', len(data))\n"
         "data[['fare_amount', 'distance', 'hour', 'day_of_week']].describe()"),

    md("## Etape 3 - Separation des donnees\n"
       "\n"
       "On predit `fare_amount` a partir de la distance, de l'heure, du jour de la "
       "semaine et du nombre de passagers. 80 % des donnees servent a l'entrainement, "
       "20 % au test."),
    code("from sklearn.model_selection import train_test_split\n"
         "\n"
         "features = ['distance', 'hour', 'day_of_week', 'passenger_count']\n"
         "X = data[features]\n"
         "y = data['fare_amount']\n"
         "\n"
         "X_train, X_test, y_train, y_test = train_test_split(\n"
         "    X, y, test_size=0.2, random_state=42)\n"
         "print(X_train.shape, X_test.shape)"),

    md("## Etape 4 - Entrainement d'une regression lineaire"),
    code("from sklearn.linear_model import LinearRegression\n"
         "\n"
         "lr = LinearRegression()\n"
         "lr.fit(X_train, y_train)\n"
         "y_pred_lr = lr.predict(X_test)\n"
         "\n"
         "# Coefficients : effet de chaque variable sur le tarif\n"
         "pd.Series(lr.coef_, index=features)"),

    md("## Etape 5 - Evaluation du modele\n"
       "\n"
       "On mesure l'erreur absolue moyenne (MAE), l'erreur quadratique moyenne "
       "(RMSE) et le coefficient de determination R2."),
    code("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
         "\n"
         "def evaluer(nom, y_vrai, y_prev):\n"
         "    mae = mean_absolute_error(y_vrai, y_prev)\n"
         "    rmse = np.sqrt(mean_squared_error(y_vrai, y_prev))\n"
         "    r2 = r2_score(y_vrai, y_prev)\n"
         "    print(f'{nom:18s} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}')\n"
         "    return mae, rmse, r2\n"
         "\n"
         "evaluer('Regression lin.', y_test, y_pred_lr)"),

    md("## Etape 6 - Amelioration avec une foret aleatoire\n"
       "\n"
       "La relation n'est pas purement lineaire (effet des heures de pointe). "
       "On compare avec un `RandomForestRegressor`."),
    code("from sklearn.ensemble import RandomForestRegressor\n"
         "\n"
         "rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)\n"
         "rf.fit(X_train, y_train)\n"
         "y_pred_rf = rf.predict(X_test)\n"
         "\n"
         "evaluer('Regression lin.', y_test, y_pred_lr)\n"
         "evaluer('Foret aleatoire', y_test, y_pred_rf)"),

    md("## Etape 7 - Visualisation des resultats"),
    code("# Valeurs predites par la foret aleatoire face aux valeurs reelles.\n"
         "# Plus les points sont proches de la diagonale, meilleure est la prediction.\n"
         "plt.figure(figsize=(6, 6))\n"
         "plt.scatter(y_test, y_pred_rf, alpha=0.3, s=10)\n"
         "lims = [y_test.min(), y_test.max()]\n"
         "plt.plot(lims, lims, 'r--', linewidth=1)\n"
         "plt.xlabel('Valeurs reelles')\n"
         "plt.ylabel('Valeurs predites')\n"
         "plt.title('Prediction du tarif Uber (foret aleatoire)')\n"
         "plt.show()"),
    code("# Importance des variables selon la foret aleatoire\n"
         "importances = pd.Series(rf.feature_importances_, index=features).sort_values()\n"
         "importances.plot(kind='barh')\n"
         "plt.title('Importance des variables')\n"
         "plt.xlabel('Importance')\n"
         "plt.show()"),

    md("## Conclusion\n"
       "\n"
       "La distance est de loin la variable la plus explicative du tarif, ce qui est "
       "attendu. La regression lineaire explique deja environ 82 % de la variance "
       "(R2 proche de 0.82). La foret aleatoire obtient ici des scores quasi "
       "identiques, sans gain reel, pour un temps de calcul bien plus eleve : sur ce "
       "jeu de donnees, la regression lineaire est donc preferable (plus simple, plus "
       "rapide et interpretable). Sur les donnees reelles de Kaggle, plus bruitees et "
       "avec davantage de variables, la foret aleatoire pourrait en revanche faire la "
       "difference."),
]

# =====================================================================
#  Exercice 2 : classification (survie Titanic)
# =====================================================================
ex2 = [
    md("# Exercice 2 - Classification : survie sur le Titanic\n"
       "\n"
       "Objectif : predire si un passager a survecu (1) ou non (0) au naufrage.\n"
       "\n"
       "Jeu de donnees : *Titanic* (Kaggle). Le fichier `data/titanic.csv` contient "
       "les colonnes classiques du probleme."),

    md("## Etape 1 - Chargement et exploration"),
    code("import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n"
         "\n"
         "data = pd.read_csv('data/titanic.csv')\n"
         "data.head()"),
    code("data.info()"),
    code("# Valeurs manquantes : Age et Embarked en comportent\n"
         "data.isnull().sum()"),

    md("## Etape 2 - Nettoyage et traitement\n"
       "\n"
       "On remplace l'age manquant par la mediane (robuste aux valeurs extremes) et "
       "le port d'embarquement manquant par la valeur la plus frequente (mode)."),
    code("data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]\n"
         "data['Age'] = data['Age'].fillna(data['Age'].median())\n"
         "data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])\n"
         "data.isnull().sum()"),

    md("## Etape 3 - Encodage des variables categorielles\n"
       "\n"
       "Les modeles n'acceptent que des nombres : on encode `Sex` et `Embarked`."),
    code("from sklearn.preprocessing import LabelEncoder\n"
         "\n"
         "le = LabelEncoder()\n"
         "data['Sex'] = le.fit_transform(data['Sex'])          # female=0, male=1\n"
         "data['Embarked'] = le.fit_transform(data['Embarked']) # C=0, Q=1, S=2\n"
         "data.head()"),

    md("## Etape 4 - Separation des donnees"),
    code("from sklearn.model_selection import train_test_split\n"
         "\n"
         "X = data.drop('Survived', axis=1)\n"
         "y = data['Survived']\n"
         "\n"
         "X_train, X_test, y_train, y_test = train_test_split(\n"
         "    X, y, test_size=0.2, random_state=42)\n"
         "print(X_train.shape, X_test.shape)"),

    md("## Etape 5 - Entrainement des modeles\n"
       "\n"
       "On compare une regression logistique (modele lineaire de reference) a une "
       "foret aleatoire."),
    code("from sklearn.linear_model import LogisticRegression\n"
         "from sklearn.ensemble import RandomForestClassifier\n"
         "\n"
         "log_model = LogisticRegression(max_iter=1000)\n"
         "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
         "\n"
         "log_model.fit(X_train, y_train)\n"
         "rf_model.fit(X_train, y_train)"),

    md("## Etape 6 - Evaluation des modeles"),
    code("from sklearn.metrics import accuracy_score, classification_report\n"
         "\n"
         "y_pred_log = log_model.predict(X_test)\n"
         "y_pred_rf = rf_model.predict(X_test)\n"
         "\n"
         "print('Accuracy (regression logistique) :', round(accuracy_score(y_test, y_pred_log), 3))\n"
         "print('Accuracy (foret aleatoire)       :', round(accuracy_score(y_test, y_pred_rf), 3))\n"
         "print()\n"
         "print('Rapport detaille - foret aleatoire :')\n"
         "print(classification_report(y_test, y_pred_rf))"),

    md("## Etape 7 - Matrice de confusion"),
    code("from sklearn.metrics import confusion_matrix\n"
         "\n"
         "cm = confusion_matrix(y_test, y_pred_rf)\n"
         "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n"
         "plt.title('Matrice de confusion - Titanic')\n"
         "plt.xlabel('Predit')\n"
         "plt.ylabel('Reel')\n"
         "plt.show()"),

    md("## Conclusion\n"
       "\n"
       "Le sexe et la classe du passager sont les facteurs les plus determinants de "
       "la survie. La foret aleatoire obtient une exactitude legerement superieure a "
       "la regression logistique et une meilleure detection des survivants. La matrice "
       "de confusion montre que les erreurs se concentrent sur les survivants predits "
       "a tort comme non survivants."),
]

# =====================================================================
#  Partie 2 : mini-projet (comparaison rigoureuse de modeles)
# =====================================================================
mp = [
    md("# Partie 2 - Mini-projet : diagnostic du cancer du sein\n"
       "\n"
       "Jeu de donnees : *Breast Cancer Wisconsin* (fourni avec scikit-learn, donc "
       "reel et directement disponible). Objectif : classer une tumeur comme benigne "
       "ou maligne a partir de 30 mesures issues d'images.\n"
       "\n"
       "On applique le workflow complet et on compare rigoureusement plusieurs "
       "modeles avant de conclure sur celui a retenir."),

    md("## 1. Chargement et exploration"),
    code("import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n"
         "from sklearn.datasets import load_breast_cancer\n"
         "\n"
         "cancer = load_breast_cancer(as_frame=True)\n"
         "df = cancer.frame\n"
         "print('Dimensions :', df.shape)\n"
         "df.head()"),
    code("# Repartition des classes (0 = maligne, 1 = benigne)\n"
         "df['target'].value_counts()"),
    code("# Aucune valeur manquante dans ce jeu de donnees\n"
         "print('Valeurs manquantes :', int(df.isnull().sum().sum()))"),

    md("## 2. Pretraitement\n"
       "\n"
       "On separe les donnees puis on standardise les variables. La mise a l'echelle "
       "est indispensable pour la regression logistique et le SVM, qui sont sensibles "
       "aux ordres de grandeur ; elle est sans effet negatif pour la foret aleatoire."),
    code("from sklearn.model_selection import train_test_split\n"
         "from sklearn.preprocessing import StandardScaler\n"
         "\n"
         "X = df.drop('target', axis=1)\n"
         "y = df['target']\n"
         "\n"
         "X_train, X_test, y_train, y_test = train_test_split(\n"
         "    X, y, test_size=0.2, random_state=42, stratify=y)\n"
         "\n"
         "scaler = StandardScaler()\n"
         "X_train_s = scaler.fit_transform(X_train)\n"
         "X_test_s = scaler.transform(X_test)"),

    md("## 3. Comparaison des modeles par validation croisee\n"
       "\n"
       "On evalue trois modeles par validation croisee a 5 plis sur l'ensemble "
       "d'entrainement, pour comparer leur performance de maniere plus fiable qu'avec "
       "une seule division."),
    code("from sklearn.linear_model import LogisticRegression\n"
         "from sklearn.ensemble import RandomForestClassifier\n"
         "from sklearn.svm import SVC\n"
         "from sklearn.model_selection import cross_val_score\n"
         "\n"
         "modeles = {\n"
         "    'Regression logistique': LogisticRegression(max_iter=5000),\n"
         "    'Foret aleatoire': RandomForestClassifier(n_estimators=200, random_state=42),\n"
         "    'SVM (noyau RBF)': SVC(kernel='rbf', probability=True, random_state=42),\n"
         "}\n"
         "\n"
         "for nom, modele in modeles.items():\n"
         "    scores = cross_val_score(modele, X_train_s, y_train, cv=5, scoring='accuracy')\n"
         "    print(f'{nom:24s} accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})')"),

    md("## 4. Evaluation sur l'ensemble de test\n"
       "\n"
       "On entraine chaque modele sur tout l'entrainement puis on mesure accuracy, "
       "precision, rappel, F1 et l'aire sous la courbe ROC (AUC) sur le test."),
    code("from sklearn.metrics import accuracy_score, f1_score, roc_auc_score\n"
         "\n"
         "resultats = []\n"
         "for nom, modele in modeles.items():\n"
         "    modele.fit(X_train_s, y_train)\n"
         "    y_pred = modele.predict(X_test_s)\n"
         "    y_proba = modele.predict_proba(X_test_s)[:, 1]\n"
         "    resultats.append({\n"
         "        'Modele': nom,\n"
         "        'Accuracy': accuracy_score(y_test, y_pred),\n"
         "        'F1': f1_score(y_test, y_pred),\n"
         "        'AUC': roc_auc_score(y_test, y_proba),\n"
         "    })\n"
         "\n"
         "pd.DataFrame(resultats).set_index('Modele').round(3)"),

    md("## 5. Analyse du meilleur modele\n"
       "\n"
       "On detaille le rapport de classification et la matrice de confusion du modele "
       "le plus performant sur le test."),
    code("from sklearn.metrics import classification_report, confusion_matrix\n"
         "\n"
         "best = SVC(kernel='rbf', probability=True, random_state=42).fit(X_train_s, y_train)\n"
         "y_pred = best.predict(X_test_s)\n"
         "\n"
         "print(classification_report(y_test, y_pred, target_names=['maligne', 'benigne']))\n"
         "\n"
         "cm = confusion_matrix(y_test, y_pred)\n"
         "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
         "            xticklabels=['maligne', 'benigne'], yticklabels=['maligne', 'benigne'])\n"
         "plt.title('Matrice de confusion - SVM')\n"
         "plt.xlabel('Predit')\n"
         "plt.ylabel('Reel')\n"
         "plt.show()"),
    code("from sklearn.metrics import roc_curve\n"
         "\n"
         "# Courbes ROC des trois modeles\n"
         "plt.figure(figsize=(6, 6))\n"
         "for nom, modele in modeles.items():\n"
         "    proba = modele.predict_proba(X_test_s)[:, 1]\n"
         "    fpr, tpr, _ = roc_curve(y_test, proba)\n"
         "    auc = roc_auc_score(y_test, proba)\n"
         "    plt.plot(fpr, tpr, label=f'{nom} (AUC={auc:.3f})')\n"
         "plt.plot([0, 1], [0, 1], 'k--', linewidth=1)\n"
         "plt.xlabel('Taux de faux positifs')\n"
         "plt.ylabel('Taux de vrais positifs')\n"
         "plt.title('Courbes ROC')\n"
         "plt.legend()\n"
         "plt.show()"),

    md("## 6. Conclusion critique\n"
       "\n"
       "Les trois modeles atteignent de tres bons scores sur ce jeu de donnees bien "
       "separable. Le SVM a noyau RBF et la regression logistique (sur donnees "
       "standardisees) obtiennent les meilleures AUC, devant la foret aleatoire.\n"
       "\n"
       "**Modele retenu : la regression logistique.** A performance quasi egale au "
       "SVM, elle est plus simple, plus rapide et surtout interpretable (coefficients "
       "par variable), ce qui est precieux dans un contexte medical ou il faut pouvoir "
       "expliquer une decision. Dans ce probleme, on privilegie le rappel sur la classe "
       "maligne afin de limiter les faux negatifs (tumeurs malignes classees benignes), "
       "plus couteux qu'une fausse alerte.\n"
       "\n"
       "**Limites :** jeu de donnees de petite taille et deja tres propre ; les "
       "resultats seraient a confirmer sur des donnees plus bruitees et avec un "
       "reglage fin des hyperparametres (GridSearchCV)."),
]

if __name__ == "__main__":
    build("exercice1_regression_uber.ipynb", ex1)
    build("exercice2_classification_titanic.ipynb", ex2)
    build("miniprojet_breast_cancer.ipynb", mp)
    print("Notebooks construits.")
