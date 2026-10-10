"""
Preparation des jeux de donnees utilises dans le TP.

- data/titanic.csv : jeu Titanic reel recupere via seaborn, renomme pour
  correspondre au schema du dataset Kaggle utilise dans l'enonce.
- data/uber.csv    : jeu synthetique qui reproduit le schema du dataset
  "Uber Fares" de Kaggle (coordonnees GPS, horodatage, montant de la course).
  Il sert uniquement a rendre le notebook reproductible sans compte Kaggle ;
  il suffit de remplacer ce fichier par le vrai uber.csv pour refaire l'analyse
  sur les donnees reelles.

Lancer depuis la racine du projet :  python scripts/prepare_data.py
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)


def prepare_titanic():
    """Recupere le Titanic via seaborn et l'aligne sur le schema Kaggle."""
    import seaborn as sns

    df = sns.load_dataset("titanic")
    # On garde les colonnes brutes equivalentes a celles de Kaggle
    out = pd.DataFrame({
        "Survived": df["survived"],
        "Pclass": df["pclass"],
        "Sex": df["sex"],
        "Age": df["age"],
        "SibSp": df["sibsp"],
        "Parch": df["parch"],
        "Fare": df["fare"],
        "Embarked": df["embarked"],
    })
    out.to_csv(os.path.join(DATA, "titanic.csv"), index=False)
    print(f"titanic.csv : {len(out)} lignes")


def haversine(lon1, lat1, lon2, lat2):
    """Distance en km entre deux points GPS (formule de haversine)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def prepare_uber(n=50000, seed=42):
    """Genere un jeu synthetique au format du dataset Uber Fares de Kaggle."""
    rng = np.random.default_rng(seed)

    # Zone de New York (comme le vrai dataset)
    lon0, lat0 = -73.97, 40.75
    pickup_lon = rng.normal(lon0, 0.03, n)
    pickup_lat = rng.normal(lat0, 0.03, n)
    # trajet : petit deplacement autour du point de depart
    dlon = rng.normal(0, 0.03, n)
    dlat = rng.normal(0, 0.03, n)
    dropoff_lon = pickup_lon + dlon
    dropoff_lat = pickup_lat + dlat

    dist = haversine(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)

    # horodatage reparti sur une annee
    start = pd.Timestamp("2014-01-01")
    seconds = rng.integers(0, 365 * 24 * 3600, n)
    pickup_dt = start + pd.to_timedelta(seconds, unit="s")
    hour = pickup_dt.hour.to_numpy()

    # supplement heures de pointe (7-9h et 17-19h)
    rush = np.isin(hour, [7, 8, 9, 17, 18, 19]).astype(float)

    passengers = rng.integers(1, 7, n)

    # modele de prix : prise en charge + tarif kilometrique + pointe + bruit
    fare = 2.5 + 1.8 * dist + 1.2 * rush + rng.normal(0, 1.5, n)
    fare = np.round(fare, 2)

    df = pd.DataFrame({
        "key": [f"{t}.{i}" for i, t in enumerate(pickup_dt.astype(str))],
        "fare_amount": fare,
        "pickup_datetime": pickup_dt.astype(str),
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "passenger_count": passengers,
    })

    # On injecte des defauts realistes a nettoyer dans le notebook :
    # quelques valeurs manquantes sur la destination
    miss = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[miss, ["dropoff_longitude", "dropoff_latitude"]] = np.nan
    # quelques tarifs nuls ou negatifs (erreurs de saisie)
    bad = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[bad, "fare_amount"] = rng.choice([0, -2.5, -5.0], size=len(bad))
    # quelques tarifs aberrants tres eleves
    out = rng.choice(n, size=int(0.005 * n), replace=False)
    df.loc[out, "fare_amount"] = rng.uniform(300, 600, size=len(out))

    df.to_csv(os.path.join(DATA, "uber.csv"), index=False)
    print(f"uber.csv : {len(df)} lignes (synthetique)")


if __name__ == "__main__":
    prepare_titanic()
    prepare_uber()
    print("Donnees ecrites dans", DATA)
