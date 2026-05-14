# =============================================================
#  feature_engineering.py — Création des variables prédictives
# =============================================================

import numpy as np
import pandas as pd
from src.config import WINDOW_SIZE, LAG_STEPS, RUL_CLIP


# ─────────────────────────────────────────────────────────────
#  1. Calcul du RUL
# ─────────────────────────────────────────────────────────────

def compute_rul(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    """
    Calcule le RUL (Remaining Useful Life) pour chaque ligne.
    RUL = max_cycle_de_l'équipement − cycle_actuel

    Args:
        df   : DataFrame avec colonnes 'unit_id' et 'cycle'
        clip : Plafond du RUL (évite les valeurs aberrantes en début de vie)

    Returns:
        DataFrame avec colonne 'RUL' ajoutée
    """
    df = df.copy()
    df["max_cycle"] = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"]       = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    # Plafonnement : un RUL > clip n'apporte pas d'info utile
    df["RUL"] = df["RUL"].clip(upper=clip)

    print(f"[OK]  RUL calculé — min: {df['RUL'].min()}  max: {df['RUL'].max()}  "
          f"moyenne: {df['RUL'].mean():.1f}")
    return df


# ─────────────────────────────────────────────────────────────
#  2. Rolling Statistics (statistiques glissantes)
# ─────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame,
                         sensor_cols: list,
                         window: int = WINDOW_SIZE) -> pd.DataFrame:
    """
    Pour chaque capteur, ajoute :
      - mean_{w}  : moyenne glissante → tendance de dégradation
      - std_{w}   : écart-type glissant → détecte l'instabilité
      - min_{w}   : minimum glissant
      - max_{w}   : maximum glissant

    Args:
        df         : DataFrame trié par unit_id et cycle
        sensor_cols: liste des colonnes capteurs
        window     : taille de la fenêtre en cycles

    Returns:
        DataFrame enrichi
    """
    df = df.copy()
    grp = df.groupby("unit_id")

    for col in sensor_cols:
        base = grp[col]
        df[f"{col}_mean{window}"] = base.transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{col}_std{window}"]  = base.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f"{col}_min{window}"]  = base.transform(
            lambda x: x.rolling(window, min_periods=1).min())
        df[f"{col}_max{window}"]  = base.transform(
            lambda x: x.rolling(window, min_periods=1).max())

    added = len(sensor_cols) * 4
    print(f"[OK]  Rolling features ajoutées : {added} colonnes (window={window})")
    return df


# ─────────────────────────────────────────────────────────────
#  3. Lag Features (décalages temporels)
# ─────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     sensor_cols: list,
                     lags: list = LAG_STEPS) -> pd.DataFrame:
    """
    Ajoute les valeurs passées des capteurs (mémoire temporelle).
    Ex: sensor_2_lag5 = valeur du capteur 2 il y a 5 cycles

    Args:
        df         : DataFrame
        sensor_cols: liste des colonnes capteurs
        lags       : liste des décalages (ex: [1, 5, 10])

    Returns:
        DataFrame enrichi
    """
    df = df.copy()
    grp = df.groupby("unit_id")

    for col in sensor_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = grp[col].transform(
                lambda x, l=lag: x.shift(l)).fillna(method="bfill")

    added = len(sensor_cols) * len(lags)
    print(f"[OK]  Lag features ajoutées : {added} colonnes (lags={lags})")
    return df


# ─────────────────────────────────────────────────────────────
#  4. Features Dérivées
# ─────────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame,
                         sensor_cols: list) -> pd.DataFrame:
    """
    Ajoute des features physiques dérivées :
      - gradient      : taux de changement (dérivée discrète)
      - ratio_initial : dégradation relative vs valeur initiale
      - cycle_norm    : cycle normalisé entre 0 et 1 par équipement
    """
    df = df.copy()
    grp = df.groupby("unit_id")

    # Gradient (diff entre cycles consécutifs)
    for col in sensor_cols:
        df[f"{col}_grad"] = grp[col].transform(
            lambda x: x.diff().fillna(0))

    # Ratio par rapport à la valeur initiale (premier cycle)
    for col in sensor_cols:
        first_val = grp[col].transform("first").replace(0, np.nan)
        df[f"{col}_ratio"] = (df[col] / first_val).fillna(1.0)

    # Cycle normalisé (0 = début de vie, 1 = fin de vie)
    max_cycle = grp["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cycle

    # Features combinées spécifiques CMAPSS
    if "sensor_2" in df.columns and "sensor_3" in df.columns:
        df["temp_delta"] = df["sensor_3"] - df["sensor_2"]
    if "sensor_7" in df.columns and "sensor_12" in df.columns:
        df["pressure_ratio"] = df["sensor_7"] / df["sensor_12"].replace(0, np.nan)

    print(f"[OK]  Features dérivées ajoutées (gradient, ratio, cycle_norm)")
    return df


# ─────────────────────────────────────────────────────────────
#  5. Pipeline complet
# ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   compute_rul_flag: bool = True) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering.

    Args:
        df              : DataFrame nettoyé avec unit_id et cycle
        compute_rul_flag: True pour les données d'entraînement

    Returns:
        DataFrame prêt pour la modélisation
    """
    print("\n── Feature Engineering ─────────────────────────────")

    # Tri obligatoire
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    # Capteurs disponibles (après nettoyage)
    sensor_cols = [c for c in df.columns
                   if c.startswith("sensor_") and "_" not in c[7:]]

    # RUL
    if compute_rul_flag:
        df = compute_rul(df)

    # Features
    df = add_rolling_features(df, sensor_cols)
    df = add_lag_features(df, sensor_cols)
    df = add_derived_features(df, sensor_cols)

    # Nettoyage final des NaN résiduels
    df.fillna(0, inplace=True)

    print(f"[OK]  Shape finale : {df.shape}")
    print("────────────────────────────────────────────────────\n")
    return df


# ─────────────────────────────────────────────────────────────
#  6. Sélection des features pour le modèle
# ─────────────────────────────────────────────────────────────

def select_features(df: pd.DataFrame) -> list:
    """
    Retourne la liste des colonnes à utiliser comme features.
    Exclut les identifiants, le cycle brut et la cible RUL.
    """
    exclude = {"unit_id", "cycle", "RUL"}
    features = [c for c in df.columns if c not in exclude]
    print(f"[OK]  Nombre de features sélectionnées : {len(features)}")
    return features
