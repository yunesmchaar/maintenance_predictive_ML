# =============================================================
#  data_loader.py — Chargement et nettoyage des données
# =============================================================

import os
import pandas as pd
import numpy as np
from src.config import DATA_DIR, RAW_COLS, DROP_SENSORS


# ─────────────────────────────────────────────────────────────
#  1. Téléchargement via Kaggle API
# ─────────────────────────────────────────────────────────────

def download_from_kaggle(dataset: str = "behrad3d/nasa-cmaps"):
    """
    Télécharge le dataset NASA CMAPSS depuis Kaggle.
    Nécessite ~/.kaggle/kaggle.json configuré.

    Usage:
        download_from_kaggle()                        # NASA CMAPSS
        download_from_kaggle("shivamb/machine-predictive-maintenance-classification")
    """
    import subprocess
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[INFO] Téléchargement de : {dataset}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset,
         "--unzip", "-p", DATA_DIR],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("[OK]  Dataset téléchargé dans :", DATA_DIR)
    else:
        print("[ERREUR]", result.stderr)
        print("→ Placez manuellement train_FD001.txt dans le dossier data/")


# ─────────────────────────────────────────────────────────────
#  2. Chargement NASA CMAPSS (train + test)
# ─────────────────────────────────────────────────────────────

def load_cmapss(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge un sous-ensemble NASA CMAPSS.

    Args:
        subset: "FD001" | "FD002" | "FD003" | "FD004"

    Returns:
        (df_train, df_test, df_rul)
    """
    def _read(filename):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[ERREUR] Fichier introuvable : {path}\n"
                f"→ Téléchargez avec download_from_kaggle()\n"
                f"→ Ou placez {filename} dans le dossier data/"
            )
        df = pd.read_csv(path, sep=r"\s+", header=None,
                         names=RAW_COLS, engine="python")
        df.dropna(axis=1, how="all", inplace=True)
        return df

    df_train = _read(f"train_{subset}.txt")
    df_test  = _read(f"test_{subset}.txt")
    df_rul   = pd.read_csv(
        os.path.join(DATA_DIR, f"RUL_{subset}.txt"),
        header=None, names=["RUL_true"]
    )

    print(f"[OK]  Train : {df_train.shape}  |  Test : {df_test.shape}")
    return df_train, df_test, df_rul


# ─────────────────────────────────────────────────────────────
#  3. Chargement AI4I 2020 (alternative simple)
# ─────────────────────────────────────────────────────────────

def load_ai4i(filename: str = "ai4i2020.csv") -> pd.DataFrame:
    """
    Charge le dataset AI4I 2020 Predictive Maintenance.
    Colonnes: Type, Air temperature, Process temperature,
              Rotational speed, Torque, Tool wear, Machine failure, ...
    """
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)

    # Renommage unifié
    rename = {
        "Air temperature [K]"      : "air_temp",
        "Process temperature [K]"  : "proc_temp",
        "Rotational speed [rpm]"   : "rpm",
        "Torque [Nm]"              : "torque",
        "Tool wear [min]"          : "tool_wear",
        "Machine failure"          : "failure",
    }
    df.rename(columns=rename, inplace=True)

    # Encodage du type (L=0, M=1, H=2)
    type_map = {"L": 0, "M": 1, "H": 2}
    if "Type" in df.columns:
        df["type_enc"] = df["Type"].map(type_map)

    print(f"[OK]  AI4I chargé : {df.shape}  |  Pannes : {df['failure'].sum()} ({df['failure'].mean()*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────
#  4. Nettoyage commun
# ─────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, drop_sensors: list = DROP_SENSORS) -> pd.DataFrame:
    """
    Nettoyage :
      - Suppression des capteurs à variance nulle
      - Gestion des NaN
      - Suppression des doublons
    """
    df = df.copy()

    # Capteurs à supprimer
    cols_to_drop = [c for c in drop_sensors if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Remplissage NaN par médiane par capteur
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Doublons
    n_dup = df.duplicated().sum()
    if n_dup:
        df.drop_duplicates(inplace=True)
        print(f"[INFO] {n_dup} doublon(s) supprimé(s)")

    print(f"[OK]  Données nettoyées : {df.shape}")
    return df
