# =============================================================
#  main.py — Point d'entrée principal du projet
#  Maintenance Prédictive par Machine Learning
#
#  Usage:
#    python main.py                  # Pipeline complet
#    python main.py --mode train     # Entraînement seul
#    python main.py --mode predict   # Prédiction seule
#    python main.py --mode compare   # Comparer les modèles
# =============================================================

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ajouter le dossier racine au path
sys.path.insert(0, os.path.dirname(__file__))

from src.config      import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from src.data_loader import (download_from_kaggle, load_cmapss,
                              load_ai4i, clean)
from src.feature_engineering import (build_features, select_features)
from src.model       import (prepare_data, train_xgboost,
                              compare_models, evaluate,
                              predict_with_alert,
                              save_model, load_model,
                              cross_validate_model)
from src.evaluation  import (plot_rul_distribution,
                              plot_sensor_degradation,
                              plot_correlation_heatmap,
                              plot_predictions,
                              plot_model_comparison,
                              plot_shap,
                              plot_dashboard)


# ─────────────────────────────────────────────────────────────
#  BANNIÈRE
# ─────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   MAINTENANCE PRÉDICTIVE — MACHINE LEARNING                  ║
║   Prédiction du RUL (Remaining Useful Life)                  ║
║   Dataset : NASA CMAPSS  |  Modèle : XGBoost                 ║
╚══════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 1 — Chargement des données
# ─────────────────────────────────────────────────────────────

def step_load(subset: str = "FD001") -> pd.DataFrame:
    print("\n[ÉTAPE 1] Chargement des données...")

    os.makedirs(DATA_DIR,    exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Vérification de l'existence du fichier
    train_file = os.path.join(DATA_DIR, f"train_{subset}.txt")
    if not os.path.exists(train_file):
        print(f"[INFO] Fichier introuvable → Tentative de téléchargement Kaggle...")
        try:
            download_from_kaggle("behrad3d/nasa-cmaps")
        except Exception:
            print(f"""
[AIDE] Téléchargement automatique impossible.
       Veuillez faire l'une de ces 2 options :

       Option A — Kaggle API :
         1. Allez sur https://www.kaggle.com/settings → API → Create Token
         2. Placez kaggle.json dans ~/.kaggle/
         3. Relancez le script

       Option B — Manuel :
         1. Téléchargez depuis https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
         2. Placez train_FD001.txt, test_FD001.txt, RUL_FD001.txt dans : {DATA_DIR}
         3. Relancez le script
""")
            sys.exit(1)

    df_train, df_test, df_rul = load_cmapss(subset)
    df_train = clean(df_train)
    df_test  = clean(df_test)
    return df_train, df_test, df_rul


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 2 — Feature Engineering
# ─────────────────────────────────────────────────────────────

def step_features(df_train: pd.DataFrame,
                  df_test:  pd.DataFrame) -> tuple:
    print("\n[ÉTAPE 2] Feature Engineering...")

    # Calcul RUL pour le train
    df_train_fe = build_features(df_train, compute_rul_flag=True)

    # Pour le test : pas de RUL connu (on le calculera via df_rul)
    df_test_fe  = build_features(df_test,  compute_rul_flag=False)

    feature_cols = select_features(df_train_fe)
    return df_train_fe, df_test_fe, feature_cols


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 3 — Visualisation exploratoire (EDA)
# ─────────────────────────────────────────────────────────────

def step_eda(df: pd.DataFrame):
    print("\n[ÉTAPE 3] Analyse Exploratoire (EDA)...")

    print(f"\n{'─'*50}")
    print(f"  Nombre d'équipements : {df['unit_id'].nunique()}")
    print(f"  Nombre de cycles total: {len(df)}")
    print(f"  RUL — min: {df['RUL'].min()}  max: {df['RUL'].max()}  "
          f"moy: {df['RUL'].mean():.1f}")
    print(f"  Capteurs disponibles : "
          f"{len([c for c in df.columns if c.startswith('sensor_') and '_' not in c[7:]])}")
    print(f"{'─'*50}\n")

    plot_rul_distribution(df)
    plot_sensor_degradation(df)
    plot_correlation_heatmap(df)


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 4 — Entraînement et évaluation
# ─────────────────────────────────────────────────────────────

def step_train(df: pd.DataFrame,
               feature_cols: list) -> tuple:
    print("\n[ÉTAPE 4] Entraînement du modèle...")

    X_train, X_test, y_train, y_test, scaler = prepare_data(
        df, feature_cols)

    # Validation croisée temporelle
    from xgboost import XGBRegressor
    from src.config import XGB_PARAMS
    xgb_tmp = XGBRegressor(**XGB_PARAMS)
    cross_validate_model(xgb_tmp, X_train, y_train)

    # Entraînement final
    model = train_xgboost(X_train, y_train)

    # Sauvegarde
    save_model(model, scaler, feature_cols)

    return model, scaler, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 5 — Comparaison des modèles
# ─────────────────────────────────────────────────────────────

def step_compare(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    print("\n[ÉTAPE 5] Comparaison des modèles...")

    results_df = compare_models(X_train, X_test, y_train, y_test)

    print("\n── Classement final ────────────────────────────────")
    print(results_df.to_string(index=False))
    print("────────────────────────────────────────────────────")

    plot_model_comparison(results_df)
    results_df.to_csv(f"{OUTPUTS_DIR}model_comparison.csv", index=False)
    return results_df


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 6 — Résultats et visualisations finales
# ─────────────────────────────────────────────────────────────

def step_evaluate(model, X_test, y_test,
                  feature_cols, results_df,
                  df_train):
    print("\n[ÉTAPE 6] Évaluation finale et visualisations...")

    metrics, preds = evaluate(model, X_test, y_test)

    plot_predictions(y_test, preds, metrics)
    plot_shap(model, X_test, feature_cols)
    plot_dashboard(df_train, y_test, preds, metrics, results_df)

    # Export résultats
    pd.DataFrame({
        "RUL_réel"  : y_test,
        "RUL_prédit": preds.round(0).astype(int),
        "erreur"    : (preds - y_test).round(1),
    }).to_csv(f"{OUTPUTS_DIR}predictions.csv", index=False)
    print(f"[OK]  Prédictions exportées : {OUTPUTS_DIR}predictions.csv")

    return metrics, preds


# ─────────────────────────────────────────────────────────────
#  ÉTAPE 7 — Prédiction sur nouvelles données
# ─────────────────────────────────────────────────────────────

def step_predict_new(model, scaler, feature_cols: list,
                     df_test_fe: pd.DataFrame,
                     df_rul: pd.DataFrame):
    """
    Prédit le RUL pour les équipements du jeu de test
    et compare avec les vraies valeurs RUL.
    """
    print("\n[ÉTAPE 7] Prédiction sur nouvelles données (jeu de test)...")

    # Pour chaque équipement, prendre le dernier cycle
    last_cycles = (df_test_fe
                   .sort_values(["unit_id", "cycle"])
                   .groupby("unit_id")
                   .last()
                   .reset_index())

    available = [c for c in feature_cols if c in last_cycles.columns]
    missing   = set(feature_cols) - set(available)
    if missing:
        for m in missing:
            last_cycles[m] = 0.0

    X_new    = last_cycles[feature_cols].values
    unit_ids = last_cycles["unit_id"].tolist()

    results = predict_with_alert(model, scaler, X_new, unit_ids)

    # Ajout du RUL réel si disponible
    if df_rul is not None and len(df_rul) == len(results):
        results["RUL_réel"] = df_rul["RUL_true"].values

    print(f"\n── Prédictions (10 premiers équipements) ───────────")
    print(results.head(10).to_string(index=False))
    print("────────────────────────────────────────────────────")

    # Résumé des alertes
    n_critique = results["Alerte"].str.contains("CRITIQUE").sum()
    n_alerte   = results["Alerte"].str.contains("ALERTE").sum()
    n_normal   = results["Alerte"].str.contains("NORMAL").sum()

    print(f"""
┌─────────────────────────────────────────┐
│  RÉSUMÉ DES ALERTES                     │
│  🔴 Critique (intervention immédiate) : {n_critique:3d} │
│  🟡 Alerte   (planifier maintenance)  : {n_alerte:3d} │
│  🟢 Normal   (surveillance standard) : {n_normal:3d} │
└─────────────────────────────────────────┘
""")

    results.to_csv(f"{OUTPUTS_DIR}alerts.csv", index=False)
    print(f"[OK]  Alertes exportées : {OUTPUTS_DIR}alerts.csv")
    return results


# ─────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run_pipeline(subset: str = "FD001"):
    print(BANNER)

    # ── Chargement
    df_train, df_test, df_rul = step_load(subset)

    # ── Feature Engineering
    df_train_fe, df_test_fe, feature_cols = step_features(df_train, df_test)

    # ── EDA
    step_eda(df_train_fe)

    # ── Entraînement
    model, scaler, X_train, X_test, y_train, y_test = step_train(
        df_train_fe, feature_cols)

    # ── Comparaison
    results_df = step_compare(X_train, X_test, y_train, y_test)

    # ── Évaluation
    metrics, preds = step_evaluate(
        model, X_test, y_test, feature_cols, results_df, df_train_fe)

    # ── Prédictions nouvelles données
    step_predict_new(model, scaler, feature_cols, df_test_fe, df_rul)

    # ── Résumé final
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  PROJET TERMINÉ AVEC SUCCÈS                                  ║
╠══════════════════════════════════════════════════════════════╣
║  MAE       : {str(metrics['MAE']):>8} cycles                          ║
║  RMSE      : {str(metrics['RMSE']):>8} cycles                          ║
║  R²        : {str(metrics['R²']):>8}                                  ║
║  Within±15 : {str(metrics['Within ±15']):>7}%                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Fichiers générés dans : {OUTPUTS_DIR:<36}║
╚══════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────
#  MODES ALTERNATIFS
# ─────────────────────────────────────────────────────────────

def run_train_only(subset: str = "FD001"):
    """Entraîne et sauvegarde le modèle uniquement."""
    print(BANNER)
    df_train, _, _ = step_load(subset)
    df_train_fe, _, feature_cols = step_features(df_train,
                                                  df_train.copy())
    step_train(df_train_fe, feature_cols)
    print("\n[OK]  Modèle entraîné et sauvegardé.")


def run_predict_only():
    """Charge un modèle existant et génère des prédictions."""
    print(BANNER)
    model, scaler, feature_cols = load_model()
    _, df_test, df_rul = step_load()
    _, df_test_fe, _ = step_features(pd.DataFrame(), df_test)
    step_predict_new(model, scaler, feature_cols, df_test_fe, df_rul)


def run_compare_only(subset: str = "FD001"):
    """Compare plusieurs modèles."""
    print(BANNER)
    df_train, _, _ = step_load(subset)
    df_train_fe, _, feature_cols = step_features(df_train, df_train.copy())
    from src.model import prepare_data
    X_train, X_test, y_train, y_test, _ = prepare_data(df_train_fe,
                                                         feature_cols)
    results_df = step_compare(X_train, X_test, y_train, y_test)
    print(results_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Maintenance Prédictive — Pipeline ML complet")
    parser.add_argument("--mode", type=str, default="full",
        choices=["full", "train", "predict", "compare"],
        help="Mode d'exécution (default: full)")
    parser.add_argument("--subset", type=str, default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Sous-ensemble NASA CMAPSS (default: FD001)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "full":
        run_pipeline(args.subset)
    elif args.mode == "train":
        run_train_only(args.subset)
    elif args.mode == "predict":
        run_predict_only()
    elif args.mode == "compare":
        run_compare_only(args.subset)
