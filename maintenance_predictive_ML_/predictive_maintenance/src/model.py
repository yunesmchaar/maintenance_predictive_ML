# =============================================================
#  model.py — Entraînement, évaluation, sauvegarde du modèle
# =============================================================

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import XGB_PARAMS, TEST_SIZE, RANDOM_STATE, MODELS_DIR


# ─────────────────────────────────────────────────────────────
#  1. Préparation Train / Test
# ─────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame,
                 feature_cols: list,
                 target: str = "RUL"):
    """
    Divise le dataset en train/test en respectant la structure
    temporelle (split par unit_id, pas aléatoire).

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Split par équipement (80% des machines → train)
    units      = df["unit_id"].unique()
    n_train    = int(len(units) * (1 - TEST_SIZE))
    train_units = units[:n_train]
    test_units  = units[n_train:]

    train_df = df[df["unit_id"].isin(train_units)]
    test_df  = df[df["unit_id"].isin(test_units)]

    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target].values

    # Normalisation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[OK]  Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"      Équipements train : {len(train_units)}  |  test : {len(test_units)}")
    return X_train, X_test, y_train, y_test, scaler


# ─────────────────────────────────────────────────────────────
#  2. Validation Croisée Temporelle
# ─────────────────────────────────────────────────────────────

def cross_validate_model(model, X_train, y_train, n_splits: int = 5):
    """
    Validation croisée adaptée aux séries temporelles (TimeSeriesSplit).
    Évite le data leakage.
    """
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X_train, y_train,
                             cv=tscv, scoring="neg_mean_absolute_error",
                             n_jobs=-1)
    mae_scores = -scores
    print(f"\n── Validation Croisée ({n_splits} folds) ─────────────────")
    print(f"   MAE par fold : {[f'{s:.1f}' for s in mae_scores]}")
    print(f"   MAE moyenne  : {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
    print("────────────────────────────────────────────────────\n")
    return mae_scores


# ─────────────────────────────────────────────────────────────
#  3. Entraînement XGBoost (modèle principal)
# ─────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train,
                  X_val=None, y_val=None,
                  params: dict = XGB_PARAMS) -> XGBRegressor:
    """
    Entraîne XGBoost avec early stopping si validation fournie.
    """
    model = XGBRegressor(**params)

    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=30)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=50)
    else:
        model.fit(X_train, y_train)

    print("[OK]  XGBoost entraîné")
    return model


# ─────────────────────────────────────────────────────────────
#  4. Comparaison de plusieurs modèles
# ─────────────────────────────────────────────────────────────

def compare_models(X_train, X_test,
                   y_train, y_test) -> pd.DataFrame:
    """
    Compare 4 modèles : Ridge, Random Forest, Gradient Boosting, XGBoost.
    Retourne un DataFrame avec les métriques.
    """
    models = {
        "Ridge (baseline)" : Ridge(alpha=1.0),
        "Random Forest"    : RandomForestRegressor(
                                n_estimators=200, max_depth=8,
                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
                                n_estimators=200, max_depth=5,
                                learning_rate=0.05, random_state=RANDOM_STATE),
        "XGBoost"          : XGBRegressor(**XGB_PARAMS),
    }

    results = []
    print("\n── Comparaison des modèles ──────────────────────────")
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        results.append({"Modèle": name, "MAE": round(mae, 2),
                         "RMSE": round(rmse, 2), "R²": round(r2, 4)})
        print(f"   {name:25s} → MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    print("────────────────────────────────────────────────────\n")
    return pd.DataFrame(results).sort_values("MAE")


# ─────────────────────────────────────────────────────────────
#  5. Métriques détaillées
# ─────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test,
             model_name: str = "XGBoost") -> dict:
    """
    Calcule et affiche toutes les métriques de performance.
    """
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)   # RUL ne peut pas être négatif

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    # Score NASA (asymétrique : pénalise les prédictions trop optimistes)
    errors = preds - y_test
    nasa_score = np.sum(
        np.where(errors < 0,
                 np.exp(-errors / 13) - 1,
                 np.exp(errors  / 10) - 1)
    )

    # Précision dans une fenêtre de ±15 cycles
    within_15 = np.mean(np.abs(errors) <= 15) * 100

    metrics = {
        "MAE"       : round(mae, 2),
        "RMSE"      : round(rmse, 2),
        "R²"        : round(r2, 4),
        "NASA Score": round(nasa_score, 1),
        "Within ±15": round(within_15, 1),
    }

    print(f"\n── Résultats {model_name} ──────────────────────────────")
    for k, v in metrics.items():
        print(f"   {k:15s}: {v}")
    print("────────────────────────────────────────────────────\n")

    return metrics, preds


# ─────────────────────────────────────────────────────────────
#  6. Prédiction avec alerte
# ─────────────────────────────────────────────────────────────

def predict_with_alert(model, scaler,
                       X_new: np.ndarray,
                       unit_ids: list = None) -> pd.DataFrame:
    """
    Prédit le RUL et génère une alerte selon le seuil.

    Returns:
        DataFrame avec RUL prédit et niveau d'alerte
    """
    from src.config import ALERT_CRITICAL, ALERT_WARNING

    X_scaled = scaler.transform(X_new)
    rul_pred  = np.clip(model.predict(X_scaled), 0, None)

    def alert_level(rul):
        if rul <= ALERT_CRITICAL:
            return "🔴 CRITIQUE — Intervention immédiate"
        elif rul <= ALERT_WARNING:
            return "🟡 ALERTE   — Planifier maintenance"
        else:
            return "🟢 NORMAL   — Surveillance standard"

    results = pd.DataFrame({
        "unit_id"    : unit_ids if unit_ids else range(len(rul_pred)),
        "RUL_prédit" : rul_pred.round(0).astype(int),
        "Alerte"     : [alert_level(r) for r in rul_pred],
    })
    return results


# ─────────────────────────────────────────────────────────────
#  7. Sauvegarde / Chargement
# ─────────────────────────────────────────────────────────────

def save_model(model, scaler, feature_cols: list,
               name: str = "xgboost_rul"):
    """Sauvegarde le modèle, le scaler et la liste des features."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,        f"{MODELS_DIR}{name}_model.pkl")
    joblib.dump(scaler,       f"{MODELS_DIR}{name}_scaler.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}{name}_features.pkl")
    print(f"[OK]  Modèle sauvegardé dans {MODELS_DIR}")


def load_model(name: str = "xgboost_rul"):
    """Charge le modèle, le scaler et la liste des features."""
    model        = joblib.load(f"{MODELS_DIR}{name}_model.pkl")
    scaler       = joblib.load(f"{MODELS_DIR}{name}_scaler.pkl")
    feature_cols = joblib.load(f"{MODELS_DIR}{name}_features.pkl")
    print(f"[OK]  Modèle chargé depuis {MODELS_DIR}")
    return model, scaler, feature_cols
