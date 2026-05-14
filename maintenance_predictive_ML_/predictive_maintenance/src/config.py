# =============================================================
#  config.py — Configuration centrale du projet
# =============================================================

# ── Chemins ──────────────────────────────────────────────────
DATA_DIR    = "data/"
MODELS_DIR  = "models/"
OUTPUTS_DIR = "outputs/"

# ── Dataset NASA CMAPSS ──────────────────────────────────────
# Colonnes brutes du fichier .txt
RAW_COLS = (
    ["unit_id", "cycle"]
    + [f"op_{i}"     for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Capteurs constants (à supprimer — variance nulle)
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6",
                "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

# ── Feature Engineering ───────────────────────────────────────
WINDOW_SIZE  = 10        # fenêtre glissante (cycles)
LAG_STEPS    = [1, 5, 10]  # décalages temporels
RUL_CLIP     = 125       # plafond du RUL (cycles)

# ── Modèle XGBoost ────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators"  : 400,
    "max_depth"     : 6,
    "learning_rate" : 0.04,
    "subsample"     : 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha"     : 0.1,
    "reg_lambda"    : 1.0,
    "random_state"  : 42,
    "n_jobs"        : -1,
}

# ── Évaluation ────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Seuils d'alerte ───────────────────────────────────────────
ALERT_CRITICAL = 10    # cycles → intervention immédiate
ALERT_WARNING  = 50    # cycles → planifier maintenance
