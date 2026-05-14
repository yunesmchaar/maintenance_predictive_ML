# 🏭 Maintenance Prédictive par Machine Learning

> **Prédiction de la Durée de Vie des Équipements Industriels**  
> Régression supervisée · XGBoost · SHAP · NASA CMAPSS

---

## 📋 Présentation du Projet

Ce projet implémente un système complet de **maintenance prédictive** capable de prédire
le **RUL (Remaining Useful Life)** d'équipements industriels à partir de données de capteurs.

| Paramètre | Valeur |
|---|---|
| Type de tâche | Régression supervisée |
| Dataset | NASA CMAPSS (Kaggle) |
| Modèle principal | XGBoost |
| Métrique cible | MAE < 15 cycles, R² > 0.90 |
| Langage | Python 3.10+ |

---

## 🗂️ Structure du Projet

```
predictive_maintenance/
│
├── main.py                    ← Point d'entrée principal
├── requirements.txt           ← Dépendances Python
├── README.md                  ← Cette documentation
│
├── src/                       ← Modules du projet
│   ├── config.py              ← Configuration globale
│   ├── data_loader.py         ← Chargement et nettoyage
│   ├── feature_engineering.py ← Création des variables
│   ├── model.py               ← Entraînement et évaluation
│   └── evaluation.py          ← Visualisations et SHAP
│
├── data/                      ← Données (à remplir)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── models/                    ← Modèles sauvegardés
│   ├── xgboost_rul_model.pkl
│   ├── xgboost_rul_scaler.pkl
│   └── xgboost_rul_features.pkl
│
└── outputs/                   ← Graphiques et résultats
    ├── 00_dashboard.png
    ├── 01_rul_distribution.png
    ├── 02_sensor_degradation.png
    ├── 03_correlation.png
    ├── 04_predictions.png
    ├── 05_model_comparison.png
    ├── 06_shap.png
    ├── predictions.csv
    ├── alerts.csv
    └── model_comparison.csv
```

---

## 🚀 Installation et Lancement

### Étape 1 — Cloner et installer les dépendances

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate          # Linux / Mac
venv\Scripts\activate             # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Étape 2 — Obtenir les données depuis Kaggle

**Option A — Via l'API Kaggle (recommandé) :**
```bash
# 1. Aller sur https://www.kaggle.com/settings → API → Create New Token
# 2. Placer le fichier kaggle.json téléchargé :
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Le script télécharge automatiquement au lancement
python main.py
```

**Option B — Téléchargement manuel :**
```
1. Aller sur : https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
2. Cliquer sur Download
3. Extraire et placer ces fichiers dans le dossier data/ :
   - train_FD001.txt
   - test_FD001.txt
   - RUL_FD001.txt
```

### Étape 3 — Lancer le projet

```bash
# Pipeline complet (recommandé)
python main.py

# Modes alternatifs
python main.py --mode train      # Entraînement seul
python main.py --mode predict    # Prédiction seule
python main.py --mode compare    # Comparer les modèles

# Changer de sous-ensemble (FD001 à FD004)
python main.py --subset FD002
```

---

## 📊 Pipeline ML — Étapes

```
Données brutes (capteurs)
        │
        ▼
  [1] Chargement & Nettoyage
      - Suppression capteurs constants
      - Gestion valeurs manquantes
        │
        ▼
  [2] Feature Engineering
      - Calcul du RUL
      - Rolling stats (mean, std, min, max) — window=10
      - Lag features (t-1, t-5, t-10)
      - Gradient, ratio initial, cycle normalisé
        │
        ▼
  [3] Analyse Exploratoire (EDA)
      - Distribution RUL
      - Dégradation des capteurs
      - Matrice de corrélation
        │
        ▼
  [4] Entraînement XGBoost
      - Split par équipement (80/20)
      - StandardScaler
      - Validation croisée temporelle (5 folds)
        │
        ▼
  [5] Comparaison des modèles
      - Ridge, Random Forest, GBM, XGBoost
        │
        ▼
  [6] Évaluation & Explicabilité
      - MAE, RMSE, R², NASA Score
      - SHAP values
      - Dashboard complet
        │
        ▼
  [7] Prédictions + Alertes
      - 🔴 RUL ≤ 10  → Intervention immédiate
      - 🟡 RUL ≤ 50  → Planifier maintenance
      - 🟢 RUL > 50  → Surveillance standard
```

---

## 🔧 Configuration

Toutes les options sont centralisées dans `src/config.py` :

```python
# Fenêtre glissante
WINDOW_SIZE  = 10        # cycles

# Plafond du RUL
RUL_CLIP     = 125       # cycles

# Seuils d'alerte
ALERT_CRITICAL = 10      # cycles
ALERT_WARNING  = 50      # cycles

# Hyperparamètres XGBoost
XGB_PARAMS = {
    "n_estimators"  : 400,
    "max_depth"     : 6,
    "learning_rate" : 0.04,
    ...
}
```

---

## 📈 Résultats Attendus

| Modèle | MAE | R² | Temps |
|---|---|---|---|
| Ridge (baseline) | ~45 cycles | 0.62 | < 1s |
| Random Forest | ~18 cycles | 0.87 | ~10s |
| Gradient Boosting | ~14 cycles | 0.91 | ~30s |
| **XGBoost** | **~11 cycles** | **0.94** | ~20s |

---

## 📦 Dépendances principales

| Package | Version | Rôle |
|---|---|---|
| pandas | 2.1.0 | Manipulation des données |
| numpy | 1.24.0 | Calculs numériques |
| scikit-learn | 1.3.0 | Pipeline ML |
| xgboost | 2.0.0 | Modèle principal |
| matplotlib | 3.7.0 | Visualisations |
| seaborn | 0.12.0 | Heatmaps |
| shap | 0.43.0 | Explicabilité |
| joblib | 1.3.0 | Sauvegarde modèle |
| kaggle | 1.5.16 | API téléchargement |

---

## 🧑‍💻 Auteur

Projet réalisé dans le cadre d'un cours de Machine Learning — 2026  
Dataset : NASA CMAPSS — Turbofan Engine Degradation Simulation
