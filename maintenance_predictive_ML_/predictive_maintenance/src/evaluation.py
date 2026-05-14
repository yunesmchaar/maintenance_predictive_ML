# =============================================================
#  evaluation.py — Visualisations & Explicabilité SHAP
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.config import OUTPUTS_DIR, ALERT_CRITICAL, ALERT_WARNING

# Style global
plt.rcParams.update({
    "figure.facecolor" : "#0A1628",
    "axes.facecolor"   : "#112240",
    "axes.edgecolor"   : "#1B4F8A",
    "axes.labelcolor"  : "white",
    "axes.titlecolor"  : "white",
    "xtick.color"      : "white",
    "ytick.color"      : "white",
    "text.color"       : "white",
    "grid.color"       : "#1B4F8A",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.4,
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
})

TEAL   = "#00C2CB"
ACCENT = "#F4A261"
GREEN  = "#2ECC71"
RED    = "#E74C3C"
YELLOW = "#F39C12"
PURPLE = "#9B59B6"

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  1. Distribution du RUL
# ─────────────────────────────────────────────────────────────

def plot_rul_distribution(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribution du RUL", fontsize=16, fontweight="bold", color=TEAL)

    # Histogramme
    axes[0].hist(df["RUL"], bins=50, color=TEAL, alpha=0.8, edgecolor="black")
    axes[0].axvline(ALERT_CRITICAL, color=RED,    linestyle="--", linewidth=2, label=f"Critique ≤{ALERT_CRITICAL}")
    axes[0].axvline(ALERT_WARNING,  color=YELLOW, linestyle="--", linewidth=2, label=f"Alerte ≤{ALERT_WARNING}")
    axes[0].set_title("Histogramme du RUL", color="white")
    axes[0].set_xlabel("RUL (cycles)")
    axes[0].set_ylabel("Fréquence")
    axes[0].legend()
    axes[0].grid(True)

    # Boxplot par unité (échantillon de 20)
    sample_units = df["unit_id"].unique()[:20]
    sample_df    = df[df["unit_id"].isin(sample_units)]
    data_by_unit = [sample_df[sample_df["unit_id"] == u]["RUL"].values
                    for u in sample_units]
    bp = axes[1].boxplot(data_by_unit, patch_artist=True,
                         medianprops=dict(color=ACCENT, linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(TEAL)
        patch.set_alpha(0.6)
    axes[1].set_title("RUL par équipement (20 premiers)", color="white")
    axes[1].set_xlabel("Unit ID")
    axes[1].set_ylabel("RUL (cycles)")
    axes[1].grid(True, axis="y")

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUTS_DIR}01_rul_distribution.png", dpi=150, bbox_inches="tight")
        print(f"[OK]  Sauvegardé : {OUTPUTS_DIR}01_rul_distribution.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  2. Dégradation des capteurs dans le temps
# ─────────────────────────────────────────────────────────────

def plot_sensor_degradation(df: pd.DataFrame,
                            sensors: list = None,
                            n_units: int = 5,
                            save: bool = True):
    if sensors is None:
        sensors = ["sensor_2", "sensor_3", "sensor_4",
                   "sensor_7", "sensor_11", "sensor_12"]
    sensors = [s for s in sensors if s in df.columns][:6]

    units  = df["unit_id"].unique()[:n_units]
    colors = [TEAL, ACCENT, GREEN, YELLOW, PURPLE]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Dégradation des Capteurs au Fil du Temps",
                 fontsize=16, fontweight="bold", color=TEAL)
    axes = axes.flatten()

    for i, sensor in enumerate(sensors):
        ax = axes[i]
        for j, unit in enumerate(units):
            unit_df = df[df["unit_id"] == unit].sort_values("cycle")
            ax.plot(unit_df["cycle"], unit_df[sensor],
                    color=colors[j % len(colors)], alpha=0.8,
                    linewidth=1.5, label=f"Unit {unit}")
        ax.set_title(sensor.replace("_", " ").title(), color="white")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Valeur")
        ax.grid(True)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUTS_DIR}02_sensor_degradation.png", dpi=150, bbox_inches="tight")
        print(f"[OK]  Sauvegardé : {OUTPUTS_DIR}02_sensor_degradation.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  3. Corrélation des features avec le RUL
# ─────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame,
                             top_n: int = 20,
                             save: bool = True):
    exclude = {"unit_id", "cycle"}
    num_df  = df.select_dtypes(include=np.number).drop(
                  columns=[c for c in exclude if c in df.columns])

    corr_with_rul = num_df.corr()["RUL"].drop("RUL").abs().sort_values(ascending=False)
    top_features  = corr_with_rul.head(top_n).index.tolist()

    corr_matrix = num_df[top_features + ["RUL"]].corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Analyse de Corrélation", fontsize=16, fontweight="bold", color=TEAL)

    # Heatmap
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_matrix, mask=mask, ax=axes[0],
                cmap="coolwarm", center=0, annot=False,
                linewidths=0.3, cbar_kws={"shrink": 0.8})
    axes[0].set_title("Matrice de corrélation (Top features)", color="white")

    # Barplot corrélation avec RUL
    colors_bar = [GREEN if v > 0 else RED for v in corr_with_rul.head(top_n)]
    axes[1].barh(range(top_n), corr_with_rul.head(top_n),
                 color=colors_bar, alpha=0.85)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(corr_with_rul.head(top_n).index, fontsize=9)
    axes[1].set_xlabel("|Corrélation avec RUL|")
    axes[1].set_title(f"Top {top_n} features corrélées au RUL", color="white")
    axes[1].axvline(0.3, color=YELLOW, linestyle="--", alpha=0.7, label="Seuil 0.3")
    axes[1].legend()
    axes[1].grid(True, axis="x")

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUTS_DIR}03_correlation.png", dpi=150, bbox_inches="tight")
        print(f"[OK]  Sauvegardé : {OUTPUTS_DIR}03_correlation.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  4. Résultats du modèle : Prédit vs Réel
# ─────────────────────────────────────────────────────────────

def plot_predictions(y_test, preds,
                     metrics: dict = None,
                     save: bool = True):
    errors = preds - y_test

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Performance du Modèle XGBoost — Prédictions vs Réalité",
                 fontsize=15, fontweight="bold", color=TEAL)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Scatter prédit vs réel
    ax1 = fig.add_subplot(gs[0])
    sc  = ax1.scatter(y_test, preds, c=np.abs(errors),
                      cmap="coolwarm", alpha=0.5, s=10)
    lims = [0, max(y_test.max(), preds.max())]
    ax1.plot(lims, lims, "w--", linewidth=1.5, label="Parfait")
    ax1.fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims],
                     alpha=0.1, color=GREEN, label="±15 cycles")
    ax1.set_xlabel("RUL Réel (cycles)")
    ax1.set_ylabel("RUL Prédit (cycles)")
    ax1.set_title("Prédit vs Réel", color="white")
    ax1.legend(fontsize=9)
    ax1.grid(True)
    plt.colorbar(sc, ax=ax1, label="|Erreur|")

    # ── Distribution des erreurs
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(errors, bins=60, color=TEAL, alpha=0.8, edgecolor="black")
    ax2.axvline(0,  color="white",  linestyle="-",  linewidth=1.5)
    ax2.axvline(errors.mean(), color=ACCENT, linestyle="--",
                linewidth=2, label=f"Moyenne={errors.mean():.1f}")
    ax2.set_xlabel("Erreur (prédit − réel)")
    ax2.set_ylabel("Fréquence")
    ax2.set_title("Distribution des Erreurs", color="white")
    ax2.legend()
    ax2.grid(True)

    # ── Métriques
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    if metrics:
        table_data = [[k, str(v)] for k, v in metrics.items()]
        tbl = ax3.table(cellText=table_data,
                        colLabels=["Métrique", "Valeur"],
                        cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.2, 2.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#112240" if r > 0 else "#1B4F8A")
            cell.set_text_props(color="white")
            cell.set_edgecolor("#0A1628")
    ax3.set_title("Métriques de Performance", color="white", pad=15)

    if save:
        plt.savefig(f"{OUTPUTS_DIR}04_predictions.png", dpi=150, bbox_inches="tight")
        print(f"[OK]  Sauvegardé : {OUTPUTS_DIR}04_predictions.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  5. Comparaison des modèles
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparaison des Modèles ML",
                 fontsize=15, fontweight="bold", color=TEAL)

    colors = [GREEN if i == results_df["MAE"].idxmin() else TEAL
              for i in range(len(results_df))]

    for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
        vals = results_df[metric]
        bars = ax.bar(results_df["Modèle"], vals, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        ax.set_title(metric, color="white", fontsize=13)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * vals.max(),
                    f"{val:.3f}" if metric == "R²" else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9, color="white")

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUTS_DIR}05_model_comparison.png", dpi=150, bbox_inches="tight")
        print(f"[OK]  Sauvegardé : {OUTPUTS_DIR}05_model_comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  6. SHAP — Explicabilité
# ─────────────────────────────────────────────────────────────

def plot_shap(model, X_test_scaled: np.ndarray,
              feature_cols: list,
              max_display: int = 20,
              save: bool = True):
    try:
        import shap
        print("\n[INFO] Calcul des SHAP values (peut prendre ~30s)...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled[:500])

        # Importance globale
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("Explicabilité SHAP — Importance des Features",
                     fontsize=15, fontweight="bold", color=TEAL)

        # Bar plot importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_idx   = np.argsort(mean_shap)[::-1][:max_display]
        top_vals  = mean_shap[top_idx]
        top_names = [feature_cols[i] for i in top_idx]

        axes[0].barh(range(max_display), top_vals[::-1], color=TEAL, alpha=0.85)
        axes[0].set_yticks(range(max_display))
        axes[0].set_yticklabels(top_names[::-1], fontsize=9)
        axes[0].set_xlabel("Importance SHAP moyenne |φ|")
        axes[0].set_title(f"Top {max_display} features les plus importantes", color="white")
        axes[0].grid(True, axis="x")

        # Beeswarm (summary plot sans affichage natif SHAP)
        axes[1].scatter(
            shap_values[:, top_idx[0]],
            np.random.normal(0, 0.05, len(shap_values)),
            c=X_test_scaled[:500, top_idx[0]],
            cmap="coolwarm", alpha=0.4, s=8
        )
        axes[1].set_xlabel(f"Valeur SHAP — {top_names[0]}")
        axes[1].set_title(f"Impact de {top_names[0]} sur le RUL", color="white")
        axes[1].axvline(0, color="white", linestyle="--", alpha=0.7)
        axes[1].grid(True)

        plt.tight_layout()
        if save:
            plt.savefig(f"{OUTPUTS_DIR}06_shap.png", dpi=150, bbox_inches="tight")
            print(f"[OK]  Sauvegardé : {OUTPUTS_DIR}06_shap.png")
        plt.show()

        # Tableau récapitulatif SHAP
        shap_df = pd.DataFrame({
            "Feature"       : top_names,
            "Importance SHAP": top_vals.round(4),
        })
        shap_df.to_csv(f"{OUTPUTS_DIR}shap_importance.csv", index=False)
        print(f"[OK]  SHAP importance sauvegardée : {OUTPUTS_DIR}shap_importance.csv")
        return shap_df

    except ImportError:
        print("[WARN] SHAP non installé. Exécutez : pip install shap")
        return None


# ─────────────────────────────────────────────────────────────
#  7. Tableau de bord final — Dashboard récapitulatif
# ─────────────────────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, y_test, preds,
                   metrics: dict, results_df: pd.DataFrame,
                   save: bool = True):
    """
    Génère un dashboard complet en une seule figure.
    """
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0A1628")
    fig.suptitle("Dashboard — Maintenance Prédictive par Machine Learning",
                 fontsize=18, fontweight="bold", color=TEAL, y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.45, wspace=0.38)

    errors = preds - y_test

    # ── KPI cards (ligne 0) ──────────────────────────────────
    kpis = [
        ("MAE",        f"{metrics['MAE']} cycles",  TEAL),
        ("R²",         str(metrics["R²"]),           GREEN),
        ("RMSE",       f"{metrics['RMSE']} cycles",  ACCENT),
        ("Within ±15", f"{metrics['Within ±15']}%",  PURPLE),
    ]
    for i, (label, val, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#112240")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.55), 1, 0.45,
                                   facecolor=color, alpha=0.15))
        ax.text(0.5, 0.77, label, ha="center", va="center",
                fontsize=12, color=color, fontweight="bold")
        ax.text(0.5, 0.28, val,   ha="center", va="center",
                fontsize=20, color="white", fontweight="bold")

    # ── Distribution RUL (ligne 1, col 0-1) ─────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.hist(df["RUL"], bins=50, color=TEAL, alpha=0.75, edgecolor="#0A1628")
    ax2.axvline(ALERT_CRITICAL, color=RED,    linestyle="--", lw=2,
                label=f"Critique ≤{ALERT_CRITICAL}")
    ax2.axvline(ALERT_WARNING,  color=YELLOW, linestyle="--", lw=2,
                label=f"Alerte ≤{ALERT_WARNING}")
    ax2.set_title("Distribution du RUL", color="white")
    ax2.set_xlabel("RUL (cycles)"); ax2.set_ylabel("Fréquence")
    ax2.legend(fontsize=9); ax2.grid(True)

    # ── Prédit vs Réel (ligne 1, col 2-3) ───────────────────
    ax3 = fig.add_subplot(gs[1, 2:])
    lims = [0, max(float(y_test.max()), float(preds.max()))]
    ax3.scatter(y_test, preds, c=np.abs(errors),
                cmap="coolwarm", alpha=0.4, s=6)
    ax3.plot(lims, lims, "w--", lw=1.5, label="Idéal")
    ax3.fill_between(lims, [l-15 for l in lims], [l+15 for l in lims],
                     alpha=0.1, color=GREEN, label="±15 cycles")
    ax3.set_title("RUL Prédit vs Réel", color="white")
    ax3.set_xlabel("Réel"); ax3.set_ylabel("Prédit")
    ax3.legend(fontsize=9); ax3.grid(True)

    # ── Erreurs (ligne 2, col 0-1) ───────────────────────────
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.hist(errors, bins=60, color=ACCENT, alpha=0.8, edgecolor="#0A1628")
    ax4.axvline(0, color="white", lw=1.5)
    ax4.axvline(errors.mean(), color=RED, linestyle="--", lw=2,
                label=f"μ={errors.mean():.1f}")
    ax4.set_title("Distribution des Erreurs de Prédiction", color="white")
    ax4.set_xlabel("Erreur (cycles)"); ax4.set_ylabel("Fréquence")
    ax4.legend(); ax4.grid(True)

    # ── Comparaison modèles (ligne 2, col 2-3) ───────────────
    ax5 = fig.add_subplot(gs[2, 2:])
    bar_colors = [GREEN if v == results_df["MAE"].min() else TEAL
                  for v in results_df["MAE"]]
    bars = ax5.bar(results_df["Modèle"], results_df["MAE"],
                   color=bar_colors, alpha=0.85, edgecolor="black")
    ax5.set_title("Comparaison MAE par Modèle", color="white")
    ax5.set_ylabel("MAE (cycles)")
    ax5.tick_params(axis="x", rotation=15)
    ax5.grid(True, axis="y")
    for bar, val in zip(bars, results_df["MAE"]):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom",
                 fontsize=10, color="white")

    if save:
        path = f"{OUTPUTS_DIR}00_dashboard.png"
        plt.savefig(path, dpi=160, bbox_inches="tight",
                    facecolor="#0A1628")
        print(f"[OK]  Dashboard sauvegardé : {path}")
    plt.show()
