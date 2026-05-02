"""
prediction_dashboard.py
-----------------------
Six-panel prediction dashboard — the hero README figure.

Combines all key results into one recruiter-facing visualisation:
  Panel 1: CV R² comparison — all four models
  Panel 2: Test set R² + RMSE + MAE table
  Panel 3: Predicted vs actual — best model (Random Forest)
  Panel 4: Top 10 feature importances
  Panel 5: Price distribution — actual vs predicted
  Panel 6: Pipeline summary metrics

Inputs : data/ (all processed arrays)
Outputs: results/prediction_dashboard.png  ← README hero image

Usage:
    python src/prediction_dashboard.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42
C = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]


def load_and_rebuild():
    X_tr   = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_te   = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_tr   = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_te   = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    fnames = np.load(os.path.join(DATA_DIR, "feature_names.npy"),
                     allow_pickle=True).tolist()

    with open(os.path.join(MODELS_DIR, "training_summary.json")) as f:
        summary = json.load(f)

    def get(key, param, default):
        v = summary[key]["params"].get(param, default)
        return None if v == "None" else type(default)(v)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=get("Ridge Regression","alpha",10.0)),
        "Lasso Regression":  Lasso(alpha=get("Lasso Regression","alpha",100.0),
                                   max_iter=10000),
        "Random Forest":     RandomForestRegressor(
            n_estimators=get("Random Forest","n_estimators",100),
            max_depth=get("Random Forest","max_depth",None),
            min_samples_split=get("Random Forest","min_samples_split",2),
            random_state=RANDOM_SEED),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_te)
        results[name] = {
            "y_pred": y_pred,
            "cv_r2":  summary[name]["cv_r2_mean"],
            "cv_std": summary[name]["cv_r2_std"],
            "R2":  r2_score(y_te, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_te, y_pred)),
            "MAE":  mean_absolute_error(y_te, y_pred),
        }

    rf = models["Random Forest"]
    return results, y_te, fnames, rf


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Generating prediction dashboard...")

    results, y_te, fnames, rf = load_and_rebuild()
    names = list(results.keys())

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "House Price Prediction Dashboard — Machine Learning Model Comparison\n"
        "Bharat Intern Machine Learning Internship  ·  Aug–Sep 2023  ·  "
        "Ames-style Housing Dataset (500 samples)",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, wspace=0.38, hspace=0.42)

    # ── Panel 1: CV R² comparison ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cv_means = [results[n]["cv_r2"] for n in names]
    cv_stds  = [results[n]["cv_std"] for n in names]
    bars = ax1.bar(range(len(names)), cv_means, color=C,
                   edgecolor="white", width=0.6)
    ax1.errorbar(range(len(names)), cv_means, yerr=cv_stds,
                 fmt="none", color="black", capsize=5, linewidth=1.2)
    ax1.bar_label(bars, fmt="%.3f", padding=4, fontsize=9, fontweight="bold")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=8)
    ax1.set_ylabel("CV R² (5-fold)", fontsize=10)
    ax1.set_title("Cross-Validation R²", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_facecolor("#FAFAFA")

    # ── Panel 2: Test metrics table ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    table_data = [[n,
                   f"{results[n]['R2']:.4f}",
                   f"${results[n]['RMSE']:,.0f}",
                   f"${results[n]['MAE']:,.0f}"]
                  for n in names]
    tbl = ax2.table(
        cellText=table_data,
        colLabels=["Model", "R²", "RMSE", "MAE"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#378ADD")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F4F8")
    ax2.set_title("Test Set Metrics", fontsize=11, fontweight="bold",
                  pad=60)

    # ── Panel 3: Predicted vs actual — Random Forest ──────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    y_pred_rf = results["Random Forest"]["y_pred"]
    ax3.scatter(y_te/1000, y_pred_rf/1000, alpha=0.55, s=20,
                color="#7F77DD", edgecolors="none")
    lims = [min(y_te.min(),y_pred_rf.min())/1000,
            max(y_te.max(),y_pred_rf.max())/1000]
    ax3.plot(lims, lims, "k--", linewidth=1.2, alpha=0.6)
    ax3.set_xlabel("Actual ($000s)", fontsize=10)
    ax3.set_ylabel("Predicted ($000s)", fontsize=10)
    ax3.set_title(f"Predicted vs Actual\nRandom Forest R²="
                  f"{results['Random Forest']['R2']:.4f}",
                  fontsize=11, fontweight="bold")
    ax3.set_facecolor("#FAFAFA")
    ax3.grid(True, alpha=0.2)

    # ── Panel 4: Feature importances — top 10 ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    imp = rf.feature_importances_
    top_idx  = np.argsort(imp)[-10:]
    top_imp  = imp[top_idx]
    top_feat = [fnames[i] for i in top_idx]
    ax4.barh(top_feat, top_imp, color="#1D9E75", edgecolor="white")
    ax4.set_xlabel("Importance", fontsize=10)
    ax4.set_title("Top 10 Feature Importances\n(Random Forest)",
                  fontsize=11, fontweight="bold")
    ax4.set_facecolor("#FAFAFA")
    ax4.grid(axis="x", alpha=0.25)

    # ── Panel 5: Price distribution — actual vs RF predicted ─────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(y_te/1000, bins=20, alpha=0.65, color="#378ADD",
             edgecolor="white", label="Actual")
    ax5.hist(y_pred_rf/1000, bins=20, alpha=0.65, color="#D85A30",
             edgecolor="white", label="RF Predicted")
    ax5.set_xlabel("Sale Price ($000s)", fontsize=10)
    ax5.set_ylabel("Count", fontsize=10)
    ax5.set_title("Price Distribution\nActual vs Predicted",
                  fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.set_facecolor("#FAFAFA")
    ax5.grid(axis="y", alpha=0.25)

    # ── Panel 6: Summary metrics ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    best = max(results, key=lambda k: results[k]["R2"])
    metrics = [
        ("Dataset",             "Ames-style Housing"),
        ("Samples",             "500 (400 train / 100 test)"),
        ("Features",            f"~35 (after encoding)"),
        ("Models compared",     "4"),
        ("Best model",          best),
        ("Best test R²",        f"{results[best]['R2']:.4f}"),
        ("Best RMSE",           f"${results[best]['RMSE']:,.0f}"),
        ("Best MAE",            f"${results[best]['MAE']:,.0f}"),
        ("CV folds",            "5"),
        ("Tuning",              "GridSearchCV"),
        ("Scaling",             "StandardScaler"),
        ("Internship",          "Bharat Intern ML · 2023"),
    ]
    for i, (label, value) in enumerate(metrics):
        y_pos = 0.95 - i * 0.077
        ax6.text(0.02, y_pos, label + ":", transform=ax6.transAxes,
                 fontsize=9, color="#555555")
        ax6.text(0.52, y_pos, value, transform=ax6.transAxes,
                 fontsize=9, fontweight="bold", color="#1A1A1A")
    ax6.set_title("Pipeline Summary", fontsize=11, fontweight="bold")
    ax6.add_patch(plt.Rectangle((0,0),1,1, fill=False,
                                 edgecolor="#CCCCCC", linewidth=1,
                                 transform=ax6.transAxes))

    out = os.path.join(RESULTS_DIR, "prediction_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    print("\nPrediction dashboard complete.")


if __name__ == "__main__":
    main()
