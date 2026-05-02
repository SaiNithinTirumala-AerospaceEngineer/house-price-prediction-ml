"""
model_training.py
-----------------
Train four regression models on the processed housing dataset.

Models:
  1. Linear Regression  — baseline, interpretable coefficients
  2. Ridge Regression   — L2 regularisation, handles multicollinearity
  3. Lasso Regression   — L1 regularisation, automatic feature selection
  4. Random Forest      — ensemble, captures non-linear relationships

Each model is tuned via 5-fold cross-validation GridSearchCV.
Trained models are saved as .npy coefficient arrays or joblib files.

Inputs : data/X_train.npy, y_train.npy
Outputs: data/models/  (trained model parameters)
         results/training_cv_scores.png

Usage:
    python src/model_training.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42


def load_data():
    X_tr = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_tr = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    return X_tr, y_tr


def train_models(X_tr, y_tr):
    """Train and tune four models via cross-validation."""

    models = {}

    # 1 — Linear Regression (no hyperparameters)
    print("  [1/4] Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    scores_lr = cross_val_score(lr, X_tr, y_tr, cv=5,
                                scoring="r2", n_jobs=-1)
    models["Linear Regression"] = {
        "model": lr,
        "cv_r2_mean": scores_lr.mean(),
        "cv_r2_std":  scores_lr.std(),
        "params":     {}
    }
    print(f"     CV R² = {scores_lr.mean():.4f} ± {scores_lr.std():.4f}")

    # 2 — Ridge Regression
    print("  [2/4] Ridge Regression...")
    ridge_params = {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
    ridge_gs = GridSearchCV(Ridge(), ridge_params, cv=5,
                            scoring="r2", n_jobs=-1)
    ridge_gs.fit(X_tr, y_tr)
    scores_rr = cross_val_score(ridge_gs.best_estimator_, X_tr, y_tr,
                                cv=5, scoring="r2", n_jobs=-1)
    models["Ridge Regression"] = {
        "model": ridge_gs.best_estimator_,
        "cv_r2_mean": scores_rr.mean(),
        "cv_r2_std":  scores_rr.std(),
        "params":     ridge_gs.best_params_
    }
    print(f"     Best alpha={ridge_gs.best_params_['alpha']}  "
          f"CV R² = {scores_rr.mean():.4f} ± {scores_rr.std():.4f}")

    # 3 — Lasso Regression
    print("  [3/4] Lasso Regression...")
    lasso_params = {"alpha": [1.0, 10.0, 100.0, 500.0, 1000.0]}
    lasso_gs = GridSearchCV(Lasso(max_iter=10000), lasso_params,
                            cv=5, scoring="r2", n_jobs=-1)
    lasso_gs.fit(X_tr, y_tr)
    scores_la = cross_val_score(lasso_gs.best_estimator_, X_tr, y_tr,
                                cv=5, scoring="r2", n_jobs=-1)
    models["Lasso Regression"] = {
        "model": lasso_gs.best_estimator_,
        "cv_r2_mean": scores_la.mean(),
        "cv_r2_std":  scores_la.std(),
        "params":     lasso_gs.best_params_
    }
    print(f"     Best alpha={lasso_gs.best_params_['alpha']}  "
          f"CV R² = {scores_la.mean():.4f} ± {scores_la.std():.4f}")

    # 4 — Random Forest
    print("  [4/4] Random Forest...")
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth":    [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf_gs = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED),
        rf_params, cv=5, scoring="r2", n_jobs=-1)
    rf_gs.fit(X_tr, y_tr)
    scores_rf = cross_val_score(rf_gs.best_estimator_, X_tr, y_tr,
                                cv=5, scoring="r2", n_jobs=-1)
    models["Random Forest"] = {
        "model": rf_gs.best_estimator_,
        "cv_r2_mean": scores_rf.mean(),
        "cv_r2_std":  scores_rf.std(),
        "params":     rf_gs.best_params_
    }
    print(f"     Best {rf_gs.best_params_}  "
          f"CV R² = {scores_rf.mean():.4f} ± {scores_rf.std():.4f}")

    return models


def save_models(models):
    """Save model coefficients and metadata as JSON."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    summary = {}
    for name, info in models.items():
        summary[name] = {
            "cv_r2_mean": round(info["cv_r2_mean"], 4),
            "cv_r2_std":  round(info["cv_r2_std"],  4),
            "params":     {k: str(v) for k, v in info["params"].items()}
        }
        # Save coefficients for linear models
        model = info["model"]
        safe_name = name.lower().replace(" ", "_")
        if hasattr(model, "coef_"):
            np.save(os.path.join(MODELS_DIR, f"{safe_name}_coef.npy"),
                    model.coef_)
        if hasattr(model, "feature_importances_"):
            np.save(os.path.join(MODELS_DIR, f"{safe_name}_importances.npy"),
                    model.feature_importances_)

    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Model metadata saved to {MODELS_DIR}/")
    return summary


def plot_cv_scores(summary, output_path):
    """Bar chart of cross-validation R² scores with error bars."""
    names  = list(summary.keys())
    means  = [summary[n]["cv_r2_mean"] for n in names]
    stds   = [summary[n]["cv_r2_std"]  for n in names]
    colours = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means, color=colours, width=0.5,
                  edgecolor="white", linewidth=1)
    ax.errorbar(names, means, yerr=stds, fmt="none",
                color="black", capsize=6, linewidth=1.5)
    ax.bar_label(bars, fmt="%.4f", padding=5,
                 fontsize=11, fontweight="bold")

    ax.set_ylabel("Cross-Validation R² Score (5-fold)", fontsize=11)
    ax.set_title(
        "Model Comparison — 5-Fold Cross-Validation R² Score\n"
        "Bharat Intern Machine Learning Internship · Aug–Sep 2023",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, 1.08)
    ax.axhline(0.9, color="grey", linewidth=0.8, linestyle=":",
               alpha=0.6, label="R² = 0.90 target")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Model Training — House Price Prediction\n")

    X_tr, y_tr = load_data()
    print(f"  Training set: {X_tr.shape[0]} samples, "
          f"{X_tr.shape[1]} features\n")

    models  = train_models(X_tr, y_tr)
    summary = save_models(models)

    plot_cv_scores(summary,
                   os.path.join(RESULTS_DIR, "training_cv_scores.png"))

    best = max(summary, key=lambda k: summary[k]["cv_r2_mean"])
    print(f"\n  Best model : {best}")
    print(f"  Best CV R² : {summary[best]['cv_r2_mean']:.4f}")
    print("\nModel training complete.")


if __name__ == "__main__":
    main()
