"""
model_evaluation.py
-------------------
Evaluate all four trained models on the held-out test set.

Metrics computed per model:
  - R² score          (coefficient of determination)
  - RMSE              (root mean squared error)
  - MAE               (mean absolute error)
  - MAPE              (mean absolute percentage error)

Generates:
  1. Metrics comparison table plot
  2. Predicted vs actual scatter plots (4-panel)
  3. Residual distribution plots

Inputs : data/X_train.npy, X_test.npy, y_train.npy, y_test.npy
Outputs: results/evaluation_metrics.png
         results/predicted_vs_actual.png
         results/residual_analysis.png

Usage:
    python src/model_evaluation.py
"""

import os
import json
import numpy as np
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


def load_data():
    return (np.load(os.path.join(DATA_DIR, "X_train.npy")),
            np.load(os.path.join(DATA_DIR, "X_test.npy")),
            np.load(os.path.join(DATA_DIR, "y_train.npy")),
            np.load(os.path.join(DATA_DIR, "y_test.npy")))


def rebuild_models(X_tr, y_tr):
    """Re-fit models with best parameters from training summary."""
    with open(os.path.join(MODELS_DIR, "training_summary.json")) as f:
        summary = json.load(f)

    lr_alpha    = float(summary["Ridge Regression"]["params"].get("alpha", 10))
    la_alpha    = float(summary["Lasso Regression"]["params"].get("alpha", 100))
    rf_depth    = summary["Random Forest"]["params"].get("max_depth", "None")
    rf_depth    = None if rf_depth == "None" else int(rf_depth)
    rf_n        = int(summary["Random Forest"]["params"].get("n_estimators", 100))
    rf_mss      = int(summary["Random Forest"]["params"].get("min_samples_split", 2))

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=lr_alpha),
        "Lasso Regression":  Lasso(alpha=la_alpha, max_iter=10000),
        "Random Forest":     RandomForestRegressor(
                                n_estimators=rf_n, max_depth=rf_depth,
                                min_samples_split=rf_mss,
                                random_state=RANDOM_SEED),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)
    return models


def compute_metrics(models, X_te, y_te):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_te)
        r2   = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)
        mape = np.mean(np.abs((y_te - y_pred) / y_te)) * 100
        results[name] = {
            "y_pred": y_pred,
            "R2":   round(r2,   4),
            "RMSE": round(rmse, 2),
            "MAE":  round(mae,  2),
            "MAPE": round(mape, 2),
        }
        print(f"  {name:<22} R²={r2:.4f}  RMSE=${rmse:,.0f}"
              f"  MAE=${mae:,.0f}  MAPE={mape:.1f}%")
    return results


def plot_metrics(results, output_path):
    names   = list(results.keys())
    metrics = ["R2", "RMSE", "MAE", "MAPE"]
    titles  = ["R² Score", "RMSE ($)", "MAE ($)", "MAPE (%)"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Model Evaluation Metrics — Test Set (100 samples)",
                 fontsize=12, fontweight="bold")

    for ax, metric, title, colour in zip(axes, metrics, titles, C):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colour,
                      edgecolor="white", width=0.55)
        ax.bar_label(bars,
                     labels=[f"{v:.4f}" if metric=="R2"
                             else f"{v:,.0f}" if metric in ["RMSE","MAE"]
                             else f"{v:.1f}%" for v in vals],
                     padding=4, fontsize=8.5, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_predicted_vs_actual(results, y_te, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Predicted vs Actual Sale Price — Test Set",
                 fontsize=13, fontweight="bold")

    for ax, (name, res), colour in zip(
            axes.flat, results.items(), C):
        y_pred = res["y_pred"]
        ax.scatter(y_te/1000, y_pred/1000,
                   alpha=0.55, s=25, color=colour, edgecolors="none")
        lims = [min(y_te.min(), y_pred.min())/1000,
                max(y_te.max(), y_pred.max())/1000]
        ax.plot(lims, lims, "k--", linewidth=1.2, alpha=0.6,
                label="Perfect prediction")
        ax.set_xlabel("Actual Price ($000s)", fontsize=10)
        ax.set_ylabel("Predicted Price ($000s)", fontsize=10)
        ax.set_title(f"{name}\nR²={res['R2']:.4f}  "
                     f"RMSE=${res['RMSE']:,.0f}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_residuals(results, y_te, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Residual Analysis — Test Set",
                 fontsize=13, fontweight="bold")

    for ax, (name, res), colour in zip(axes.flat, results.items(), C):
        residuals = (y_te - res["y_pred"]) / 1000
        ax.hist(residuals, bins=25, color=colour,
                edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.axvline(residuals.mean(), color="red", linewidth=1.0,
                   linestyle=":", label=f"Mean={residuals.mean():.1f}k")
        ax.set_xlabel("Residual ($000s)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{name}  σ=${residuals.std():.1f}k",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Model Evaluation — House Price Prediction\n")

    X_tr, X_te, y_tr, y_te = load_data()
    models  = rebuild_models(X_tr, y_tr)
    results = compute_metrics(models, X_te, y_te)

    best = max(results, key=lambda k: results[k]["R2"])
    print(f"\n  Best model on test set : {best}")
    print(f"  Test R²                : {results[best]['R2']:.4f}")

    print("\nGenerating plots...")
    plot_metrics(results,
                 os.path.join(RESULTS_DIR, "evaluation_metrics.png"))
    plot_predicted_vs_actual(
        results, y_te,
        os.path.join(RESULTS_DIR, "predicted_vs_actual.png"))
    plot_residuals(
        results, y_te,
        os.path.join(RESULTS_DIR, "residual_analysis.png"))

    print("\nModel evaluation complete.")


if __name__ == "__main__":
    main()
