"""
feature_analysis.py
-------------------
Feature importance and correlation analysis for the house price dataset.

Generates:
  1. Feature correlation heatmap
  2. Top 15 Random Forest feature importances
  3. Lasso coefficient plot (features selected vs zeroed)
  4. Partial dependence — price vs key features

Inputs : data/house_prices.csv
         data/X_train.npy, y_train.npy
         data/feature_names.npy
Outputs: results/feature_correlation.png
         results/feature_importance.png
         results/lasso_coefficients.png

Usage:
    python src/feature_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42


def load_data():
    X_tr   = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_tr   = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    fnames = np.load(os.path.join(DATA_DIR, "feature_names.npy"),
                     allow_pickle=True).tolist()
    df     = pd.read_csv(os.path.join(DATA_DIR, "house_prices.csv"))
    return X_tr, y_tr, fnames, df


def plot_correlation(df, output_path):
    """Correlation heatmap of numerical features vs sale price."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.4, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix\n"
                 "Numerical features — House Price Dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_feature_importance(X_tr, y_tr, fnames, output_path):
    """Top 15 Random Forest feature importances."""
    with open(os.path.join(MODELS_DIR, "training_summary.json")) as f:
        summary = json.load(f)
    rf_params = summary["Random Forest"]["params"]
    depth  = None if rf_params.get("max_depth","None")=="None" \
             else int(rf_params["max_depth"])
    n_est  = int(rf_params.get("n_estimators", 100))
    mss    = int(rf_params.get("min_samples_split", 2))

    rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                               min_samples_split=mss,
                               random_state=RANDOM_SEED)
    rf.fit(X_tr, y_tr)
    importances = rf.feature_importances_

    top_idx  = np.argsort(importances)[-15:]
    top_imp  = importances[top_idx]
    top_feat = [fnames[i] for i in top_idx]
    colours  = ["#D85A30" if i > importances.mean() else "#378ADD"
                for i in top_imp]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top_feat, top_imp, color=colours, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8.5,
                 fontweight="bold")
    ax.axvline(importances.mean(), color="grey", linewidth=1.0,
               linestyle="--", alpha=0.6,
               label=f"Mean importance ({importances.mean():.4f})")
    ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=11)
    ax.set_title("Top 15 Feature Importances — Random Forest\n"
                 "Darker bars exceed mean importance",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_lasso_coefficients(X_tr, y_tr, fnames, output_path):
    """Lasso coefficient bar chart — selected vs zeroed features."""
    with open(os.path.join(MODELS_DIR, "training_summary.json")) as f:
        summary = json.load(f)
    alpha = float(summary["Lasso Regression"]["params"].get("alpha", 100))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_tr, y_tr)
    coefs = lasso.coef_

    nonzero = [(fnames[i], coefs[i]) for i in range(len(coefs))
               if abs(coefs[i]) > 0]
    nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
    nonzero = nonzero[:20]

    names_nz = [n for n, _ in nonzero]
    vals_nz  = [v for _, v in nonzero]
    colours  = ["#1D9E75" if v > 0 else "#D85A30" for v in vals_nz]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names_nz, vals_nz, color=colours, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lasso Coefficient Value", fontsize=11)
    ax.set_title(
        f"Lasso Regression Coefficients (α={alpha:.0f})\n"
        f"Selected {len(nonzero)}/{len(coefs)} features — "
        f"{len(coefs)-len([c for c in coefs if c!=0])} zeroed out",
        fontsize=12, fontweight="bold"
    )
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="x", alpha=0.25)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1D9E75", label="Positive effect"),
                        Patch(color="#D85A30", label="Negative effect")],
              fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Feature Analysis — House Price Prediction\n")

    X_tr, y_tr, fnames, df = load_data()
    print(f"  Features: {len(fnames)}")
    print(f"  Training samples: {X_tr.shape[0]}\n")

    print("Generating plots...")
    plot_correlation(df,
        os.path.join(RESULTS_DIR, "feature_correlation.png"))
    plot_feature_importance(X_tr, y_tr, fnames,
        os.path.join(RESULTS_DIR, "feature_importance.png"))
    plot_lasso_coefficients(X_tr, y_tr, fnames,
        os.path.join(RESULTS_DIR, "lasso_coefficients.png"))

    # Print top correlations with price
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr_price = df[num_cols].corr()["SalePrice"].drop("SalePrice")
    print("\n  Top correlations with SalePrice:")
    for feat, val in corr_price.abs().sort_values(ascending=False).head(8).items():
        direction = "↑" if corr_price[feat] > 0 else "↓"
        print(f"    {feat:<20} {corr_price[feat]:>+.3f} {direction}")

    print("\nFeature analysis complete.")


if __name__ == "__main__":
    main()
