"""
data_preprocessing.py
---------------------
Data loading, cleaning, and feature engineering for the house price
prediction pipeline.

Steps:
  1. Load raw CSV dataset
  2. Encode categorical features (one-hot encoding)
  3. Engineer new features (house age, remodel age, quality-area interaction)
  4. Split into train/test sets (80/20, stratified by price quartile)
  5. Scale numerical features (StandardScaler)
  6. Save processed arrays and feature names

Inputs : data/house_prices.csv
Outputs: data/X_train.npy, X_test.npy, y_train.npy, y_test.npy
         data/feature_names.npy
         results/preprocessing_summary.png

Usage:
    python src/data_preprocessing.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(ROOT, "data", "house_prices.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

CATEGORICAL = ["Neighborhood", "BldgType", "HouseStyle", "ExterQual", "KitchenQual"]
TARGET      = "SalePrice"
RANDOM_SEED = 42


def load_and_engineer(path):
    df = pd.read_csv(path)

    # Feature engineering
    df["HouseAge"]    = 2023 - df["YearBuilt"]
    df["RemodelAge"]  = 2023 - df["YearRemodAdd"]
    df["QualArea"]    = df["OverallQual"] * df["GrLivArea"]
    df["TotalSF"]     = df["GrLivArea"] + df["TotalBsmtSF"]
    df["BathPerBed"]  = (df["FullBath"] + 0.5).div(df["BedroomAbvGr"].replace(0, 1))

    df.drop(columns=["YearBuilt", "YearRemodAdd"], inplace=True)
    return df


def encode_and_split(df):
    df_enc = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)

    X = df_enc.drop(columns=[TARGET]).values.astype(float)
    y = df_enc[TARGET].values.astype(float)
    feat_names = df_enc.drop(columns=[TARGET]).columns.tolist()

    # Stratified split by price quartile
    quartile = pd.qcut(y, 4, labels=False)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=quartile)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    return X_tr, X_te, y_tr, y_te, feat_names


def plot_summary(df, output_path):
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Dataset Preprocessing Summary — House Price Prediction",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, wspace=0.38, hspace=0.42)

    # Price distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df[TARGET]/1000, bins=35, color="#1A6BAD", edgecolor="white")
    ax1.set_xlabel("Sale Price ($000s)", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.set_title("Price Distribution", fontsize=11, fontweight="bold")
    ax1.set_facecolor("#FAFAFA")

    # Price vs living area
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(df["GrLivArea"], df[TARGET]/1000,
                     c=df["OverallQual"], cmap="RdYlGn",
                     alpha=0.6, s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax2, label="Overall Quality")
    ax2.set_xlabel("Above Ground Living Area (sq ft)", fontsize=10)
    ax2.set_ylabel("Sale Price ($000s)", fontsize=10)
    ax2.set_title("Price vs Living Area", fontsize=11, fontweight="bold")
    ax2.set_facecolor("#FAFAFA")

    # Price by quality
    ax3 = fig.add_subplot(gs[0, 2])
    qual_price = df.groupby("OverallQual")[TARGET].median() / 1000
    ax3.bar(qual_price.index, qual_price.values, color="#1D9E75", edgecolor="white")
    ax3.set_xlabel("Overall Quality (1–10)", fontsize=10)
    ax3.set_ylabel("Median Price ($000s)", fontsize=10)
    ax3.set_title("Price by Quality Rating", fontsize=11, fontweight="bold")
    ax3.set_facecolor("#FAFAFA")

    # Price by neighbourhood
    ax4 = fig.add_subplot(gs[1, 0:2])
    neigh_price = df.groupby("Neighborhood")[TARGET].median().sort_values() / 1000
    colours = ["#D85A30" if v > neigh_price.median() else "#378ADD"
               for v in neigh_price.values]
    ax4.barh(neigh_price.index, neigh_price.values, color=colours, edgecolor="white")
    ax4.set_xlabel("Median Sale Price ($000s)", fontsize=10)
    ax4.set_title("Median Price by Neighbourhood",
                  fontsize=11, fontweight="bold")
    ax4.set_facecolor("#FAFAFA")

    # Feature engineering summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    summary = [
        ("Samples",         "500"),
        ("Raw features",    "15"),
        ("Engineered",      "+ 5 new"),
        ("After encoding",  "~35 features"),
        ("Train / Test",    "400 / 100"),
        ("Scaling",         "StandardScaler"),
        ("Target",          "SalePrice"),
        ("Price range",     f"${df[TARGET].min()/1000:.0f}k–"
                            f"${df[TARGET].max()/1000:.0f}k"),
        ("Mean price",      f"${df[TARGET].mean()/1000:.0f}k"),
    ]
    for i, (label, value) in enumerate(summary):
        y_pos = 0.93 - i * 0.10
        ax5.text(0.02, y_pos, label + ":", transform=ax5.transAxes,
                 fontsize=9.5, color="#555555")
        ax5.text(0.55, y_pos, value, transform=ax5.transAxes,
                 fontsize=9.5, fontweight="bold", color="#1A1A1A")
    ax5.set_title("Preprocessing Summary", fontsize=11, fontweight="bold")
    ax5.add_patch(plt.Rectangle((0,0),1,1, fill=False,
                                 edgecolor="#CCCCCC", linewidth=1,
                                 transform=ax5.transAxes))

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_and_engineer(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Price range: ${df[TARGET].min():,.0f} – ${df[TARGET].max():,.0f}")
    print(f"  Mean price : ${df[TARGET].mean():,.0f}")
    print(f"  Engineered features added: HouseAge, RemodelAge, QualArea, TotalSF, BathPerBed")

    X_tr, X_te, y_tr, y_te, feat_names = encode_and_split(df)
    print(f"\n  Train set : {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
    print(f"  Test set  : {X_te.shape[0]} samples")

    # Save processed data
    data_dir = os.path.join(ROOT, "data")
    np.save(os.path.join(data_dir, "X_train.npy"), X_tr)
    np.save(os.path.join(data_dir, "X_test.npy"),  X_te)
    np.save(os.path.join(data_dir, "y_train.npy"), y_tr)
    np.save(os.path.join(data_dir, "y_test.npy"),  y_te)
    np.save(os.path.join(data_dir, "feature_names.npy"),
            np.array(feat_names))
    print(f"\n  Processed arrays saved to data/")

    plot_summary(df, os.path.join(RESULTS_DIR, "preprocessing_summary.png"))
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
