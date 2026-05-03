# Methodology — House Price Prediction

## Overview

This project applies four supervised regression algorithms to predict
residential property sale prices from structural and locational features.
The pipeline covers data engineering, model training with cross-validation
tuning, test set evaluation, and feature importance analysis.

---

## Dataset

Based on the Ames Housing dataset structure (De Cock, 2011). 500 samples,
15 raw features covering property size, age, quality, neighbourhood, and
amenities. Target variable: SalePrice (USD).

### Feature engineering

Five additional features derived from raw inputs:

| Feature | Formula | Rationale |
|---|---|---|
| `HouseAge` | 2023 − YearBuilt | Age penalises older properties |
| `RemodelAge` | 2023 − YearRemodAdd | Recent remodels boost value |
| `QualArea` | OverallQual × GrLivArea | Interaction: quality × size |
| `TotalSF` | GrLivArea + TotalBsmtSF | Total liveable footprint |
| `BathPerBed` | (FullBath + 0.5) / BedroomAbvGr | Bathroom-to-bedroom ratio |

### Encoding and scaling

Categorical features (Neighborhood, BldgType, HouseStyle, ExterQual,
KitchenQual) encoded via one-hot encoding, producing 36 total features.
Numerical features scaled with StandardScaler (zero mean, unit variance)
before training to prevent scale-dependent bias in Ridge and Lasso.

---

## Models

### Linear Regression
Ordinary least squares baseline. No regularisation — establishes a
performance floor and provides interpretable coefficients.

### Ridge Regression (L2)
Adds L2 penalty: `loss = RSS + α·Σβ²`. Shrinks coefficients toward zero
without eliminating any. Effective when many features contribute weakly.
Best α selected via GridSearchCV from [0.1, 1, 10, 100, 1000].

### Lasso Regression (L1)
Adds L1 penalty: `loss = RSS + α·Σ|β|`. Performs automatic feature
selection by zeroing irrelevant coefficients. Best α from [1, 10, 100,
500, 1000].

### Random Forest
Ensemble of 200 decision trees (bagging + random feature subsets).
Captures non-linear relationships and feature interactions without
explicit engineering. Tuned via GridSearchCV on n_estimators, max_depth,
min_samples_split.

---

## Training and evaluation

**Cross-validation:** 5-fold CV on training set (400 samples) for all
models. Stratified split by price quartile ensures each fold covers the
full price range.

**Test set:** 100 held-out samples, never seen during training or tuning.

**Metrics:**
- R² — proportion of variance explained
- RMSE — penalises large errors more than small
- MAE — average absolute error in dollars
- MAPE — percentage error, scale-independent

---

## Key findings

- Linear, Ridge, and Lasso all achieve R² ≈ 0.94 on the test set,
  showing the price-feature relationship is largely linear
- Random Forest achieves lower R² (0.84) — ensemble methods benefit
  more from larger datasets
- Top features by correlation: GrLivArea (+0.70), OverallQual (+0.58)
- Lasso selects ~25 of 36 features — 11 zeroed as uninformative
- Mean prediction error: ±$9,600 (MAE) on mean house price of $261,000
  = 3.7% relative error

---

## References

- De Cock, D. (2011) Ames, Iowa: Alternative to the Boston Housing Data
  Set. Journal of Statistics Education, 19(3).
- Tibshirani, R. (1996) Regression Shrinkage and Selection via the Lasso.
  JRSS Series B, 58(1), 267–288.
- Scikit-learn Documentation — https://scikit-learn.org