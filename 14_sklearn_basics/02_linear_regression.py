# Revision Notes:
# Topic: Linear Regression with scikit-learn
# Why it matters for AI/ML: Linear regression is the foundation for understanding supervised learning.
# Most ML concepts extend from linear models (coefficients, predictions, residuals, R²).
# Understanding linear regression builds intuition for more complex algorithms.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

np.random.seed(42)

# ---
# SIMPLE LINEAR REGRESSION: y = mx + b
# ---

# WHY: Predict continuous target from single feature.

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Must be 2D
y = np.array([2, 4, 5, 4, 5])  # Target with noise

print("=== Simple Linear Regression ===")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Create and train model
# WHY: LinearRegression finds optimal line minimizing residual sum of squares.
model = LinearRegression()
model.fit(X, y)

# Model parameters
# WHY: Coefficients tell us feature importance; intercept is baseline.
print(f"Coefficient (slope): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Equation: y = {model.coef_[0]:.4f}*x + {model.intercept_:.4f}")

# Predictions
# WHY: Apply learned model to new data.
y_pred = model.predict(X)
print(f"\nPredictions: {y_pred.round(2)}")
print(f"Actual: {y}")

# Performance metrics
# WHY: Quantify how well model fits.
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ---
# MULTIPLE LINEAR REGRESSION
# ---

# WHY: Predict from multiple features.
# y = b0 + b1*x1 + b2*x2 + ... + bn*xn

from sklearn.datasets import make_regression

X_multi, y_multi = make_regression(n_samples=100, n_features=5, n_informative=3,
                                    noise=10, random_state=42)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Train model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Coefficients for each feature
# WHY: Positive coefficient = feature increases output; negative = decreases.
print("\n" + "="*60)
print("\n=== Multiple Linear Regression ===")
print(f"Coefficients for 5 features: {model_multi.coef_.round(4)}")
print(f"Intercept: {model_multi.intercept_:.4f}")

# Feature importance from coefficients
# WHY: Larger magnitude coef = more important feature (if scaled).
feature_importance = np.abs(model_multi.coef_)
top_features = np.argsort(feature_importance)[::-1]
print(f"\nTop 3 most important features (by coefficient magnitude):")
for feat_idx in top_features[:3]:
    print(f"  Feature {feat_idx}: coef = {model_multi.coef_[feat_idx]:.4f}")

# Predictions on test set
y_pred_test = model_multi.predict(X_test)
r2_train = model_multi.score(X_train, y_train)
r2_test = model_multi.score(X_test, y_test)

print(f"\nTrain R²: {r2_train:.4f}")
print(f"Test R²: {r2_test:.4f}")
if r2_train > r2_test + 0.1:
    print("→ Potential overfitting (train R² much higher than test)")

# ---
# FEATURE SCALING IMPORTANCE
# ---

# WHY: Scale features so coefficients are comparable; algorithms converge faster.

from sklearn.preprocessing import StandardScaler

print("\n" + "="*60)
print("\n=== Feature Scaling Impact ===")

# Data with different scales
X_unscaled = np.array([[1000, 0.5],
                       [2000, 1.0],
                       [3000, 1.5]]).astype(float)
y_scale = np.array([50, 100, 150])

# Without scaling
model_unscaled = LinearRegression()
model_unscaled.fit(X_unscaled, y_scale)
print(f"Unscaled coefficients: {model_unscaled.coef_}")
print("First feature coefficient is small because feature has large values")

# With scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y_scale)
print(f"\nScaled coefficients: {model_scaled.coef_}")
print("Coefficients now on same scale; easier to compare importance")

# ---
# REGULARIZATION: RIDGE AND LASSO
# ---

# WHY: Prevent overfitting by penalizing large coefficients.

from sklearn.linear_model import Ridge, Lasso

X_reg, y_reg = make_regression(n_samples=100, n_features=20, n_informative=5,
                                noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("\n" + "="*60)
print("\n=== Regularization: Ridge vs Lasso ===")

# Linear Regression (no regularization)
model_linear = LinearRegression()
model_linear.fit(X_train_reg, y_train_reg)
train_linear = model_linear.score(X_train_reg, y_train_reg)
test_linear = model_linear.score(X_test_reg, y_test_reg)

# Ridge Regression (L2 penalty)
# WHY: Penalizes large coefficients; shrinks toward zero but doesn't eliminate.
model_ridge = Ridge(alpha=1.0)  # alpha controls strength
model_ridge.fit(X_train_reg, y_train_reg)
train_ridge = model_ridge.score(X_train_reg, y_train_reg)
test_ridge = model_ridge.score(X_test_reg, y_test_reg)

# Lasso Regression (L1 penalty)
# WHY: Can shrink coefficients to exactly zero (feature selection).
model_lasso = Lasso(alpha=0.5)
model_lasso.fit(X_train_reg, y_train_reg)
train_lasso = model_lasso.score(X_train_reg, y_train_reg)
test_lasso = model_lasso.score(X_test_reg, y_test_reg)

print(f"Linear:  Train R² = {train_linear:.4f}, Test R² = {test_linear:.4f}")
print(f"Ridge:   Train R² = {train_ridge:.4f}, Test R² = {test_ridge:.4f}")
print(f"Lasso:   Train R² = {train_lasso:.4f}, Test R² = {test_lasso:.4f}")

print(f"\nLasso selected {(model_lasso.coef_ != 0).sum()} out of 20 features")

# ---
# ASSUMPTIONS OF LINEAR REGRESSION
# ---

# WHY: Understand when linear regression works well.

print("\n" + "="*60)
print("\n=== Linear Regression Assumptions ===")

# Generate data with assumptions  violated
X_nonlinear = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y_nonlinear = np.sin(X_nonlinear).ravel() + np.random.randn(100) * 0.1

model_nonlinear = LinearRegression()
model_nonlinear.fit(X_nonlinear, y_nonlinear)
y_pred_nonlinear = model_nonlinear.predict(X_nonlinear)
r2_nonlinear = r2_score(y_nonlinear, y_pred_nonlinear)

print("1. Linearity: relationship must be linear")
print(f"   Sine wave data with linear model: R² = {r2_nonlinear:.4f} (poor!)")

# Residuals should be normally distributed
residuals = y_nonlinear - y_pred_nonlinear
print(f"\n2. Normality: residuals should be ~ normal")
print(f"   Residuals mean: {residuals.mean():.4f} (should be ≈ 0)")
print(f"   Residuals std: {residuals.std():.4f}")

# Homoscedasticity: constant residual variance
print(f"\n3. Homoscedasticity: constant variance across predictions")
print(f"   Residual variance: {np.var(residuals):.4f}")

# Independence: residuals uncorrelated
print(f"\n4. Independence: observations should be independent")
print("   Check: time series data, spatial clustering violates this")

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: House price prediction
# WHY: Classic regression task.
prices = np.random.normal(300000, 50000, 100)
sqft = np.random.normal(2000, 500, 100)
X_house = sqft.reshape(-1, 1)
y_house = prices

model_house = LinearRegression()
model_house.fit(X_house, y_house)
print("House price prediction:")
print(f"  Coefficient (price per sqft): ${model_house.coef_[0]:.2f}")
print(f"  For 2500 sqft house: ${model_house.predict([[2500]])[0]:.2f}")

# Scenario 2: Model comparison with cross-validation
# WHY: Robust performance comparison.
from sklearn.model_selection import cross_val_score
scores_linear = cross_val_score(LinearRegression(), X_multi, y_multi, cv=5, scoring='r2')
scores_ridge = cross_val_score(Ridge(), X_multi, y_multi, cv=5, scoring='r2')
print(f"\n5-fold CV scores:")
print(f"  Linear: {scores_linear.mean():.4f} (+/- {scores_linear.std():.4f})")
print(f"  Ridge:  {scores_ridge.mean():.4f} (+/- {scores_ridge.std():.4f})")

# Scenario 3: Hyperparameter tuning for regularization strength
# WHY: Find optimal alpha that minimizes test error.
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_reg, y_train_reg)
print(f"\nBest alpha: {grid_search.best_params_['alpha']}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# KEY TAKEAWAY:
# Linear regression: minimize (actual - predicted)² for continuous targets.
# Coefficients: feature importance (if scaled); intercept: baseline.
# R²: proportion of variance explained (0-1,closer to 1 is better).
# Ridge/Lasso: prevent overfitting via regularization (L2/L1 penalty).
# Scale features: makes coefficients comparable and speeds optimization.
# Assumptions: linear relationship, normal residuals, constant variance, independence.
