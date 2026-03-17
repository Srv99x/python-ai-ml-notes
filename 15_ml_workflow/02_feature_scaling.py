# Revision Notes:
# Topic: Feature Scaling & Normalization
# Why it matters for AI/ML: Many algorithms (KNN, SVM, Neural Networks, Linear Models) assume normalized input.
# Unscaled features with different ranges can cause convergence issues and bias results.
# Tree-based models don't need scaling but scaling can help regularization.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)

# ---
# THE PROBLEM: UNSCALED DATA
# ---

# WHY: Features with different ranges dominate distance-based algorithms.

print("="*60)
print("FEATURE SCALING IN ML")
print("="*60)

# Create sample data with very different ranges
X = np.array([
    [1000, 0.5],      # Feature 1: 0-2000, Feature 2: 0-1
    [2000, 1.0],
    [500, 0.3],
    [1500, 0.8]
])

print("\n=== Problem: Different Feature Ranges ===")
print(f"Original data:")
print(f"  Feature 1 range: [{X[:, 0].min()}, {X[:, 0].max()}]")
print(f"  Feature 2 range: [{X[:, 1].min()}, {X[:, 1].max()}]")
print(f"\nRaw data:")
print(X)

# Compute distances (Euclidean)
# WHY: Unscaled Feature 1 dominates distance calculation.
dist_12 = np.linalg.norm(X[0] - X[1])
dist_13 = np.linalg.norm(X[0] - X[2])
print(f"\nEuclidean distances (unscaled):")
print(f"  Sample 0 to 1: {dist_12:.2f}")
print(f"  Sample 0 to 2: {dist_13:.2f}")
print(f"  Feature 1 contribution dominates!")

# ---
# STANDARDSCALER: ZERO MEAN, UNIT VARIANCE
# ---

# WHY: Most common; assumes normal distribution; works well for most algorithms.

print("\n" + "="*60)
print("\n=== StandardScaler (Z-score normalization) ===")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaled data:")
print(X_scaled)
print(f"\nMean: {X_scaled.mean(axis=0)}")
print(f"Std: {X_scaled.std(axis=0)}")

# Distances after scaling
# WHY: Now both features contribute equally.
dist_12_scaled = np.linalg.norm(X_scaled[0] - X_scaled[1])
dist_13_scaled = np.linalg.norm(X_scaled[0] - X_scaled[2])
print(f"\nEuclidean distances (scaled):")
print(f"  Sample 0 to 1: {dist_12_scaled:.2f}")
print(f"  Sample 0 to 2: {dist_13_scaled:.2f}")

# Formula: (x - mean) / std
# WHY: Centers data at 0 with spread of 1.
print(f"\nFormula: (x - mean) / std")
manual_scaled_f1 = (X[0, 0] - X[:, 0].mean()) / X[:, 0].std()
print(f"Manual scaling sample 0, feature 1: {manual_scaled_f1:.4f}")
print(f"StandardScaler result: {X_scaled[0, 0]:.4f}")

# ---
# MINMAXSCALER: BOUNDS TO [0, 1]
# ---

# WHY: Useful when you know min/max bounds; preserves distribution shape.

print("\n" + "="*60)
print("\n=== MinMaxScaler (bounds to [0, 1]) ===")

minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

print(f"Scaled data (MinMaxScaler):")
print(X_minmax)
print(f"\nMin: {X_minmax.min(axis=0)}")
print(f"Max: {X_minmax.max(axis=0)}")

# Formula: (x - min) / (max - min)
# WHY: Linear transformation, preserves relationships.
print(f"\nFormula: (x - min) / (max - min)")
manual_minmax_f1 = (X[0, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
print(f"Manual scaling sample 0, feature 1: {manual_minmax_f1:.4f}")
print(f"MinMaxScaler result: {X_minmax[0, 0]:.4f}")

# Comparison
# WHY: StandardScaler vs MinMaxScaler depends on data distribution.
print(f"\nComparison of StandardScaler vs MinMaxScaler:")
print(f"  StandardScaler: infinite range (unbounded)")
print(f"  MinMaxScaler: bounded [0, 1]")

# ---
# ROBUSTSCALER: RESILIENT TO OUTLIERS
# ---

# WHY: Uses median/IQR instead of mean/std; ignores extreme values.

print("\n" + "="*60)
print("\n=== RobustScaler (outlier-resistant) ===")

# Add outlier
X_with_outlier = np.vstack([X, [[50000, -10]]])

robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X_with_outlier)

print(f"Original data with outlier:")
print(X_with_outlier[-1])
print(f"\nRobustScaler result:")
print(X_robust)

# Compare to StandardScaler
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X_with_outlier)
print(f"\nStandardScaler result (for comparison):")
print(X_standard)

print(f"\nRobustScaler formula: (x - median) / IQR")
print(f"  Less affected by outliers than StandardScaler")
print(f"  Bounded output possible with extreme values")

# ---
# NORMALIZER: ROW-WISE NORMALIZATION
# ---

# WHY: Normalize each sample to unit norm (length=1); useful for text/image data.

print("\n" + "="*60)
print("\n=== Normalizer (row-wise, L2 norm) ===")

normalizer = Normalizer(norm='l2')
X_normalized = normalizer.transform(X)

print(f"Normalized data (L2 norm per row = 1):")
print(X_normalized)

# Verify L2 norms
norms = np.linalg.norm(X_normalized, axis=1)
print(f"\nL2 norms: {norms}")  # Should be all 1.0

# L1 norm (Manhattan distance)
normalizer_l1 = Normalizer(norm='l1')
X_normalized_l1 = normalizer_l1.transform(X)
print(f"\nL1 normalized (sum of absolute values = 1):")
print(X_normalized_l1)
l1_sums = np.abs(X_normalized_l1).sum(axis=1)
print(f"L1 sums: {l1_sums}")

# ---
# ALGORITHM-SPECIFIC REQUIREMENTS
# ---

# WHY: Different algorithms have different scaling needs.

print("\n" + "="*60)
print("\n=== Algorithm Scaling Requirements ===")

data = load_breast_cancer()
X_data = data.data
y_data = data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, 
                                                      random_state=42, stratify=y_data)

print(f"Dataset: Breast Cancer ({X_data.shape[0]} samples, {X_data.shape[1]} features)")
print(f"Feature ranges: [{X_data.min():.2f}, {X_data.max():.2f}]")

# KNN: NEEDS SCALING
# WHY: Distance-based; unscaled features dominate.
print(f"\n1. KNN (k=5):")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
acc_unscaled_knn = knn_unscaled.score(X_test, y_test)

scaler_knn = StandardScaler()
X_train_scaled_knn = scaler_knn.fit_transform(X_train)
X_test_scaled_knn = scaler_knn.transform(X_test)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled_knn, y_train)
acc_scaled_knn = knn_scaled.score(X_test_scaled_knn, y_test)

print(f"  Unscaled accuracy: {acc_unscaled_knn:.4f}")
print(f"  Scaled accuracy: {acc_scaled_knn:.4f}")
print(f"  Difference: {abs(acc_scaled_knn - acc_unscaled_knn):.4f} ← SCALING HELPS")

# Logistic Regression: NEEDS SCALING
# WHY: Gradient descent converges faster with scaled features.
print(f"\n2. Logistic Regression:")
lr_unscaled = LogisticRegression(max_iter=10000, random_state=42)
lr_unscaled.fit(X_train, y_train)
acc_unscaled_lr = lr_unscaled.score(X_test, y_test)

scaler_lr = StandardScaler()
X_train_scaled_lr = scaler_lr.fit_transform(X_train)
X_test_scaled_lr = scaler_lr.transform(X_test)
lr_scaled = LogisticRegression(max_iter=10000, random_state=42)
lr_scaled.fit(X_train_scaled_lr, y_train)
acc_scaled_lr = lr_scaled.score(X_test_scaled_lr, y_test)

print(f"  Unscaled accuracy: {acc_unscaled_lr:.4f}")
print(f"  Scaled accuracy: {acc_scaled_lr:.4f}")
print(f"  Difference: {abs(acc_scaled_lr - acc_unscaled_lr):.4f} ← SCALING HELPS")

# Decision Trees: DON'T NEED SCALING
# WHY: Split-based; invariant to feature scale.
from sklearn.tree import DecisionTreeClassifier
print(f"\n3. Decision Tree:")
dt_unscaled = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_unscaled.fit(X_train, y_train)
acc_unscaled_dt = dt_unscaled.score(X_test, y_test)

scaler_dt = StandardScaler()
X_train_scaled_dt = scaler_dt.fit_transform(X_train)
X_test_scaled_dt = scaler_dt.transform(X_test)
dt_scaled = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_scaled.fit(X_train_scaled_dt, y_train)
acc_scaled_dt = dt_scaled.score(X_test_scaled_dt, y_test)

print(f"  Unscaled accuracy: {acc_unscaled_dt:.4f}")
print(f"  Scaled accuracy: {acc_scaled_dt:.4f}")
print(f"  Difference: {abs(acc_scaled_dt - acc_unscaled_dt):.4f} ← NO DIFFERENCE")

# ---
# SCALING IN PIPELINES
# ---

# WHY: Prevents accidentally fitting scaler on test data (data leakage).

print("\n" + "="*60)
print("\n=== Best Practice: Fit Scaler on Train Data Only ===")

print(f"CORRECT APPROACH:")
print(f"  1. Split into train/test")
print(f"  2. Fit scaler on TRAINING data")
print(f"  3. Transform both train and test using fitted scaler")
print(f"  4. Train model on scaled training data")
print(f"  5. Evaluate on scaled test data")

print(f"\nWHY: Scaler statistics (mean, std, min, max) should come from training data only.")
print(f"  Using test data statistics would leak information!")

# Demonstrate correct vs incorrect
print(f"\nExample:")
scaler_correct = StandardScaler()
scaler_correct.fit(X_train)  # Fit on train only
X_train_correct = scaler_correct.transform(X_train)
X_test_correct = scaler_correct.transform(X_test)
print(f"  Train mean after scaling: {X_train_correct.mean():.4f}")
print(f"  Test mean after scaling: {X_test_correct.mean():.4f}")
print(f"  → Test has non-zero mean (correct; test wasn't used to fit)")

print(f"\nINCORRECT (DON'T DO THIS):")
scaler_incorrect = StandardScaler()
scaler_incorrect.fit_transform(np.vstack([X_train, X_test]))  # Fit on train+test!
print(f"  This leaks information about test data into training!")

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

print("Scenario 1: Choose scaler based on data distribution")
print("  Normal distribution → StandardScaler (most common)")
print("  Bounded data [min, max] → MinMaxScaler")
print("  Data with outliers → RobustScaler")
print("  Text/image embeddings → Normalizer")

print("\nScenario 2: Scikit-learn Pipeline (automatic scaling)")
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
accuracy_pipeline = pipeline.score(X_test, y_test)
print(f"  Pipeline automatically handles scaling/training")
print(f"  Accuracy: {accuracy_pipeline:.4f}")

print("\nScenario 3: Save scaler for deployment")
import pickle
scaler_deploy = StandardScaler()
scaler_deploy.fit(X_train)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler_deploy, f)
print(f"  Save scaler object with trained model")
print(f"  Use SAME scaler to transform new data in production")

# KEY TAKEAWAY:
# Scaling essential for: KNN, SVM, Linear Models, Neural Networks, gradient descent-based algorithms.
# NOT needed for: Tree-based models (Decision Trees, Random Forests, Gradient Boosting).
# StandardScaler: (x - mean) / std → most common.
# MinMaxScaler: (x - min) / (max - min) → bounds [0, 1].
# RobustScaler: (x - median) / IQR → handles outliers.
# Normalizer: row-wise normalization → sum/norm to 1.
# CRITICAL: Fit scaler on TRAINING data only; apply to test data.
# Use Pipeline to automate scaling during train/predict.
