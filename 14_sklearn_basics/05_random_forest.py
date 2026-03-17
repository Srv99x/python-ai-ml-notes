# Revision Notes:
# Topic: Random Forest with scikit-learn
# Why it matters for AI/ML: Ensemble method that reduces overfitting.
# Combines multiple trees for better generalization and robustness.
# Random forests are industry standard for classification and regression tasks.

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt

np.random.seed(42)

# ---
# RANDOM FOREST CLASSIFICATION
# ---

# WHY: Bootstrap aggregating (bagging) + random feature selection reduces variance.
# Each tree sees slightly different data; averaging predictions improves stability.

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

print("=== Random Forest Classification ===")
print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# Train random forest
# WHY: n_estimators: number of trees (more = better, but diminishing returns).
# random_state: reproducibility
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Accuracy: {accuracy:.4f}")

# Probabilistic predictions
# WHY: Average probability across all trees for calibrated estimates.
y_proba = rf.predict_proba(X_test)
print(f"Predicted probabilities (first sample):")
print(f"  Class probabilities: {y_proba[0]}")
print(f"  Most likely class: {y_proba[0].argmax()}")

# Feature importance (averaged across all trees)
# WHY: Shows which features matter across the ensemble.
feature_importance = rf.feature_importances_
print(f"\nFeature importance (all trees averaged):")
for idx in np.argsort(feature_importance)[::-1]:
    print(f"  {iris.feature_names[idx]}: {feature_importance[idx]:.4f}")

# Out-of-bag score
# WHY: Use unselected samples (~ 1/3) for free validation during training.
oob_score = rf.oob_score_
print(f"\nOut-of-bag score: {oob_score:.4f}")

# ---
# SINGLE TREE VS RANDOM FOREST
# ---

# WHY: Compare variance reduction from ensemble.

print("\n" + "="*60)
print("\n=== Single Tree vs Random Forest ===")

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_accuracy = tree.score(X_test, y_test)

rf_accuracy = rf.score(X_test, y_test)

print(f"Single Decision Tree accuracy: {tree_accuracy:.4f}")
print(f"Random Forest (100 trees) accuracy: {rf_accuracy:.4f}")
if rf_accuracy > tree_accuracy:
    improvement = (rf_accuracy - tree_accuracy) / tree_accuracy * 100
    print(f"Improvement: +{improvement:.1f}%")

# Training accuracy (overfitting check)
# WHY: Trees overfit; forests generalize better.
print(f"\nTraining accuracy:")
print(f"  Single tree: {tree.score(X_train, y_train):.4f}")
print(f"  Random forest: {rf.score(X_train, y_train):.4f}")

# ---
# HYPERPARAMETER TUNING
# ---

# WHY: Key parameters control bias-variance tradeoff.

print("\n" + "="*60)
print("\n=== Hyperparameter Effects ===")

# n_estimators: number of trees
# WHY: More trees reduce variance; diminishing returns after ~100.
estimators = [5, 10, 20, 50, 100, 200]
train_scores = []
test_scores = []

for n_est in estimators:
    rf_est = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf_est.fit(X_train, y_train)
    train_scores.append(rf_est.score(X_train, y_train))
    test_scores.append(rf_est.score(X_test, y_test))

print(f"n_estimators effect:")
for n_est, train, test in zip(estimators, train_scores, test_scores):
    print(f"  {n_est:3d} trees: Train={train:.4f}, Test={test:.4f}")

# max_depth: control tree complexity
# WHY: Shallower trees reduce overfitting but may underfit.
print(f"\nmax_depth effect (100 trees):")
depths = [2, 3, 5, 10, None]
for depth in depths:
    rf_depth = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    rf_depth.fit(X_train, y_train)
    print(f"  max_depth={depth}: Train={rf_depth.score(X_train, y_train):.4f}, "
          f"Test={rf_depth.score(X_test, y_test):.4f}")

# min_samples_leaf: min samples per leaf
# WHY: Larger values prevent overfitting by keeping leaves less pure.
print(f"\nmin_samples_leaf effect (100 trees):")
leaf_sizes = [1, 2, 4, 8]
for leaf_size in leaf_sizes:
    rf_leaf = RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf_size, 
                                      random_state=42)
    rf_leaf.fit(X_train, y_train)
    print(f"  min_samples_leaf={leaf_size}: Train={rf_leaf.score(X_train, y_train):.4f}, "
          f"Test={rf_leaf.score(X_test, y_test):.4f}")

# max_features: features per split
# WHY: sqrt(n_features) default for classification; reduces correlation between trees.
print(f"\nmax_features effect (100 trees):")
print(f"  'sqrt': randomly select sqrt(n_features) per split")
print(f"  'log2': randomly select log2(n_features) per split")
print(f"  None: use all features (same as bagging)")

# ---
# RANDOM FOREST REGRESSION
# ---

# WHY: Ensemble regression for continuous targets.

print("\n" + "="*60)
print("\n=== Random Forest Regression ===")

X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=20, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train regression forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_reg_train, y_reg_train)

# Predictions
y_reg_pred = rf_reg.predict(X_reg_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
r2 = rf_reg.score(X_reg_test, y_reg_test)

print(f"RMSE: {rmse:.4f}")
print(f"R² score: {r2:.4f}")

# Prediction variance estimation
# WHY: Can estimate prediction uncertainty.
predictions = np.array([tree.predict(X_reg_test) for tree in rf_reg.estimators_])
pred_std = predictions.std(axis=0)
print(f"\nPrediction uncertainty (first 5 samples):")
for i in range(5):
    pred_mean = predictions[:, i].mean()
    print(f"  Sample {i}: {pred_mean:.2f} ± {pred_std[i]:.2f}")

# ---
# FEATURE IMPORTANCE INTERPRETATION
# ---

# WHY: Understand which features drive decisions.

print("\n" + "="*60)
print("\n=== Feature Importance ===")

# Importance from impurity decrease (MDI)
# WHY: Default; how much each feature decreases impurity in splits.
mdi_importance = rf.feature_importances_

# Permutation importance
# WHY: Shuffle feature and measure performance drop; true importance.
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

print("Feature importance comparison:")
print(f"{'Feature':<25} {'MDI':<10} {'Permutation':<10}")
print("-" * 45)
for feat_idx in range(len(iris.feature_names)):
    feat_name = iris.feature_names[feat_idx]
    mdi = mdi_importance[feat_idx]
    perm = perm_importance.importances_mean[feat_idx]
    print(f"{feat_name:<25} {mdi:<10.4f} {perm:<10.4f}")

# ---
# OUT-OF-BAG ERROR ESTIMATION
# ---

# WHY: Each tree trained on bootstrap sample (~63%); rest (~37%) used for validation.

print("\n" + "="*60)
print("\n=== Out-of-Bag (OOB) Score ===")

# Train forest with oob_score=True
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)

print(f"OOB score (training set): {rf_oob.oob_score_:.4f}")
print(f"Test set score: {rf_oob.score(X_test, y_test):.4f}")
print(f"OOB provides ~unbiased estimate without separate validation set")

# ---
# HANDLING IMBALANCED DATA
# ---

# WHY: Class weight adjustments improve minority class detection.

print("\n" + "="*60)
print("\n=== Imbalanced Data Handling ===")

from sklearn.datasets import make_classification
X_imbal, y_imbal = make_classification(n_samples=1000, n_features=5, n_classes=2,
                                       weights=[0.9, 0.1], random_state=42)

X_imbal_train, X_imbal_test, y_imbal_train, y_imbal_test = train_test_split(
    X_imbal, y_imbal, test_size=0.2, random_state=42, stratify=y_imbal
)

# Unbalanced forest
rf_unbalanced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_unbalanced.fit(X_imbal_train, y_imbal_train)

# Balanced forest
rf_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                      random_state=42)
rf_balanced.fit(X_imbal_train, y_imbal_train)

pred_unbal = rf_unbalanced.predict(X_imbal_test)
pred_bal = rf_balanced.predict(X_imbal_test)

print(f"Minority class predictions (out of {(y_imbal_test==1).sum()} actual):")
print(f"  Unbalanced forest: {(pred_unbal==1).sum()}")
print(f"  Balanced forest (class_weight='balanced'): {(pred_bal==1).sum()}")

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Feature selection
# WHY: Remove low-importance features for simpler model.
print("Feature selection via importance:")
threshold = np.percentile(mdi_importance, 50)  # Top 50%
selected = np.where(mdi_importance >= threshold)[0]
print(f"  Features above 50th percentile: {[iris.feature_names[i] for i in selected]}")

# Scenario 2: Hyperparameter tuning with GridSearchCV
# WHY: Find best parameters via cross-validation.
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"\nGridSearchCV best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
final_accuracy = grid_search.score(X_test, y_test)
print(f"Test set accuracy: {final_accuracy:.4f}")

# Scenario 3: Probability calibration
# WHY: Random forest probabilities not perfectly calibrated; may need adjustment.
from sklearn.calibration import CalibratedClassifierCV
rf_uncalibrated = RandomForestClassifier(n_estimators=100, random_state=42)
calibrated_rf = CalibratedClassifierCV(rf_uncalibrated, method='sigmoid', cv=5)
calibrated_rf.fit(X_train, y_train)
print(f"\nCalibrated RF test accuracy: {calibrated_rf.score(X_test, y_test):.4f}")

# KEY TAKEAWAY:
# Random Forest: Bootstrap aggregating + random features reduces variance.
# n_estimators: more trees = better; diminishing returns ~100.
# max_depth, min_samples_leaf: control overfitting.
# feature_importances_: identify important features.
# OOB score: free validation during training.
# No feature scaling required.
# Parallel training: n_jobs=-1 uses all cores.
# Handles imbalanced data via class_weight='balanced'.
# Less interpretable than single tree but much better generalization.
