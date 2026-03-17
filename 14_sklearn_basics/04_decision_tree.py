# Revision Notes:
# Topic: Decision Trees with scikit-learn
# Why it matters for AI/ML: Decision trees are interpretable, handle non-linearity, and form basis for ensemble methods.
# Trees don't require feature scaling and can capture complex interactions.
# Understanding trees is essential before learning random forests and gradient boosting.

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)

# ---
# CLASSIFICATION TREE
# ---

# WHY: Recursively split data to create hierarchical decision rules.
# Each split maximizes information gain (reduces impurity).

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print("=== Decision Tree Classification ===")
print(f"Dataset: {len(X)} samples, 150 features")
print(f"Classes: {np.unique(y)} ({len(np.unique(y))} classes)")

# Train decision tree with default parameters
# WHY: Split using Gini impurity (minimizes class mixing).
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Predictions
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Feature importance
# WHY: How much each feature contributes to decisions.
feature_importance = tree.feature_importances_
top_features = np.argsort(feature_importance)[::-1][:3]
print(f"\nTop 3 most important features:")
for feat_idx in top_features:
    print(f"  {iris.feature_names[feat_idx]}: {feature_importance[feat_idx]:.4f}")

# Tree depth
# WHY: Deeper trees overfit; shallow trees underfit.
print(f"\nTree depth: {tree.get_depth()}")
print(f"Number of leaves: {tree.get_n_leaves()}")

# ---
# CONTROLLING TREE COMPLEXITY
# ---

# WHY: Prevent overfitting by constraining tree growth.

print("\n" + "="*60)
print("\n=== Tree Complexity Control ===")

# Different max_depths
depths = [1, 3, 5, 10, 20]
train_acc = []
test_acc = []

for depth in depths:
    tree_depth = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_depth.fit(X_train, y_train)
    train_acc.append(tree_depth.score(X_train, y_train))
    test_acc.append(tree_depth.score(X_test, y_test))

for depth, train, test in zip(depths, train_acc, test_acc):
    print(f"max_depth={depth}: Train={train:.4f}, Test={test:.4f}", end="")
    if train > test + 0.05:
        print(" → overfitting")
    else:
        print()

# Optimal depth: where test accuracy peaks
optimal_depth = depths[np.argmax(test_acc)]
print(f"Optimal max_depth: {optimal_depth}")

# Other complexity parameters
# WHY: Control split and leaf criteria.
print(f"\nOther complexity parameters:")
print(f"  min_samples_split: min samples to create new split (overfitting if too low)")
print(f"  min_samples_leaf: min samples in leaf node")
print(f"  max_features: limit features considered per split")

# ---
# REGRESSION TREE
# ---

# WHY: Predict continuous targets by creating average predictions per leaf.

from sklearn.datasets import make_regression
X_reg, y_reg = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("\n" + "="*60)
print("\n=== Decision Tree Regression ===")

# Train regression tree
tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_reg.fit(X_reg_train, y_reg_train)

# Predictions
y_reg_pred = tree_reg.predict(X_reg_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse:.4f}")
print(f"R² score: {tree_reg.score(X_reg_test, y_reg_test):.4f}")

# Tree splits data into rectangular regions
# WHY: Each leaf is a constant prediction (average of samples in that leaf).
print(f"\nTree depth: {tree_reg.get_depth()}")
print(f"Number of leaves: {tree_reg.get_n_leaves()}")

# ---
# SPLITCRITERION: GINI VS ENTROPY
# ---

# WHY: Different measures of impurity.

print("\n" + "="*60)
print("\n=== Split Criterion: Gini vs Entropy ===")

# Gini impurity: measures class mixing
# WHY: Default; computationally faster; works well in practice.
tree_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
tree_gini.fit(X_train, y_train)
acc_gini = tree_gini.score(X_test, y_test)

# Entropy (Information Gain)
# WHY: Information-theoretic measure; often similar results to Gini.
tree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
tree_entropy.fit(X_train, y_train)
acc_entropy = tree_entropy.score(X_test, y_test)

print(f"Gini criterion accuracy: {acc_gini:.4f}")
print(f"Entropy criterion accuracy: {acc_entropy:.4f}")
print(f"Usually similar; Gini is default and faster")

# ---
# HANDLING IMBALANCED DATA
# ---

# WHY: Tree tends to favor majority class when imbalanced.

print("\n" + "="*60)
print("\n=== Imbalanced Data Handling ===")

X_imbal, y_imbal = make_classification(n_samples=1000, n_features=5, n_classes=2,
                                       weights=[0.9, 0.1], random_state=42)

# Standard tree
tree_standard = DecisionTreeClassifier(random_state=42)
tree_standard.fit(X_imbal, y_imbal)
pred_standard = tree_standard.predict(X_imbal)

# Balanced tree
tree_balanced = DecisionTreeClassifier(class_weight='balanced', random_state=42)
tree_balanced.fit(X_imbal, y_imbal)
pred_balanced = tree_balanced.predict(X_imbal)

print(f"Standard tree: predicts class 1 {(pred_standard==1).sum()} times")
print(f"Balanced tree: predicts class 1 {(pred_balanced==1).sum()} times")
print(f"Balanced weights penalize majority class to improve minority detection")

# ---
# FEATURE SCALING NOT REQUIRED
# ---

# WHY: Trees don't use distance metrics; only split thresholds.

print("\n" + "="*60)
print("\n=== Feature Scaling Not Required ===")

X_unscaled = np.array([[1000, 0.5], [2000, 1.0], [3000, 1.5]]) * 100
X_unscaled = X_unscaled.astype(float)
y_simple = np.array([0, 1, 1])

tree_unscaled = DecisionTreeClassifier(random_state=42)
tree_unscaled.fit(X_unscaled, y_simple)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)
tree_scaled = DecisionTreeClassifier(random_state=42)
tree_scaled.fit(X_scaled, y_simple)

print(f"Unscaled accuracy: {tree_unscaled.score(X_unscaled, y_simple):.4f}")
print(f"Scaled accuracy: {tree_scaled.score(X_scaled, y_simple):.4f}")
print(f"Same accuracy! Trees are invariant to scaling")

# ---
# TREE INTERPRETATION: DECISION PATHS
# ---

# WHY: Trees are highly interpretable; can explain predictions.

print("\n" + "="*60)
print("\n=== Tree Interpretability ===")

# Get decision path for first test sample
decision_path = tree.decision_path(X_test[:1])
leaf_id = tree.apply(X_test[:1])

print(f"Sample #0 features: {X_test[0]}")
print(f"Predicted class: {y_pred[0]}")
print(f"Prediction path complexity: {decision_path.nnz[0]} nodes visited")

# Visualize tree (text representation)
# WHY: Understand exact decision rules.
from sklearn.tree import export_text
tree_rules = export_text(tree, feature_names=iris.feature_names, max_depth=2)
print(f"\nTree rules (first 2 levels):")
print(tree_rules[:500])

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Overfitting demonstration
# WHY: Unpruned trees memorize training data.
print("Overfitting demonstration:")
tree_deep = DecisionTreeClassifier(random_state=42)  # No depth limit
tree_deep.fit(X_train, y_train)
print(f"  Deep tree (unpruned):")
print(f"    Train accuracy: {tree_deep.score(X_train, y_train):.4f}")
print(f"    Test accuracy: {tree_deep.score(X_test, y_test):.4f}")

tree_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_shallow.fit(X_train, y_train)
print(f"  Shallow tree (max_depth=3):")
print(f"    Train accuracy: {tree_shallow.score(X_train, y_train):.4f}")
print(f"    Test accuracy: {tree_shallow.score(X_test, y_test):.4f}")

# Scenario 2: Feature importance for selection
# WHY: Remove low-importance features to improve generalization.
print(f"\nFeature importance ranking:")
for idx in np.argsort(tree.feature_importances_)[::-1]:
    print(f"  {iris.feature_names[idx]}: {tree.feature_importances_[idx]:.4f}")

# Scenario 3: Hyperparameter tuning with GridSearchCV
# WHY: Find best max_depth via cross-validation.
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [2, 3, 4, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# KEY TAKEAWAY:
# Decision trees: recursively split to minimize impurity (Gini or Entropy).
# max_depth: control complexity; deeper = overfitting risk.
# feature_importances_: show which features matter.
# No feature scaling required; tree-based, not distance-based.
# Interpretable: each path is a decision rule.
# Prone to overfitting; use max_depth, min_samples_leaf for regularization.
# Foundation for Random Forest and Gradient Boosting.
