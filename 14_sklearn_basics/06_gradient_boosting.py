# Revision Notes:
# Topic: Gradient Boosting with scikit-learn
# Why it matters for AI/ML: Sequential ensemble method that trains trees on residuals.
# Achieves state-of-the-art performance on structured data.
# Requires careful hyperparameter tuning to prevent overfitting.

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
import matplotlib.pyplot as plt

np.random.seed(42)

# ---
# GRADIENT BOOSTING vs RANDOM FOREST
# ---

# WHY: Bagging (Random Forest) reduces variance; Boosting reduces bias + variance.
# Trees grown sequentially, each correcting previous errors.

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

print("=== Gradient Boosting Classification ===")
print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# Compare Random Forest vs Gradient Boosting
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_accuracy = rf.score(X_test, y_test)

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_accuracy = gb.score(X_test, y_test)

print(f"\nRandom Forest accuracy: {rf_accuracy:.4f}")
print(f"Gradient Boosting accuracy: {gb_accuracy:.4f}")
print(f"GB typically performs better but requires more tuning")

# ---
# LEARNING RATE AND N_ESTIMATORS
# ---

# WHY: Learning rate controls step size; n_estimators is number of trees.
# Lower learning rate requires more trees but often better generalization.

print("\n" + "="*60)
print("\n=== Learning Rate Effect ===")

learning_rates = [0.01, 0.05, 0.1, 0.2]
test_accuracies = []

for lr in learning_rates:
    gb_lr = GradientBoostingClassifier(learning_rate=lr, n_estimators=100, random_state=42)
    gb_lr.fit(X_train, y_train)
    acc = gb_lr.score(X_test, y_test)
    test_accuracies.append(acc)
    print(f"learning_rate={lr}: Accuracy={acc:.4f}")

# Learning rate tradeoff
# WHY: Too-high learning rate: may diverge or overfit.
# Too-low learning rate: slow convergence, needs more trees.
print(f"\nOptimal learning rate balance: 0.05-0.1 for most problems")

# ---
# MAX_DEPTH: WEAK LEARNERS
# ---

# WHY: GB uses shallow trees (max_depth=3-5 default); deep trees hurt generalization.

print("\n" + "="*60)
print("\n=== Tree Depth Effect ===")

depths = [1, 2, 3, 5, 7, 10]
print(f"Effect on accuracy (100 estimators, lr=0.1):")
for depth in depths:
    gb_depth = GradientBoostingClassifier(max_depth=depth, n_estimators=100, 
                                          learning_rate=0.1, random_state=42)
    gb_depth.fit(X_train, y_train)
    acc = gb_depth.score(X_test, y_test)
    print(f"  max_depth={depth:2d}: {acc:.4f}")

print(f"\nOptimal depth: 3-5 (shallow trees for weak learners)")

# ---
# SUBSAMPLE: STOCHASTIC GRADIENT BOOSTING
# ---

# WHY: Train each tree on random subset reduces overfitting and speeds up training.

print("\n" + "="*60)
print("\n=== Subsample Effect ===")

subsamples = [0.5, 0.7, 0.8, 1.0]
print(f"Effect on accuracy (100 estimators, lr=0.1, depth=3):")
for subsample in subsamples:
    gb_sub = GradientBoostingClassifier(subsample=subsample, n_estimators=100,
                                        learning_rate=0.1, max_depth=3, random_state=42)
    gb_sub.fit(X_train, y_train)
    acc = gb_sub.score(X_test, y_test)
    print(f"  subsample={subsample}: {acc:.4f}")

print(f"\nTypical: subsample=0.8 improves generalization")

# ---
# FEATURE IMPORTANCE AND STAGING
# ---

# WHY: Features ranked by how often they split; early stops show importance.

print("\n" + "="*60)
print("\n=== Feature Importance ===")

from sklearn.inspection import permutation_importance

# Train optimized GB model
gb_final = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                       max_depth=3, subsample=0.8, random_state=42)
gb_final.fit(X_train, y_train)

# Feature importance from tree splits
importance = gb_final.feature_importances_
print(f"Feature importance (split-based):")
for idx in np.argsort(importance)[::-1]:
    print(f"  {iris.feature_names[idx]}: {importance[idx]:.4f}")

# Permutation importance (more reliable)
# WHY: Randomly permute feature; measure performance drop.
perm_importance = permutation_importance(gb_final, X_test, y_test, n_repeats=10, 
                                         random_state=42)
print(f"\nPermutation importance:")
for idx in np.argsort(perm_importance.importances_mean)[::-1]:
    print(f"  {iris.feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f}")

# ---
# STAGED PREDICTION: MONITORING PROGRESS
# ---

# WHY: Observe training progress; detect overfitting by iteration number.

print("\n" + "="*60)
print("\n=== Staged Prediction ===")

from sklearn.model_selection import cross_val_score

# Create staged predictions
train_scores = []
test_scores = []
for y_pred in gb_final.staged_predict(X_train):
    train_scores.append(accuracy_score(y_train, y_pred))
for y_pred in gb_final.staged_predict(X_test):
    test_scores.append(accuracy_score(y_test, y_pred))

# Find overfitting point
best_test_idx = np.argmax(test_scores)
print(f"Best test accuracy at iteration {best_test_idx + 1}:")
print(f"  Train accuracy: {train_scores[best_test_idx]:.4f}")
print(f"  Test accuracy: {test_scores[best_test_idx]:.4f}")
if best_test_idx < len(test_scores) - 1:
    print(f"  Overfitting starts after iteration {best_test_idx + 1}")

# ---
# EARLY STOPPING
# ---

# WHY: Stop training when validation score plateaus to prevent overfitting.

print("\n" + "="*60)
print("\n=== Early Stopping ===")

# Manual early stopping via staged prediction
gb_early = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                       max_depth=3, random_state=42)
gb_early.fit(X_train, y_train)

best_test_score = 0
best_n_estimators = 0
for n_est, y_pred in enumerate(gb_early.staged_predict(X_test), 1):
    score = accuracy_score(y_test, y_pred)
    if score > best_test_score:
        best_test_score = score
        best_n_estimators = n_est

print(f"Without early stopping: n_estimators=200")
print(f"  Test accuracy: {gb_early.score(X_test, y_test):.4f}")
print(f"With early stopping at n_estimators={best_n_estimators}:")
print(f"  Test accuracy would be: {best_test_score:.4f}")

# Built-in early stopping (sklearn 1.3+)
# WHY: Stop automatically based on validation score.
print(f"\nScikit-learn supports validation_fraction and n_iter_no_change for automatic stopping")

# ---
# GRADIENT BOOSTING REGRESSION
# ---

# WHY: Boosting works for continuous targets too.

print("\n" + "="*60)
print("\n=== Gradient Boosting Regression ===")

X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=20, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, subsample=0.8, random_state=42)
gb_reg.fit(X_reg_train, y_reg_train)

y_pred_reg = gb_reg.predict(X_reg_test)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
r2 = gb_reg.score(X_reg_test, y_reg_test)

print(f"RMSE: {rmse:.4f}")
print(f"R² score: {r2:.4f}")

# Residuals from each tree
# WHY: GB trains on residuals; each tree corrects previous errors.
print(f"\nGB regression: each tree predicts residual from previous prediction")
print(f"Final prediction = tree1 + lr*tree2 + lr*tree3 + ...")

# ---
# HUBER LOSS FOR ROBUSTNESS
# ---

# WHY: Less sensitive to outliers than squared error.

print("\n" + "="*60)
print("\n=== Loss Functions ===")

losses = ['squared_error', 'huber', 'quantile']
print(f"Classification loss: 'log_loss' (logistic loss)")
print(f"Regression loss options:")
for loss_fn in losses:
    try:
        gb_loss = GradientBoostingRegressor(loss=loss_fn, n_estimators=100,
                                            learning_rate=0.1, max_depth=3, random_state=42)
        gb_loss.fit(X_reg_train, y_reg_train)
        r2 = gb_loss.score(X_reg_test, y_reg_test)
        print(f"  {loss_fn}: R² = {r2:.4f}")
    except:
        print(f"  {loss_fn}: not available")

# ---
# INITIALIZATION: WARM START
# ---

# WHY: Can incrementally add more boosting rounds (n_estimators).

print("\n" + "="*60)
print("\n=== Warm Start ===")

gb_warm = GradientBoostingClassifier(n_estimators=10, warm_start=True, random_state=42)
gb_warm.fit(X_train, y_train)
print(f"Initial (10 estimators): {gb_warm.score(X_test, y_test):.4f}")

gb_warm.n_estimators = 50
gb_warm.fit(X_train, y_train)
print(f"After adding 40 more (50 total): {gb_warm.score(X_test, y_test):.4f}")

gb_warm.n_estimators = 100
gb_warm.fit(X_train, y_train)
print(f"After adding 50 more (100 total): {gb_warm.score(X_test, y_test):.4f}")

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Hyperparameter tuning
print("Typical GB hyperparameter tuning:")
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9]
}
print(f"  Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
print(f"  Use RandomizedSearchCV or Bayesian optimization for 81+ combinations")

from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42),
                                   param_grid, n_iter=20, cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
print(f"  Best parameters: {random_search.best_params_}")
print(f"  Best CV score: {random_search.best_score_:.4f}")

# Scenario 2: Imbalanced classification
# WHY: GB can handle imbalanced via init estimator.
from sklearn.datasets import make_classification
X_imbal, y_imbal = make_classification(n_samples=1000, n_features=5, n_classes=2,
                                       weights=[0.9, 0.1], random_state=42)

from sklearn.model_selection import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
scores_imbal = []
for train_idx, test_idx in cv.split(X_imbal, y_imbal):
    gb_imbal = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                          max_depth=3, random_state=42)
    gb_imbal.fit(X_imbal[train_idx], y_imbal[train_idx])
    scores_imbal.append(gb_imbal.score(X_imbal[test_idx], y_imbal[test_idx]))

print(f"\nImbalanced data CV scores: {[f'{s:.4f}' for s in scores_imbal]}")

# Scenario 3: Training monitoring
# WHY: Plot train/test loss over iterations to detect overfitting.
print(f"\nTo monitor training:")
print(f"  Use staged_predict to get predictions at each iteration")
print(f"  Plot accuracy/loss over iteration number")
print(f"  Stop when test loss increases (overfitting)")

# KEY TAKEAWAY:
# Gradient Boosting: Sequential trees, each correcting previous errors.
# learning_rate: control step size (0.01-0.1 typical).
# n_estimators: number of trees (100-1000 typical).
# max_depth: use SHALLOW trees (3-5); weak learners better than strong.
# subsample: use 0.7-0.8 for SGBoost (stochastic).
# Requires careful tuning but achieves excellent performance.
# staged_predict: monitor progress and detect overfitting.
# No feature scaling required.
# Slower training than Random Forest but often better accuracy.
