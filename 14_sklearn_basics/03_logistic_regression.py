# Revision Notes:
# Topic: Logistic Regression with scikit-learn
# Why it matters for AI/ML: Logistic regression is the standard baseline for binary/multiclass classification.
# Despite its name, it's a classification algorithm (not regression).
# Understanding logistic regression is essential before moving to complex classifiers.

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_breast_cancer, make_classification
import matplotlib.pyplot as plt

np.random.seed(42)

# ---
# BINARY CLASSIFICATION: LOGISTIC REGRESSION
# ---

# WHY: Predict binary outcome (0 or 1, yes or no, positive or negative).
# Uses sigmoid function to convert linear output to probability [0, 1].

# Load binary classification dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target  # 0: malignant, 1: benign

print("=== Logistic Regression: Binary Classification ===")
print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
print(f"Classes: {np.unique(y)} (0: malignant, 1: benign)")
print(f"Class distribution: {np.bincount(y)}")

# Scale features (important for logistic regression)
# WHY: Algorithm uses distance metrics; unscaled features dominate.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

# Train logistic regression model
# WHY: Finds linear decision boundary; probability interpreted as P(class=1|features).
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print(f"\nModel coefficients shape: {model.coef_.shape}")
print(f"Coefficients: {model.coef_[0][:5]}... (first 5)")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Predictions
# WHY: Predict class (0 or 1).
y_pred = model.predict(X_test)
print(f"\nPredictions (sample): {y_pred[:10]}")

# Predicted probabilities
# WHY: Probability of each class; useful for ranking or setting custom threshold.
y_prob = model.predict_proba(X_test)
print(f"\nProbabilities (first 5 samples):")
for i in range(5):
    print(f"  Sample {i}: P(class=0)={y_prob[i][0]:.4f}, P(class=1)={y_prob[i][1]:.4f}")

# Classification accuracy
# WHY: Fraction of correct predictions.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# ---
# MULTICLASS CLASSIFICATION
# ---

# WHY: Extend logistic regression to >2 classes.
# Default: one-vs-rest (separate binary classifier for each class).

from sklearn.datasets import load_iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target  # 3 classes: 0, 1, 2

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# Train multiclass model
# WHY: One-vs-rest creates 3 binary classifiers.
model_multi = LogisticRegression(max_iter=1000, random_state=42)
model_multi.fit(X_train_iris, y_train_iris)

y_pred_iris = model_multi.predict(X_test_iris)
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)

print("\n" + "="*60)
print("\n=== Multiclass Logistic Regression ===")
print(f"Classes: {np.unique(y_iris)} ({len(np.unique(y_iris))} classes)")
print(f"Accuracy: {accuracy_iris:.4f}")

# Predicted probabilities for multiclass
y_prob_iris = model_multi.predict_proba(X_test_iris)
print(f"\nProbabilities for first sample (3 classes):")
for class_id in range(3):
    print(f"  P(class={class_id}) = {y_prob_iris[0][class_id]:.4f}")

# ---
# PROBABILITY THRESHOLD TUNING
# ---

# WHY: Default threshold is 0.5; adjust based on problem needs.
# Lower threshold = more positive predictions (higher recall, lower precision).

print("\n" + "="*60)
print("\n=== Threshold Tuning ===")

# Get probabilities for positive class
y_prob_pos = y_prob[:, 1]

# Different thresholds
thresholds = [0.3, 0.5, 0.7, 0.9]
for thresh in thresholds:
    y_pred_thresh = (y_prob_pos >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    print(f"Threshold {thresh}: Accuracy = {acc:.4f}, Predictions = {y_pred_thresh.sum()} positive")

# WHY: Lower threshold increases sensitivity to positive class (useful for rare diseases).
print("\nLower threshold → more positive predictions → higher recall, lower precision")

# ---
# REGULARIZATION: L1 AND L2
# ---

# WHY: Prevent overfitting by penalizing large coefficients.

print("\n" + "="*60)
print("\n=== Regularization in Logistic Regression ===")

# L2 regularization (Ridge)
# WHY: Shrink coefficients toward zero; prevents any from being too large.
model_l2 = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)
model_l2.fit(X_train, y_train)
acc_l2 = model_l2.score(X_test, y_test)

# L1 regularization (Lasso)
# WHY: Can shrink coefficients to exactly zero (automatic feature selection).
model_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42, max_iter=1000)
model_l1.fit(X_train, y_train)
acc_l1 = model_l1.score(X_test, y_test)

print(f"L2 accuracy: {acc_l2:.4f}, Non-zero coefficients: {(model_l2.coef_[0] != 0).sum()}")
print(f"L1 accuracy: {acc_l1:.4f}, Non-zero coefficients: {(model_l1.coef_[0] != 0).sum()}")

# Inverse regularization parameter C
# WHY: C = 1/lambda; smaller C = stronger regularization.
C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
scores = []
for C in C_values:
    model_C = LogisticRegression(C=C, random_state=42, max_iter=1000)
    model_C.fit(X_train, y_train)
    scores.append(model_C.score(X_test, y_test))

print(f"\nC vs Test Accuracy:")
for C, acc in zip(C_values, scores):
    print(f"  C = {C}: accuracy = {acc:.4f}")

# ---
# FEATURE IMPORTANCE FROM COEFFICIENTS
# ---

# WHY: Interpret which features matter most for predictions.

print("\n" + "="*60)
print("\n=== Feature Importance ===")

# Get coefficients from binary model
coef = model.coef_[0]
feature_importance_idx = np.argsort(np.abs(coef))[::-1]

print(f"Top 5 most important features (by coefficient magnitude):")
for rank, feat_idx in enumerate(feature_importance_idx[:5], 1):
    feat_name = cancer.feature_names[feat_idx]
    coef_val = coef[feat_idx]
    print(f"  {rank}. {feat_name}: {coef_val:.4f}")

print(f"\nPositive coefficient → increases P(positive class)")
print(f"Negative coefficient → decreases P(positive class)")

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Imbalanced binary classification
# WHY: Adjust class weights when classes are imbalanced.
X_imbal, y_imbal = make_classification(n_samples=1000, n_features=10, n_classes=2,
                                       weights=[0.9, 0.1], random_state=42)

# Standard model (biased toward majority class)
model_standard = LogisticRegression(random_state=42, max_iter=1000)
model_standard.fit(X_imbal, y_imbal)
pred_standard = model_standard.predict(X_imbal)
print(f"Standard model: predicts class 1 {(pred_standard==1).sum()} times")

# Balanced model (equal weight for imbalanced data)
model_balanced = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model_balanced.fit(X_imbal, y_imbal)
pred_balanced = model_balanced.predict(X_imbal)
print(f"Balanced model: predicts class 1 {(pred_balanced==1).sum()} times")

# Scenario 2: Cross-validation for robust performance
# WHY: Single train-test split has high variance.
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(random_state=42, max_iter=1000),
                         X_scaled, y, cv=5, scoring='accuracy')
print(f"\n5-fold CV scores: {scores.round(4)}")
print(f"Mean: {scores.mean():.4f} +/- {scores.std():.4f}")

# Scenario 3: Decision boundary visualization (2D)
# WHY: Understand how model separates classes.
from sklearn.datasets import make_blobs
X_2d, y_2d = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
X2d_train, X2d_test, y2d_train, y2d_test = train_test_split(X_2d, y_2d, test_size=0.3, random_state=42)

model_2d = LogisticRegression(random_state=42)
model_2d.fit(X2d_train, y2d_train)

print(f"\n2D classification accuracy: {model_2d.score(X2d_test, y2d_test):.4f}")
# Visualizing would show linear decision boundary

# KEY TAKEAWAY:
# Logistic regression: P(y=1|x) = sigmoid(β₀ + β₁x₁ + ... + βₙxₙ)
# Scale features before training.
# Coefficients show feature direction; magnitude shows strength (if scaled).
# predict(): hard class; predict_proba(): soft probabilities.
# Adjust threshold based on precision-recall tradeoff.
# L1/L2 regularization: C parameter controls strength (lower C = stronger).
# class_weight='balanced': helps with imbalanced datasets.
