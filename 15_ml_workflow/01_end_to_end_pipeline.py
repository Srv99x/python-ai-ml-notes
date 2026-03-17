# Revision Notes:
# Topic: End-to-End ML Pipeline
# Why it matters for AI/ML: Real-world ML follows a structured workflow: load → explore → preprocess → train → evaluate → deploy.
# Understanding the full pipeline prevents common mistakes and ensures reproducibility.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("="*60)
print("END-TO-END ML WORKFLOW EXAMPLE")
print("="*60)

# ---
# STEP 1: LOAD AND EXPLORE DATA
# ---

# WHY: Understand data structure, types, missing values before modeling.

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print("\n=== STEP 1: LOAD AND EXPLORE ===")
print(f"Dataset shape: {X.shape}")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"\nFirst few rows:")
print(X.head())

# Data quality check
# WHY: Missing values and data types affect preprocessing.
print(f"\nData quality:")
print(f"  Missing values: {X.isnull().sum().sum()}")
print(f"  Data types: {X.dtypes.unique()}")
print(f"  Feature ranges:")
for col in X.columns[:3]:  # Show first 3
    print(f"    {col}: [{X[col].min():.2f}, {X[col].max():.2f}]")

# Target distribution
print(f"\nTarget distribution:")
print(f"  Class 0 (Malignant): {(y==0).sum()} samples")
print(f"  Class 1 (Benign): {(y==1).sum()} samples")
print(f"  Balance ratio: {(y==1).sum() / (y==0).sum():.2f}")

# ---
# STEP 2: DATA SPLITTING
# ---

# WHY: Train/test split prevents leakage and enables unbiased evaluation.

print("\n" + "="*60)
print("\n=== STEP 2: DATA SPLITTING ===")

# Stratified split preserves class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nTrain set class distribution:")
print(f"  Class 0: {(y_train==0).sum()}")
print(f"  Class 1: {(y_train==1).sum()}")
print(f"Test set class distribution:")
print(f"  Class 0: {(y_test==0).sum()}")
print(f"  Class 1: {(y_test==1).sum()}")

# ---
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ---

# WHY: Visualize patterns, outliers, correlations guide feature engineering.

print("\n" + "="*60)
print("\n=== STEP 3: EXPLORATORY DATA ANALYSIS ===")

# Feature statistics
print(f"Feature statistics (training set):")
print(f"  Mean feature value: {X_train.mean().mean():.4f}")
print(f"  Std feature value: {X_train.std().mean():.4f}")
print(f"  Feature ranges vary widely (0-1000+)")

# Correlation analysis
# WHY: Identify highly correlated features (multicollinearity).
corr_matrix = X_train.corr()
high_corr = np.where(np.abs(corr_matrix) > 0.95)
high_corr_pairs = [(X.columns[i], X.columns[j]) for i, j in zip(*high_corr) if i < j]
print(f"\nHighly correlated feature pairs (>0.95):")
for i, (feat1, feat2) in enumerate(high_corr_pairs[:3]):
    corr_val = corr_matrix.loc[feat1, feat2]
    print(f"  {feat1} <-> {feat2}: {corr_val:.4f}")

# Skewness analysis
# WHY: Highly skewed features may need transformation.
skewness = X_train.skew()
skewed_features = skewness[np.abs(skewness) > 1].sort_values(ascending=False)
print(f"\nHighly skewed features (|skewness| > 1):")
for feat, skew_val in skewed_features.head(3).items():
    print(f"  {feat}: {skew_val:.4f}")

# ---
# STEP 4: PREPROCESSING & FEATURE SCALING
# ---

# WHY: Feature scaling centers data; many algorithms assume normalized input.

print("\n" + "="*60)
print("\n=== STEP 4: PREPROCESSING ===")

# WHY: Fit scaler on training set only (prevents data leakage).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"After StandardScaler:")
print(f"  Training set mean: {X_train_scaled.mean():.4f}")
print(f"  Training set std: {X_train_scaled.std():.4f}")
print(f"  Test set mean: {X_test_scaled.mean():.4f}")
print(f"  Test set std: {X_test_scaled.std():.4f}")

# Feature selection (optional)
# WHY: Remove low-variance features for efficiency and interpretability.
variances = X_train_scaled.var(axis=0)
high_var_threshold = np.percentile(variances, 80)  # Keep top 80%
print(f"\nFeature selection (top 80% by variance):")
print(f"  Threshold: {high_var_threshold:.4f}")

# ---
# STEP 5: TRAIN BASELINE MODEL
# ---

# WHY: Start simple (logistic regression) before complex models.

print("\n" + "="*60)
print("\n=== STEP 5: BASELINE MODEL ===")

# Logistic Regression: simple, interpretable, fast
baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train_scaled, y_train)

y_train_pred = baseline.predict(X_train_scaled)
y_test_pred = baseline.predict(X_test_scaled)

print(f"Logistic Regression performance:")
print(f"  Train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"  Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"  Train/Test gap: {accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred):.4f}")

# Cross-validation baseline
# WHY: More reliable estimate than single train/test split.
cv_scores = cross_val_score(baseline, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n5-fold Cross-validation scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean: {cv_scores.mean():.4f}")
print(f"  Std: {cv_scores.std():.4f}")

# ---
# STEP 6: DETAILED EVALUATION
# ---

# WHY: Accuracy alone insufficient; precision/recall/F1 reveal class-specific performance.

print("\n" + "="*60)
print("\n=== STEP 6: DETAILED EVALUATION ===")

print(f"Binary classification metrics (test set):")
y_pred_baseline = baseline.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred_baseline)
precision = precision_score(y_test, y_pred_baseline)
recall = recall_score(y_test, y_pred_baseline)
f1 = f1_score(y_test, y_pred_baseline)

print(f"  Accuracy:  {accuracy:.4f} (correct predictions / total)")
print(f"  Precision: {precision:.4f} (true positives / predicted positives)")
print(f"  Recall:    {recall:.4f} (true positives / actual positives)")
print(f"  F1 Score:  {f1:.4f} (harmonic mean of precision and recall)")

# Confusion matrix
# WHY: See true positives, true negatives, false positives, false negatives.
cm = confusion_matrix(y_test, y_pred_baseline)
print(f"\nConfusion Matrix:")
print(f"  True Negatives: {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives: {cm[1,1]}")

# Probability predictions
# WHY: Threshold tuning may improve performance.
y_proba = baseline.predict_proba(X_test_scaled)[:, 1]
print(f"\nProbability distribution:")
print(f"  Min probability: {y_proba.min():.4f}")
print(f"  Max probability: {y_proba.max():.4f}")
print(f"  Mean probability: {y_proba.mean():.4f}")

# ---
# STEP 7: TRY ADVANCED MODEL
# ---

# WHY: Compare with Random Forest (typically better for structured data).

print("\n" + "="*60)
print("\n=== STEP 7: ADVANCED MODEL ===")

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)  # Note: RF doesn't need scaling

y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

print(f"Random Forest performance:")
print(f"  Train accuracy: {accuracy_score(y_train, y_rf_train_pred):.4f}")
print(f"  Test accuracy: {accuracy_score(y_test, y_rf_test_pred):.4f}")
print(f"  Train/Test gap: {accuracy_score(y_train, y_rf_train_pred) - accuracy_score(y_test, y_rf_test_pred):.4f}")

# Compare models
print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 56)
lr_prec = precision_score(y_test, baseline.predict(X_test_scaled))
lr_rec = recall_score(y_test, baseline.predict(X_test_scaled))
rf_prec = precision_score(y_test, y_rf_test_pred)
rf_rec = recall_score(y_test, y_rf_test_pred)

print(f"{'Logistic Regression':<20} {accuracy_score(y_test, baseline.predict(X_test_scaled)):<12.4f} {lr_prec:<12.4f} {lr_rec:<12.4f}")
print(f"{'Random Forest':<20} {accuracy_score(y_test, y_rf_test_pred):<12.4f} {rf_prec:<12.4f} {rf_rec:<12.4f}")

# Feature importance (RF)
# WHY: Understand which features matter.
feature_importance = rf.feature_importances_
top_5_features = np.argsort(feature_importance)[::-1][:5]
print(f"\nTop 5 features (Random Forest):")
for idx in top_5_features:
    print(f"  {data.feature_names[idx]}: {feature_importance[idx]:.4f}")

# ---
# STEP 8: HYPERPARAMETER TUNING
# ---

# WHY: Systematic search for best hyperparameters via CV.

print("\n" + "="*60)
print("\n=== STEP 8: HYPERPARAMETER TUNING ===")

from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'n_estimators': [50, 100, 200]
}

# WHY: GridSearchCV: try all combinations, use cross-validation.
print(f"Grid search: {np.prod([len(v) for v in param_grid_rf.values()])} parameter combinations")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='f1',  # Optimize for F1 (balanced metric)
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Evaluate best model
best_rf = grid_search.best_estimator_
best_test_acc = best_rf.score(X_test, y_test)
print(f"Best model test accuracy: {best_test_acc:.4f}")

# ---
# STEP 9: FINAL EVALUATION ON HOLDOUT TEST SET
# ---

# WHY: Report metrics on unseen test data (never used in tuning).

print("\n" + "="*60)
print("\n=== STEP 9: FINAL EVALUATION ===")

y_final_pred = best_rf.predict(X_test)
y_final_proba = best_rf.predict_proba(X_test)

final_accuracy = accuracy_score(y_test, y_final_pred)
final_precision = precision_score(y_test, y_final_pred)
final_recall = recall_score(y_test, y_final_pred)
final_f1 = f1_score(y_test, y_final_pred)

print(f"Final Model Performance (Test Set):")
print(f"  Accuracy:  {final_accuracy:.4f}")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall:    {final_recall:.4f}")
print(f"  F1 Score:  {final_f1:.4f}")

# Confusion matrix for final model
cm_final = confusion_matrix(y_test, y_final_pred)
print(f"\nFinal Confusion Matrix:")
print(f"  TN: {cm_final[0,0]}, FP: {cm_final[0,1]}")
print(f"  FN: {cm_final[1,0]}, TP: {cm_final[1,1]}")

# ---
# STEP 10: SAVE FOR DEPLOYMENT
# ---

# WHY: Serialize model for production use.

print("\n" + "="*60)
print("\n=== STEP 10: SAVE MODEL ===")

import pickle
model_path = "c:\\Users\\SOURAV\\Desktop\\Pybasics\\15_ml_workflow\\best_model.pkl"
scaler_path = "c:\\Users\\SOURAV\\Desktop\\Pybasics\\15_ml_workflow\\scaler.pkl"

# Save model and scaler
pickle.dump(best_rf, open(model_path, 'wb'))
pickle.dump(scaler, open(scaler_path, 'wb'))
print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")

# To load and make predictions on new data:
# loaded_model = pickle.load(open(model_path, 'rb'))
# loaded_scaler = pickle.load(open(scaler_path, 'rb'))
# new_data_scaled = loaded_scaler.transform(new_data)
# predictions = loaded_model.predict(new_data_scaled)

# KEY TAKEAWAY:
# ML Pipeline: Load → Explore → Split → Preprocess → Train → Evaluate → Tune → Deploy.
# Always split before any preprocessing (prevents data leakage).
# Fit scaler/feature transformers on TRAINING data only.
# Use cross-validation for robust performance estimates.
# Compare multiple models; baseline guides expectation.
# Optimize for problem-appropriate metric (accuracy, F1, precision, recall).
# Save model and preprocessor for deployment.
# Document data shape, preprocessing, hyperparameters for reproducibility.
