# Revision Notes:
# Topic: Scikit-Learn Pipeline for Reproducible Workflows
# Why it matters for AI/ML: Pipelines automate preprocessing and model training steps.
# Prevents data leakage by ensuring transformers fit only on training data.
# Ensures consistent preprocessing during development and deployment.

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cross_val_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("SKLEARN PIPELINES FOR REPRODUCIBLE ML")
print("="*60)

# ---
# PROBLEM: WITHOUT PIPELINE (DATA LEAKAGE RISK)
# ---

# WHY: Manual steps risk fitting on test data or applying steps in wrong order.

print("\n=== Problem: Without Pipeline ===")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Manual preprocessing (RISKY!)
# WHY: Easy to accidentally fit on test data.
scaler_bad = StandardScaler()
scaler_bad.fit(X_test)  # WRONG! Fitting on TEST data
X_train_bad = scaler_bad.transform(X_train)  # Using test statistics on training data
X_test_bad = scaler_bad.transform(X_test)

print(f"Manual scaling (WRONG approach):")
print(f"  Scaler fit on TEST data → DATA LEAKAGE")
print(f"  Train set mean: {X_train_bad.mean():.4f}")
print(f"  Test set mean: {X_test_bad.mean():.4f}")

# ---
# SOLUTION 1: SIMPLE PIPELINE
# ---

# WHY: Automatically handle preprocessing in correct order, fit on train only.

print("\n" + "="*60)
print("\n=== Solution 1: Simple Pipeline ===")

# Define pipeline: scaler → model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

print(f"Pipeline steps:")
print(f"  1. StandardScaler (fit on train, transform train+test)")
print(f"  2. LogisticRegression (train on scaled training data)")

# Fit pipeline (automatic scaling, then model training)
# WHY: fit() applies all transformers to training data.
pipeline.fit(X_train, y_train)

# Predict (automatic scaling applied first)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nResults:")
print(f"  Accuracy: {accuracy:.4f}")

# Access individual components
# WHY: Extract trained scaler or model for inspection.
fitted_scaler = pipeline.named_steps['scaler']
print(f"  Scaler mean: {fitted_scaler.mean_[:3]}")  # First 3 features

fitted_model = pipeline.named_steps['model']
print(f"  Model coefficients shape: {fitted_model.coef_.shape}")

# Predict with probabilities
y_proba = pipeline.predict_proba(X_test)
print(f"  Probability shape: {y_proba.shape}")

# ---
# CROSS-VALIDATION WITH PIPELINE
# ---

# WHY: CV correctly applies transformations for each fold.

print("\n" + "="*60)
print("\n=== Cross-Validation with Pipeline ===")

# WHY: Each fold: train transformers on train fold, apply to validation fold.
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

print(f"5-fold Cross-validation scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean: {cv_scores.mean():.4f}")
print(f"  Std: {cv_scores.std():.4f}")

# ---
# HYPERPARAMETER TUNING WITH PIPELINE
# ---

# WHY: GridSearchCV works with pipelines; search over all steps.

print("\n" + "="*60)
print("\n=== Hyperparameter Tuning ===")

pipeline_tunable = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# Define grid: prefix step name with '__'
# WHY: 'model__C' refers to C parameter of LogisticRegression step.
param_grid = {
    'model__C': [0.01, 0.1, 1.0, 10.0],
    'model__solver': ['lbfgs', 'liblinear']
}

print(f"GridSearchCV parameters: {param_grid}")

grid_search = GridSearchCV(pipeline_tunable, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {grid_search.score(X_test, y_test):.4f}")

# ---
# FEATURE ENGINEERING IN PIPELINE
# ---

# WHY: Transform features before model without manual steps.

print("\n" + "="*60)
print("\n=== Feature Engineering in Pipeline ===")

# Add polynomial features
pipeline_poly = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LogisticRegression(max_iter=2000, random_state=42))
])

pipeline_poly.fit(X_train, y_train)
accuracy_poly = pipeline_poly.score(X_test, y_test)

print(f"Original features: {X_train.shape[1]}")
print(f"With polynomial features (degree=2):")
poly_features = pipeline_poly.named_steps['poly']
print(f"  New features: {poly_features.n_output_features_}")
print(f"  Accuracy: {accuracy_poly:.4f}")

# Feature selection in pipeline
# WHY: Automatic feature selection before model training.
pipeline_selection = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=2)),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

pipeline_selection.fit(X_train, y_train)
accuracy_selection = pipeline_selection.score(X_test, y_test)

print(f"\nWith feature selection (k=2):")
print(f"  Accuracy: {accuracy_selection:.4f}")

# ---
# COLUMNTRANSFORMER: DIFFERENT PREPROCESSING PER FEATURE
# ---

# WHY: Different features may need different preprocessing (e.g., numeric vs categorical).

print("\n" + "="*60)
print("\n=== ColumnTransformer: Feature-Specific Preprocessing ===")

# Create dataset with mixed types
df = pd.DataFrame({
    'age': [25, 45, 35, 55, 30],
    'income': [30000, 120000, 45000, 180000, 55000],
    'city': ['NYC', 'LA', 'NYC', 'NYC', 'LA']
})
y_mixed = np.array([0, 1, 0, 1, 0])

print(f"\nMixed data types:")
print(f"  Numeric: age, income")
print(f"  Categorical: city")
print(df)

# Define transformations per feature type
preprocessor = ColumnTransformer([
    ('numeric', StandardScaler(), ['age', 'income']),
    ('categorical', OneHotEncoder(drop='first', sparse_output=False), ['city'])
])

# Create full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

print(f"\nColumnTransformer pipeline:")
print(f"  Numeric features → StandardScaler")
print(f"  Categorical features → OneHotEncoder")

X_mixed = df[['age', 'income', 'city']]
full_pipeline.fit(X_mixed, y_mixed)
accuracy_mixed = full_pipeline.score(X_mixed, y_mixed)
print(f"  Model accuracy: {accuracy_mixed:.4f}")

# ---
# FEATUREUNION: COMBINE MULTIPLE FEATURE TRANSFORMATIONS
# ---

# WHY: Create multiple feature sets and concatenate them.

print("\n" + "="*60)
print("\n=== FeatureUnion: Multiple Feature Extractors ===")

# Load simpler dataset
X_simple = iris.data
y_simple = iris.target

# FeatureUnion: apply both scaling and polynomial features, combine results
union = FeatureUnion([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

# Note: This creates union with duplicated features (not typical)
# More realistic: use Pipeline → FeatureUnion inside ColumnTransformer

pipeline_union = Pipeline([
    ('features', SelectKBest(f_classif, k=2)),  # Select 2 features
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

pipeline_union.fit(X_train_simple, y_train_simple)
accuracy_union = pipeline_union.score(X_test_simple, y_test_simple)

print(f"Pipeline with feature selection:")
print(f"  Accuracy: {accuracy_union:.4f}")

# ---
# PIPELINE WITH RANDOM FOREST
# ---

# WHY: RandomForest doesn't need scaling but benefits from feature engineering.

print("\n" + "="*60)
print("\n=== Pipeline: Random Forest ===")

pipeline_rf = Pipeline([
    ('feature_selection', SelectKBest(f_classif, k=3)),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_rf.fit(X_train_simple, y_train_simple)
accuracy_rf = pipeline_rf.score(X_test_simple, y_test_simple)

print(f"Random Forest pipeline (with feature selection):")
print(f"  Accuracy: {accuracy_rf:.4f}")
print(f"  Note: Scaling not needed for tree-based models")

# ---
# SAVING AND LOADING PIPELINES
# ---

# WHY: Serialize entire pipeline for deployment.

print("\n" + "="*60)
print("\n=== Saving and Loading Pipelines ===")

import pickle

# Save pipeline
pipeline_save = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
pipeline_save.fit(X_train, y_train)

save_path = 'c:\\Users\\SOURAV\\Desktop\\Pybasics\\15_ml_workflow\\pipeline.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(pipeline_save, f)

print(f"Pipeline saved to: {save_path}")

# Load and use pipeline
with open(save_path, 'rb') as f:
    loaded_pipeline = pickle.load(f)

predictions_loaded = loaded_pipeline.predict(X_test)
accuracy_loaded = accuracy_score(y_test, predictions_loaded)
print(f"Loaded pipeline accuracy: {accuracy_loaded:.4f}")

# ---
# PIPELINE SUMMARY AND BEST PRACTICES
# ---

print("\n" + "="*60)
print("\n=== Pipeline Best Practices ===")

print(f"\n1. PREVENT DATA LEAKAGE:")
print(f"   - Use Pipeline to fit transformers on training data only")
print(f"   - Pipeline.fit() automatically handles train/test splits correctly")

print(f"\n2. REPRODUCIBILITY:")
print(f"   - Identical preprocessing every time")
print(f"   - Same transformations in dev and production")

print(f"\n3. CROSS-VALIDATION:")
print(f"   - cross_val_score(pipeline, X, y) applies preprocessing per fold")
print(f"   - Each fold: train transformers on train subset, apply to val subset")

print(f"\n4. HYPERPARAMETER TUNING:")
print(f"   - GridSearchCV searches over all pipeline steps")
print(f"   - Use 'step__param' notation to reference parameters")

print(f"\n5. PRODUCTION DEPLOYMENT:")
print(f"   - Save entire pipeline (transformers + model)")
print(f"   - Load and apply to new data: pipeline.predict(new_data)")

print(f"\n6. CUSTOM TRANSFORMERS:")
print(f"   - Inherit from BaseEstimator and TransformerMixin")
print(f"   - Implement fit(X, y) and transform(X) methods")

# example_transformer.fit(X_train, y_train)
# example_transformer.transform(X_test)

# KEY TAKEAWAY:
# Pipeline: Automate preprocessing → ensure correct order → prevent data leakage.
# Steps: transformers (fit_transform) → final estimator (fit).
# GridSearchCV: search parameters using 'step__param' notation.
# ColumnTransformer: different preprocessing for different feature types.
# cross_val_score: automatically handles preprocessing per fold.
# Save pipeline with pickle for reproducible deployment.
# Never manually scale or preprocess separately; use Pipeline.
