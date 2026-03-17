# Revision Notes:
# Topic: Train-Test Split in scikit-learn
# Why it matters for AI/ML: Proper dataset splitting is critical for honest model evaluation.
# Data leakage occurs if test data influences training; stratified splits ensure class distribution.
# Understanding splitting techniques prevents overfitting and ensures reproducible results.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
from sklearn.datasets import load_iris, load_breast_cancer

np.random.seed(42)

# ---
# BASIC TRAIN-TEST SPLIT
# ---

# WHY: Divide data into training (learn patterns) and testing (evaluate performance).

# Load sample dataset
iris = load_iris()
X = iris.data  # Features (150 samples, 4 features)
y = iris.target  # Target (3 classes: 0, 1, 2)

print("=== Train-Test Split ===")
print(f"Total samples: {len(X)}")

# Split into 80% train, 20% test
# WHY: 80-20 is common; more training data when dataset is large.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# WHY random_state: ensures reproducibility across runs.
print(f"\nWith random_state=42, same split every time")
X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Same first sample? {np.array_equal(X_train[0], X_train2[0])}")

# ---
# STRATIFIED SPLIT
# ---

# WHY: Maintain class distribution in train and test sets.
# Critical for imbalanced datasets.

# Example: imbalanced binary classification
from sklearn.datasets import make_classification
X_imbal, y_imbal = make_classification(n_samples=1000, n_features=10, n_classes=2,
                                       weights=[0.9, 0.1], random_state=42)

print("\n" + "="*60)
print("\n=== Stratified Split for Imbalanced Data ===")
print(f"Original class distribution: {np.bincount(y_imbal)}")
print(f"Proportions: Class 0: {(y_imbal==0).sum()/len(y_imbal)*100:.1f}%, Class 1: {(y_imbal==1).sum()/len(y_imbal)*100:.1f}%")

# Non-stratified split (can lose minority class)
# WHY: Random split might not preserve class balance.
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X_imbal, y_imbal, test_size=0.2, random_state=42, stratify=None
)
print(f"\nRandom split - Train: {np.bincount(y_train_random)}")
print(f"Random split - Test: {np.bincount(y_test_random)}")

# Stratified split (preserves class distribution)
# WHY: Ensure minority class is represented in both train and test.
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X_imbal, y_imbal, test_size=0.2, random_state=42, stratify=y_imbal
)
print(f"\nStratified split - Train: {np.bincount(y_train_strat)}")
print(f"Train proportion - Class 1: {(y_train_strat==1).sum()/len(y_train_strat)*100:.1f}%")
print(f"Stratified split - Test: {np.bincount(y_test_strat)}")
print(f"Test proportion - Class 1: {(y_test_strat==1).sum()/len(y_test_strat)*100:.1f}%")

# ---
# CROSS-VALIDATION
# ---

# WHY: Use all data efficiently for training and validation.
# Reduces variance in performance estimate.

print("\n" + "="*60)
print("\n=== Cross-Validation ===")

# K-fold cross-validation
# WHY: Divide into k folds; train on k-1, evaluate on 1 (repeat k times).
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_idx = 1
for train_idx, val_idx in kfold.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    print(f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")
    fold_idx += 1

# Stratified K-fold
# WHY: Stratify across all folds to preserve class distribution.
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"\nStratified 5-fold CV:")
for fold_idx, (train_idx, val_idx) in enumerate(skfold.split(X_imbal, y_imbal), 1):
    print(f"Fold {fold_idx}: {np.bincount(y_imbal[train_idx])}")

# Leave-One-Out Cross-Validation (LOOCV)
# WHY: Most thorough; remove 1 sample, train on n-1, test on 1.
# Expensive but useful for small datasets.
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
print(f"\nLOOCV would create {loo.get_n_splits(X)} folds (one per sample)")
print("Use only for small datasets due to computational cost")

# ---
# TIME SERIES SPLITTING
# ---

# WHY: For time series data, never use future data in training.
# Split sequentially: train on past, test on future.

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

# Create time series-like data
X_ts = np.arange(100).reshape(-1, 1)
print("\n" + "="*60)
print("\n=== Time Series Split ===")
for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_ts), 1):
    print(f"Fold {fold_idx}: train={train_idx.min()}-{train_idx.max()}, test={test_idx.min()}-{test_idx.max()}")

# ---
# HOLDOUT VALIDATION ARCHITECTURE
# ---

# WHY: Standard practice: train split, validation split, test split.

print("\n" + "="*60)
print("\n=== Train-Validation-Test Architecture ===")

cancer = load_breast_cancer()
X_data, y_data = cancer.data, cancer.target

# Step 1: Separate test set (hold out completely)
# WHY: Test set never touches training or hyperparameter tuning.
X_temp, X_test, y_temp, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Step 2: Split remaining into train and validation
# WHY: Train learns patterns; validation tunes hyperparameters.
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"Original: {len(X_data)} samples")
print(f"Train: {len(X_train)} ({len(X_train)/len(X_data)*100:.1f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X_data)*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(X_data)*100:.1f}%)")

# Class distribution check
print(f"\nClass distribution in train: {np.bincount(y_train)}")
print(f"Class distribution in test: {np.bincount(y_test)}")

# ---
# PRACTICAL MACHINE LEARNING SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Small dataset → use cross-validation
# WHY: Every sample matters; splitting wastes data.
print("Small dataset (< 1000 samples):")
print("  → Use k-fold cross-validation (k=5 or 10)")
print("  → Reason: Use all data both for training and validation")

# Scenario 2: Large imbalanced dataset → stratified train-test
# WHY: Preserve rare class in both sets.
print("\nLarge imbalanced dataset:")
print("  → Use stratified train-test split")
print(f"  → Example: 80/10/10 train/val/test with stratification")

# Scenario 3: Time series data → temporal split
# WHY: Prevent data leakage from future.
print("\nTime series data:")
print("  → Use TimeSeriesSplit or custom sequential split")
print("  → Never use future data in training!")

# Scenario 4: Hyperparameter tuning → nested CV
# WHY: Avoid optimizing hyperparameters on test set.
print("\nHyperparameter tuning:")
print("  → Outer loop: cross-validation for performance estimate")
print("  → Inner loop: cross-validation on fold training sets for tuning")

# KEY TAKEAWAY:
# test_size: fraction for test set (common 0.2 or 0.1).
# stratify: preserve class distribution; use for classification and imbalanced data.
# random_state: reproducibility (set to any fixed number).
# K-fold CV: efficient reuse of all data; best for small-medium datasets.
# Train-val-test split: standard architecture for large datasets and hyperparameter tuning.
# TimeSeriesSplit: temporal ordering required for forecasting data.
# Always keep test set completely separate; never use for training or tuning.
