# Revision Notes:
# Topic: NumPy Useful Functions for ML
# Why it matters for AI/ML: These functions are used constantly for data exploration,
# selection, and transformation in machine learning pipelines. Understanding them
# avoids reinventing the wheel and enables efficient numerical computing.

import numpy as np

# ---
# AGGREGATION FUNCTIONS
# ---

# WHY: Summarize data across dimensions for statistics and data validation.

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Basic statistics
# WHY: Understand data distribution and detect anomalies.
print("Sum:", np.sum(arr))  # 31
print("Mean:", np.mean(arr))  # 3.875
print("Median:", np.median(arr))  # 3.5
print("Std Dev:", np.std(arr))  # ~2.93
print("Variance:", np.variance(arr))
print("Min:", np.min(arr))  # 1
print("Max:", np.max(arr))  # 9

# Percentiles
# WHY: Understand data spread and outliers.
p25 = np.percentile(arr, 25)  # 25th percentile
p75 = np.percentile(arr, 75)  # 75th percentile
print(f"\n25th-75th percentile: {p25} - {p75}")

# For 2D data, use axis parameter
# WHY: Compute statistics per row or column (per sample or feature).
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
col_means = np.mean(matrix, axis=0)  # Mean of each column
row_max = np.max(matrix, axis=1)  # Max of each row
print("\nColumn means:", col_means)
print("Row maxes:", row_max)

# ---
# ARGMAX AND ARGMIN
# ---

# WHY: Find indices of maximum/minimum values for classification predictions or debugging.

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# argmax returns index of maximum value
# WHY: In classification, argmax gives the predicted class.
max_idx = np.argmax(arr)
min_idx = np.argmin(arr)
print("\nIndex of max value (9):", max_idx)  # 5
print("Index of min value (1):", min_idx)  # 1 (first occurrence)

# argmax on 2D array
# WHY: Find class predictions or feature importance rankings.
probabilities = np.array([[0.1, 0.7, 0.2],
                          [0.4, 0.3, 0.3]])
predicted_classes = np.argmax(probabilities, axis=1)  # Class with highest probability
print("\nPredicted classes:", predicted_classes)  # [1, 0]

# ---
# WHERE FUNCTION
# ---

# WHY: Conditional selection without explicit loops. Replaces if-else logic.

arr = np.array([1, 2, 3, 4, 5, 6])

# Basic where: select from two arrays based on condition
# WHY: Replace values conditionally (e.g., clip outliers, apply rules).
result = np.where(arr > 3, arr * 10, arr)  # If > 3, multiply by 10; else keep
print("\nWhere function (double if > 3):", result)  # [1, 2, 3, 40, 50, 60]

# where for indexing
# WHY: Find indices where condition is true.
indices = np.where(arr > 3)[0]
print("Indices where arr > 3:", indices)  # [3, 4, 5]

# Nested where for multi-condition logic
# WHY: Categorize or transform data based on multiple conditions.
categories = np.where(arr < 3, 'low',
                      np.where(arr < 5, 'medium', 'high'))
print("\nCategorized:", categories)  # ['low' 'low' 'medium' 'medium' 'high' 'high']

# ---
# LINSPACE, ARANGE, LOGSPACE
# ---

# WHY: Generate sequences for feature engineering, testing, or visualization.

# linspace: evenly-spaced values over interval
# WHY: Create features like polynomial features or time steps.
lin = np.linspace(0, 10, 5)  # 5 points from 0 to 10
print("\nLinspace(0, 10, 5):", lin)  # [0.0, 2.5, 5.0, 7.5, 10.0]

# arange: range with step size
# WHY: Create integer sequences or control step precision.
arr_range = np.arange(0, 10, 0.5)  # Supports float steps (linspace can't)
print("Arange(0, 10, 0.5):", arr_range)

# logspace: log-spaced values (base 10 default)
# WHY: Generate parameters for exponential or logarithmic features.
log_space = np.logspace(0, 2, 5)  # 5 points from 10^0 to 10^2
print("Logspace(0, 2, 5):", log_space)  # [1, 3.16, 10, 31.6, 100]

# ---
# UNIQUE, SORT, ARGSORT
# ---

# WHY: Data validation, handling duplicates, and ranking implementations.

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])

# Unique: find distinct values
# WHY: Identify classes, validate categorical features, remove duplicates.
unique_vals, counts = np.unique(arr, return_counts=True)
print("\nUnique values:", unique_vals)  # [1, 2, 3, 4, 5, 6, 9]
print("Counts:", counts)  # [2, 1, 1, 1, 2, 1, 1]

# Sort array
# WHY: Order data for analysis or algorithm prerequisites.
sorted_arr = np.sort(arr)
print("\nSorted array:", sorted_arr)

# argsort: indices that would sort array
# WHY: Rank-based operations or sorting by one array while tracking original indices.
sort_indices = np.argsort(arr)
print("Indices for sorting:", sort_indices)
print("Verifying:", arr[sort_indices])  # Sorted array

# ---
# REPEAT AND TILE
# ---

# WHY: Duplicate data for broadcasting or creating augmented datasets.

arr = np.array([1, 2, 3])

# Repeat: repeat each element
# WHY: Expand samples for data augmentation or upsampling.
repeated = np.repeat(arr, 3)  # Each element repeated 3 times
print("\nRepeat each element 3x:", repeated)  # [1, 1, 1, 2, 2, 2, 3, 3, 3]

# Tile: repeat entire array
# WHY: Duplicate entire dataset for batching or concatenation.
tiled = np.tile(arr, 2)  # Repeat entire array 2 times
print("Tile array 2x:", tiled)  # [1, 2, 3, 1, 2, 3]

# Practical: create batch by tiling
batch = np.tile(arr, (4, 1))  # Repeat horizontally 1x, vertically 4x
print("\nBatch (4 copies vertically):", batch.shape)  # (4, 3)

# ---
# ALLCLOSE AND ISCLOSE
# ---

# WHY: Compare floating-point arrays accounting for numerical precision errors.

arr1 = np.array([1.0, 2.0, 3.0])
arr2 = np.array([1.0000001, 2.0, 3.0000005])

# allclose: check if all elements are close (within tolerance)
# WHY: Validate ML model outputs and test predictions.
is_close = np.allclose(arr1, arr2, rtol=1e-5)  # rtol = relative tolerance
print("\nArrays close (rtol=1e-5)?", is_close)  # True

# isclose: element-wise comparison
# WHY: Identify which items differ slightly.
element_wise = np.isclose(arr1, arr2, rtol=1e-5)
print("Element-wise close:", element_wise)  # [True, True, True]

# ---
# RANDOM FUNCTIONS
# ---

# WHY: Generate random numbers for initialization, sampling, and experimentation.

# Random uniform [0, 1)
# WHY: Initialize neural network weights or create synthetic test data.
rand_uniform = np.random.rand(3, 2)  # 3x2 matrix of random values
print("\nRandom uniform (3x2):\n", rand_uniform)

# Random from normal distribution
# WHY: Initialize weights with appropriate scale or model natural phenomena.
rand_normal = np.random.randn(3, 2) * 0.01  # Scale for NN initialization
print("\nRandom normal * 0.01 (3x2):\n", rand_normal)

# Random integers
# WHY: Create train-test splits, sample batches, or generate discrete features.
rand_int = np.random.randint(0, 10, size=5)  # 5 random integers [0, 10)
print("\nRandom integers [0,10):", rand_int)

# Random choice (sampling with/without replacement)
# WHY: Create mini-batches or stratified sampling.
choices = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)  # No replacement
print("Random choice (no replace):", choices)

# Seed for reproducibility
# WHY: Reproduce results, share code with consistent initialization.
np.random.seed(42)
print("\nAfter seed(42), first random:", np.random.rand())

# KEY TAKEAWAY:
# argmax/argmin find indices of extreme values (used in classification).
# where enables conditional selection without loops.
# linspace and logspace generate sequences for features.
# unique, sort, argsort enable data validation and ranking.
# Random functions initialize models and create synthetic data.
# These functions replace common loops with vectorized operations.
