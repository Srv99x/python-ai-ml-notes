# Revision Notes:
# Topic: NumPy Array Operations and Broadcasting
# Why it matters for AI/ML: Element-wise operations on arrays are far faster than loops.
# Broadcasting allows operations between arrays of different shapes, avoiding explicit loops.
# This is critical for efficient numerical computation in machine learning algorithms.

import numpy as np

# ---
# ELEMENT-WISE OPERATIONS
# ---

# WHY: Vectorized operations are orders of magnitude faster than Python loops.
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

# Arithmetic operations
# WHY: These are immediately applied to every element without explicit loops.
sum_arr = arr1 + arr2  # [11, 22, 33, 44]
diff_arr = arr1 - arr2  # [-9, -18, -27, -36]
prod_arr = arr1 * arr2  # [10, 40, 90, 160]
div_arr = arr2 / arr1  # [10.0, 10.0, 10.0, 10.0]

print("Addition:", sum_arr)
print("Multiplication:", prod_arr)

# Operations with scalars
# WHY: Broadcasting a scalar to all elements enables quick transformations.
scaled = arr1 * 5  # [5, 10, 15, 20]
shifted = arr1 + 100  # [101, 102, 103, 104]
print("\nScaled by 5:", scaled)

# Mathematical functions applied element-wise
# WHY: NumPy provides vectorized math functions instead of looping with math.sqrt(), etc.
arr_float = np.array([1.0, 4.0, 9.0, 16.0])
sqrt_arr = np.sqrt(arr_float)  # [1.0, 2.0, 3.0, 4.0]
exp_arr = np.exp(arr1)  # Element-wise exponential
sin_arr = np.sin(np.linspace(0, np.pi, 4))  # Sine of 4 evenly-spaced angles
print("\nSquare root:", sqrt_arr)

# ---
# BROADCASTING
# ---

# WHY: Broadcasting lets you combine arrays of different shapes without explicit loops.
# This is essential for operations like adding a bias vector to each column of a matrix.

# Example 1: Adding a 1D array to a 2D array
# WHY: In ML, this models adding bias terms or normalizing features column-wise.
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
bias = np.array([10, 20, 30])  # Shape (3,)

# Broadcasting: bias is stretched to match matrix shape
# [10, 20, 30] becomes [[10, 20, 30],
#                       [10, 20, 30],
#                       [10, 20, 30]]
result = matrix + bias
print("\nMatrix + bias (broadcasting):")
print(result)

# Example 2: Matrix operations with shape mismatch
# WHY: Normalizing a dataset by subtracting mean from each sample.
data = np.array([[10, 20, 30],
                 [40, 50, 60]])
means = np.mean(data, axis=0)  # Mean of each column → [25, 35, 45]
centered = data - means  # Broadcasting subtracts mean column-wise
print("\nCentered data (subtract column means):")
print(centered)

# ---
# ROW-WISE AND COLUMN-WISE OPERATIONS
# ---

# WHY: Machine learning often requires aggregating data along rows (across features)
# or columns (across samples).

arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Sum along axis
# WHY: axis=0 → sum each column (across samples); axis=1 → sum each row (across features)
col_sums = np.sum(arr2d, axis=0)  # [12, 15, 18] (sum down each column)
row_sums = np.sum(arr2d, axis=1)  # [6, 15, 24] (sum across each row)
print("\nColumn sums:", col_sums)
print("Row sums:", row_sums)

# Mean and std along axis
# WHY: These are used for feature normalization and scaling in ML preprocessing.
col_means = np.mean(arr2d, axis=0)  # [4.0, 5.0, 6.0]
row_stds = np.std(arr2d, axis=1)  # Standard deviation of each row
print("\nColumn means:", col_means)
print("Row stds:", row_stds)

# Max and min along axis
# WHY: Finding extremes is used in min-max scaling and data validation.
col_max = np.max(arr2d, axis=0)  # [7, 8, 9]
row_min = np.min(arr2d, axis=1)  # [1, 4, 7]
print("\nColumn maxes:", col_max)
print("Row mins:", row_min)

# ---
# DOT PRODUCT AND MATRIX MULTIPLICATION
# ---

# WHY: Dot products are core to neural networks, least-squares regression, and all linear algebra.

vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

# Dot product of two 1D arrays
# WHY: Computes similarity, weighted sums, and linear combinations efficiently.
dot_prod = np.dot(vec1, vec2)  # 1*4 + 2*5 + 3*6 = 32
print("\nDot product:", dot_prod)

# Matrix multiplication
# WHY: The core operation in neural networks and linear models.
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
mat_prod = np.dot(mat1, mat2)  # or mat1 @ mat2
print("\nMatrix product:\n", mat_prod)

# ---
# COMPARISON AND LOGICAL OPERATIONS
# ---

# WHY: Essential for filtering data and creating boolean masks in data preprocessing.

arr = np.array([1, 2, 3, 4, 5])

# Element-wise comparisons return boolean arrays
# WHY: Use these masks to filter samples or features.
mask = arr > 3  # [False, False, False, True, True]
filtered = arr[mask]  # [4, 5] (keep only values > 3)
print("\nBoolean mask (arr > 3):", mask)
print("Filtered array:", filtered)

# Logical operations
# WHY: Combine conditions for more complex filtering rules.
condition = (arr > 2) & (arr < 5)  # True for 3 and 4 only
filtered2 = arr[condition]
print("\nElements in range (2, 5):", filtered2)

# KEY TAKEAWAY:
# Element-wise operations and broadcasting are the foundation of NumPy's speed.
# Use axis parameter to aggregate rows/columns; use broadcasting to avoid loops.
# Dot products and matrix multiplication are essential for all ML algorithms.
# Boolean masking allows efficient data filtering without explicit loops.
