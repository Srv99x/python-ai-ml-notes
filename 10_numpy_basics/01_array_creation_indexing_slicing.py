# Revision Notes:
# Topic: NumPy Array Creation, Indexing, and Slicing
# Why it matters for AI/ML: NumPy is the foundation for all numerical computing in Python.
# Arrays are more efficient than lists and enable vectorized operations critical for data science.
# This is essential for handling datasets and numerical computations at scale.

import numpy as np

# ---
# ARRAY CREATION
# ---

# Create arrays from lists
# WHY: Converting Python lists to NumPy arrays makes them optimized for mathematical operations.
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Create arrays with specific shapes and values
# WHY: Often you need arrays pre-initialized with zeros, ones, or random values for ML models.
zeros_arr = np.zeros((3, 3))  # 3x3 matrix of zeros
ones_arr = np.ones((2, 5))    # 2x5 matrix of ones
random_arr = np.random.rand(4, 3)  # 4x3 matrix of random values [0, 1)

# Create sequences using built-in functions
# WHY: arange and linspace are commonly used to generate feature values or time steps.
seq = np.arange(0, 10, 2)  # Start at 0, stop before 10, step by 2 → [0, 2, 4, 6, 8]
lin_space = np.linspace(0, 1, 5)  # 5 evenly-spaced points from 0 to 1

# Create identity matrix
# WHY: Identity matrices are used in linear algebra and regularization techniques.
identity = np.eye(3)  # 3x3 identity matrix

print("1D Array:", arr_1d)
print("\n2D Array:\n", arr_2d)
print("\nZeros array shape:", zeros_arr.shape)
print("Random array:\n", random_arr)

# ---
# INDEXING AND ACCESSING ELEMENTS
# ---

# 1D array indexing
# WHY: Extract specific values to check or modify individual data points.
first_elem = arr_1d[0]  # First element (0-indexed)
last_elem = arr_1d[-1]  # Last element using negative index
print("\nFirst element:", first_elem)
print("Last element:", last_elem)

# 2D array indexing
# WHY: In ML, you often access rows (samples) or columns (features).
elem_2d = arr_2d[0, 1]  # Row 0, Column 1 → 2
first_row = arr_2d[0]  # Entire first row → [1, 2, 3]
first_col = arr_2d[:, 0]  # All rows, column 0 → [1, 4]
print("\nElement at [0,1]:", elem_2d)
print("First row:", first_row)
print("First column:", first_col)

# ---
# SLICING
# ---

# 1D slicing
# WHY: Extract subsets of data for train/test splits or batch processing.
slice_1d = arr_1d[1:4]  # Elements from index 1 to 3 (not including 4) → [2, 3, 4]
slice_step = arr_1d[::2]  # Every 2nd element → [1, 3, 5]
print("\nSlice [1:4]:", slice_1d)
print("Every 2nd element:", slice_step)

# 2D slicing
# WHY: Select specific rows/columns or rectangular regions from datasets.
slice_2d = arr_2d[0:2, 1:3]  # Rows 0-1, columns 1-2
print("\n2D Slice [0:2, 1:3]:\n", slice_2d)

# Negative indexing in slicing
# WHY: Useful for getting the last N rows/columns without knowing total size.
last_two_rows = arr_2d[-2:, :]  # Last 2 rows, all columns
print("\nLast 2 rows:\n", last_two_rows)

# ---
# ARRAY PROPERTIES
# ---

# WHY: Understanding array shape, size, and dtype is crucial for debugging ML pipelines.
print("\n--- Array Properties ---")
print("Shape of arr_2d:", arr_2d.shape)  # (2, 3) → 2 rows, 3 columns
print("Total elements:", arr_2d.size)  # 6
print("Data type:", arr_2d.dtype)  # int64 (depends on system)
print("Number of dimensions:", arr_2d.ndim)  # 2

# KEY TAKEAWAY:
# NumPy arrays are fixed-size, homogeneous (single data type), and enable fast mathematical operations.
# Indexing uses 0-based positions; slicing uses [start:stop:step] where stop is exclusive.
# Understanding shape and indexing is foundational for reshaping data in ML pipelines.
