# Revision Notes:
# Topic: NumPy Reshaping, Stacking, and Splitting
# Why it matters for AI/ML: Data rarely arrives in the shape your model expects.
# Reshaping, stacking, and splitting are essential for preparing data for ML pipelines.
# These operations are performed constantly when preprocessing datasets.

import numpy as np

# ---
# RESHAPING ARRAYS
# ---

# WHY: Reshape converts data between different dimensions without copying values.
# This is essential for converting flat data into matrix form for ML algorithms.

# Create a 1D array and reshape it into 2D
arr_flat = np.array([1, 2, 3, 4, 5, 6])
arr_2d = arr_flat.reshape(2, 3)  # Convert 6 elements into 2x3 matrix
print("Original 1D:", arr_flat)
print("Reshaped to 2x3:\n", arr_2d)

# Reshape with -1 (automatic dimension inference)
# WHY: When you know total elements but not one dimension, use -1 to auto-compute.
arr_3d = arr_flat.reshape(2, -1)  # -1 → infer as 3 (since 2*? = 6)
arr_3d_alt = arr_flat.reshape(-1, 3)  # -1 → infer as 2
print("\nReshaped with -1 (2, -1):\n", arr_3d)

# Flatten and ravel
# WHY: Convert multidimensional arrays to 1D for certain operations.
matrix = np.array([[1, 2, 3], [4, 5, 6]])
flattened = matrix.flatten()  # Creates a copy
raveled = matrix.ravel()  # May use original data (more efficient)
print("\nFlattened 2D array:", flattened)

# Reshape for batch processing
# WHY: Neural networks often expect (batch_size, features) format.
batch_data = np.arange(12)
batch_reshaped = batch_data.reshape(4, 3)  # 4 samples, 3 features each
print("\nBatch data (4 samples, 3 features):\n", batch_reshaped)

# ---
# STACKING ARRAYS
# ---

# WHY: Combine multiple arrays along a new or existing axis.
# This is common when concatenating feature sets or combining datasets.

arr_a = np.array([1, 2, 3])
arr_b = np.array([4, 5, 6])

# Vertical stack (row-wise concatenation)
# WHY: Combine multiple samples into a single dataset.
vstacked = np.vstack([arr_a, arr_b])  # Stack as rows
print("\nVertical stack:\n", vstacked)

# Horizontal stack (column-wise concatenation)
# WHY: Combine multiple feature sets into one matrix.
hstacked = np.hstack([arr_a, arr_b])  # Stack as [1, 2, 3, 4, 5, 6]
print("\nHorizontal stack:", hstacked)

# Stack along new axis
# WHY: Create 3D structures for batch operations on multiple matrices.
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
stacked = np.stack([mat1, mat2], axis=0)  # Stack along new first dimension
print("\nStacked matrices (shape 2, 2, 2):\n", stacked)

# Concatenate along existing axis
# WHY: More flexible than vstack/hstack for arbitrary axis specification.
concat_result = np.concatenate([vstacked, vstacked], axis=0)  # Combine rows
print("\nConcatenated along axis 0 (rows):\n", concat_result)

# ---
# SPLITTING ARRAYS
# ---

# WHY: Split data into train/test or stratified subsets for model evaluation.

arr = np.arange(10)  # [0, 1, 2, ..., 9]

# Split into equal parts
# WHY: Divide dataset into k-folds for cross-validation.
splits = np.array_split(arr, 3)  # Split into 3 roughly equal parts
print("\nArray split into 3 parts:")
for i, part in enumerate(splits):
    print(f"  Part {i}: {part}")

# Horizontal split for feature sets
# WHY: Separate input features (X) from target (y).
data = np.arange(20).reshape(5, 4)  # 5 samples, 4 features
X, y = np.hsplit(data, [3])  # Split after column 2 (first 3 cols vs last col)
print("\nHorizontal split (X and y):")
print("Features (X):\n", X)
print("Target (y):\n", y)

# Manual train-test split
# WHY: Use indices to create reproducible splits for ML evaluation.
n_samples = arr.shape[0]
train_size = int(0.7 * n_samples)  # 70% train, 30% test

train_data = arr[:train_size]
test_data = arr[train_size:]
print("\nTrain-test split (70-30):")
print("Train:", train_data)
print("Test:", test_data)

# ---
# TRANSPOSE
# ---

# WHY: Transpose converts (n_samples, features) ↔ (features, n_samples).
# Used for dimension manipulation in matrix operations.

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
transposed = matrix.T
print("\nOriginal matrix shape:", matrix.shape)
print("Transposed shape:", transposed.shape)
print("Transposed:\n", transposed)

# Transpose as axis rearrangement
# WHY: Reorder dimensions in multidimensional arrays for broadcasting.
arr_3d = np.arange(24).reshape(2, 3, 4)
rearranged = np.transpose(arr_3d, (2, 0, 1))  # Move dimension axes
print("\nOriginal 3D shape:", arr_3d.shape)
print("Rearranged shape:", rearranged.shape)

# ---
# SQUEEZE AND EXPAND DIMENSIONS
# ---

# WHY: Remove singleton dimensions or add new dimensions for compatibility.

# Squeeze (remove dimensions of size 1)
# WHY: Clean up arrays after operations that create unnecessary singleton dims.
arr_squeezable = np.array([[[1, 2, 3]]])  # Shape (1, 1, 3)
squeezed = arr_squeezable.squeeze()  # Shape (3,)
print("\nBefore squeeze shape:", arr_squeezable.shape)
print("After squeeze shape:", squeezed.shape)

# Expand dimensions
# WHY: Add dimensions to make arrays compatible for broadcasting.
arr_1d = np.array([1, 2, 3])
expanded = np.expand_dims(arr_1d, axis=0)  # Add dimension at beginning: shape (1, 3)
print("\nBefore expand_dims shape:", arr_1d.shape)
print("After expand_dims shape:", expanded.shape)

# KEY TAKEAWAY:
# reshape() rearranges data into new shapes; -1 auto-infers dimensions.
# vstack/hstack/concatenate combine arrays; array_split divides them.
# Transpose swaps axes; squeeze/expand_dims modify singleton dimensions.
# These operations are fundamental for preparing data in ML pipelines.
