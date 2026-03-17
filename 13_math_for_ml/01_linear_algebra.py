# Revision Notes:
# Topic: Linear Algebra with NumPy
# Why it matters for AI/ML: All ML algorithms are built on linear algebra.
# Matrix operations enable efficient computation of neural networks, transformations, and optimizations.
# Understanding these operations is critical for debugging and implementing advanced models.

import numpy as np

# ---
# VECTORS AND BASIC OPERATIONS
# ---

# WHY: Vectors are 1D arrays representing data points or feature values.

# Create vectors
# WHY: Represent samples or single features.
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition and subtraction
# WHY: Combine or compare feature vectors.
v_sum = v1 + v2  # [5, 7, 9]
v_diff = v1 - v2  # [-3, -3, -3]
print("Vector addition:", v_sum)
print("Vector subtraction:", v_diff)

# Scalar multiplication
# WHY: Scale vectors (e.g., learning rate * gradient).
v_scaled = v1 * 2  # [2, 4, 6]
print("Scaled vector:", v_scaled)

# ---
# DOT PRODUCT (INNER PRODUCT)
# ---

# WHY: Measure similarity between vectors; compute predictions in linear models.
# Mathematical: dot(u, v) = u1*v1 + u2*v2 + ... + un*vn

u = np.array([1, 2, 3])
v = np.array([0, 1, 2])

dot_product = np.dot(u, v)
# Or using @ operator
dot_product_alt = u @ v
print("\nDot product:", dot_product)  # 1*0 + 2*1 + 3*2 = 8
print("Dot product (@ operator):", dot_product_alt)

# WHY: In ML, dot products compute:
# - Similarity scores between embeddings
# - Weighted sums (features * weights in linear regression/neural nets)
# - Attention mechanisms in transformers

# ---
# VECTOR NORMS
# ---

# WHY: Measure vector magnitude; essential for normalization and regularization.

v = np.array([3, 4])

# L2 norm (Euclidean distance)
# WHY: Standard distance metric; used in SVM and regularization.
l2_norm = np.linalg.norm(v)  # sqrt(3^2 + 4^2) = 5
print("\nL2 norm:", l2_norm)

# L1 norm (Manhattan distance)
# WHY: Alternative metric; used in L1 regularization.
l1_norm = np.linalg.norm(v, ord=1)  # |3| + |4| = 7
print("L1 norm:", l1_norm)

# Normalize vector to unit length
# WHY: Make vector magnitude = 1 (useful for similarity measures).
v_normalized = v / np.linalg.norm(v)
print("Normalized vector:", v_normalized)

# ---
# MATRICES AND MATRIX OPERATIONS
# ---

# WHY: Matrices represent datasets (rows=samples, cols=features) or transformations.

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Matrix addition and subtraction
# WHY: Element-wise combination of matrices.
C_sum = A + B
print("\n" + "="*60)
print("\nMatrix addition:\n", C_sum)

# Element-wise multiplication
# WHY: Apply activation functions or element-wise transformations.
C_mult = A * B
print("Element-wise multiplication:\n", C_mult)

# ---
# MATRIX MULTIPLICATION
# ---

# WHY: Core operation in neural networks and linear transformations.
# (m x n) * (n x p) = (m x p)

A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])  # 3x2

# Matrix multiplication
# WHY: Apply linear transformation or compute neural network layer.
C = np.dot(A, B)
# Or using @ operator (preferred in modern Python)
C_alt = A @ B
print("\nMatrix multiplication (2x3 * 3x2 = 2x2):\n", C)

# WHY: In neural networks:
# output = input @ weights.T + bias

# ---
# TRANSPOSE
# ---

# WHY: Flip rows and columns; convert (n x m) to (m x n).

A = np.array([[1, 2, 3],
              [4, 5, 6]])
A_T = A.T
print("\n" + "="*60)
print("\nOriginal matrix shape:", A.shape)
print("Transposed matrix shape:", A_T.shape)
print("Transposed:\n", A_T)

# WHY: Used in:
# - Converting between (samples, features) and (features, samples)
# - Computing covariance matrix: (X.T @ X) / n
# - Solving linear systems

# ---
# DETERMINANT AND MATRIX INVERSE
# ---

# WHY: Determinant checks if matrix is invertible; inverse solves linear systems.

A = np.array([[1, 2],
              [3, 4]])

# Determinant
# WHY: Determinant = 0 means matrix is singular (non-invertible).
det_A = np.linalg.det(A)
print("\nDeterminant:", det_A)  # 1*4 - 2*3 = -2

# Matrix inverse
# WHY: Solve Ax = b as x = A^-1 * b
A_inv = np.linalg.inv(A)
print("Inverse:\n", A_inv)

# Verify: A * A^-1 = I (identity)
I = A @ A_inv
print("A * A^-1 (should be identity):\n", I.round())

# ---
# EIGENVALUES AND EIGENVECTORS
# ---

# WHY: Used in PCA for dimensionality reduction and in neural network analysis.
# Av = λv (A: matrix, v: eigenvector, λ: eigenvalue)

A = np.array([[4, -2],
              [-2, 4]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\n" + "="*60)
print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verify: A @ v = λ @ v
v = eigenvectors[:, 0]
lambda_val = eigenvalues[0]
lhs = A @ v
rhs = lambda_val * v
print("Verification (A*v vs λ*v):")
print("A*v:", lhs)
print("λ*v:", rhs)

# ---
# RANK AND MATRIX DECOMPOSITION
# ---

# WHY: Rank indicates independent dimensions; decompositions reveal structure.

A = np.array([[1, 2, 3],
              [2, 4, 6],
              [1, 1, 1]])

# Rank
# WHY: Number of linearly independent rows/columns.
rank = np.linalg.matrix_rank(A)
print("\nMatrix rank:", rank)  # 2 (row 2 is 2*row 1)

# Singular Value Decomposition (SVD)
# WHY: Decomposes matrix into U, Σ, V.T; fundamental for PCA and compression.
U, sigma, V_T = np.linalg.svd(A)
print("\nSVD - Singular values:", sigma)

# Reconstruct: A ≈ U @ diag(sigma) @ V_T
A_reconstructed = U @ np.diag(sigma) @ V_T
print("Reconstruction error:", np.linalg.norm(A - A_reconstructed))

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Linear regression
# WHY: Solve y = X @ w using matrix operations.
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
y = np.array([5, 11, 17])

# Normal equation: w = (X.T @ X)^-1 @ X.T @ y
w = np.linalg.inv(X.T @ X) @ X.T @ y
print("Linear regression weights:", w)

# Predictions
y_pred = X @ w
print("Predictions:", y_pred)
print("Actual:", y)

# Scenario 2: Dimensionality reduction via PCA
# WHY: PCA uses eigenvalues/eigenvectors of covariance matrix.
X = np.random.randn(100, 5)  # 100 samples, 5 features
cov_matrix = (X.T @ X) / X.shape[0]  # Covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (descending)
sorted_idx = np.argsort(eigenvalues)[::-1]
top_components = eigenvectors[:, sorted_idx[:2]]  # Top 2 components
X_reduced = X @ top_components  # Project to 2D
print(f"\nPCA: reduced from shape {X.shape} to {X_reduced.shape}")

# KEY TAKEAWAY:
# Dot product: measure similarity; matrix multiplication: apply transformations.
# Transpose: flip dimensions; inverse: solve linear systems.
# Eigenvalues/eigenvectors: reveal matrix structure (used in PCA, deep learning).
# SVD: universal decomposition method for compression and analysis.
# Matrix operations are foundations for all ML algorithms.
