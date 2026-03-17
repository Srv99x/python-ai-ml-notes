# Revision Notes:
# Topic: Pandas Handling Missing Values
# Why it matters for AI/ML: Real datasets almost always contain missing values.
# Proper handling is crucial for data quality and model performance.
# Different strategies (drop, fill, impute) suit different scenarios.

import pandas as pd
import numpy as np

# Create dataset with missing values
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Age': [25, np.nan, 28, np.nan, 30, 26],
    'Salary': [50000, 60000, np.nan, 55000, 70000, np.nan],
    'Department': ['Sales', 'IT', 'HR', np.nan, 'Sales', 'IT'],
    'YearsExp': [2, 5, np.nan, 3, 6, 4]
})

print("Dataset with missing values:")
print(df)
print("\n" + "="*60)

# ---
# DETECTING MISSING VALUES
# ---

# WHY: Always inspect data before deciding on handling strategy.

# isnull() / isna() - Return boolean mask
# WHY: Identify exactly which cells are missing.
print("\nMissing values (True = missing):")
print(df.isnull())

# Count missing values per column
# WHY: Understand impact and decide strategy (drop vs fill).
missing_counts = df.isnull().sum()
print("\nMissing value counts per column:")
print(missing_counts)

# Percentage of missing values
# WHY: Columns with >50% missing may need dropping.
missing_percent = (df.isnull().sum() / len(df)) * 100
print("\nPercentage missing:")
print(missing_percent)

# Visualize missing data
# WHY: Quickly see patterns (e.g., clustered missingness).
print("\nWhich rows have any missing values:")
print(df[df.isnull().any(axis=1)])

# ---
# DROP MISSING VALUES
# ---

# WHY: Use when missingness is minimal or data is random missing.

# Drop rows with any missing value
# WHY: Simplest approach but loses data; use when missing is < 5%.
df_dropped_any = df.dropna()
print("\n" + "="*60)
print("\nAfter dropna() (remove rows with ANY missing):")
print(df_dropped_any)
print(f"Shape: {df_dropped_any.shape} (lost {len(df) - len(df_dropped_any)} rows)")

# Drop rows with all missing values
# WHY: Remove completely empty rows (how='all').
df_dropped_all = df.dropna(how='all')
print("\nAfter dropna(how='all') (remove rows with ALL missing):")
print(df_dropped_all)

# Drop specific columns with too many missing values
# WHY: If column > 50% missing, often better to drop than impute.
df_dropped_col = df.dropna(axis=1, thresh=4)  # Keep columns with >= 4 non-null values
print("\nAfter dropping columns with < 4 non-null values:")
print(df_dropped_col)

# Drop rows where specific column is missing
# WHY: Focus on critical features (e.g., target variable).
df_dropped_specific = df.dropna(subset=['Age'])
print("\nAfter dropping rows where Age is missing:")
print(df_dropped_specific)

# ---
# FILL MISSING VALUES
# ---

# WHY: Retain more data by imputing instead of dropping.

# Forward fill: propagate last known value forward
# WHY: Use for time series where last value is reasonable estimate.
df_ffill = df.fillna(method='ffill')  # fillna works on copy
print("\n" + "="*60)
print("\nForward fill (ffill):")
print(df_ffill)

# Backward fill: propagate next known value backward
# WHY: Use when future values are more representative.
df_bfill = df.fillna(method='bfill')
print("\nBackward fill (bfill):")
print(df_bfill)

# Fill with constant value
# WHY: Use domain knowledge (e.g., 0 for no experience, 'Unknown' for category).
df_constant = df.fillna({'Age': df['Age'].mean(), 'Department': 'Unknown', 'Salary': 0})
print("\nFill with constant values:")
print(df_constant)

# Fill with mean (numeric columns)
# WHY: Preserve mean distribution for numeric features.
df_mean = df.copy()
df_mean['Age'] = df_mean['Age'].fillna(df_mean['Age'].mean())
df_mean['Salary'] = df_mean['Salary'].fillna(df_mean['Salary'].mean())
print("\nFill numeric columns with mean:")
print(df_mean)

# Fill with median
# WHY: Median is more robust to outliers than mean (preferred for skewed data).
df_median = df.copy()
df_median['Age'] = df_median['Age'].fillna(df_median['Age'].median())
print("\nFill with median:")
print(df_median)

# Fill with mode (most common value)
# WHY: Use for categorical columns.
df_mode = df.copy()
df_mode['Department'] = df_mode['Department'].fillna(df_mode['Department'].mode()[0])
print("\nFill categorical with mode:")
print(df_mode)

# Fill by group
# WHY: More sophisticated; use group-specific statistics.
df_group_fill = df.copy()
df_group_fill['Salary'] = df_group_fill.groupby('Department')['Salary'].transform(
    lambda x: x.fillna(x.mean())
)
print("\nFill salary with department-wise mean:")
print(df_group_fill)

# ---
# INTERPOLATION
# ---

# WHY: For time series or ordered data, interpolate between known values.

# Linear interpolation
# WHY: Smooth estimates between known points.
df_interp = df.copy()
df_interp['Age'] = df_interp['Age'].interpolate(method='linear')
print("\n" + "="*60)
print("\nLinear interpolation on Age:")
print(df_interp)

# Polynomial interpolation
# WHY: Fit polynomial curve through known values.
df_poly = df.copy()
df_poly['Age'] = df_poly['Age'].interpolate(method='polynomial', order=2)
print("\nPolynomial interpolation (order=2):")
print(df_poly)

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Strategy depends on missingness type
# WHY: Different strategies for different patterns.

df_research = df.copy()
for col in df_research.columns:
    missing_pct = (df_research[col].isnull().sum() / len(df_research)) * 100
    print(f"{col}: {missing_pct:.1f}% missing")
    
    if missing_pct > 30:
        print(f"  → Consider dropping {col}")
    elif df_research[col].dtype in ['float64', 'int64']:
        print(f"  → Fill with median")
    else:
        print(f"  → Fill with mode or custom value")

# Scenario 2: Prepare data for model training
# WHY: Models can't handle NaN; clean before modeling.

df_clean = df.copy()
# Strategy: Drop columns with >30% missing, fill others with median/mode
df_clean = df_clean.drop(columns=['YearsExp'])  # 33% missing
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
df_clean['Salary'].fillna(df_clean['Salary'].median(), inplace=True)
df_clean['Department'].fillna('Unknown', inplace=True)

print("\nCleaned dataset ready for modeling:")
print(df_clean)
print("\nMissing values remain:", df_clean.isnull().sum().sum())

# Scenario 3: Add indicator for was-missing (informative feature)
# WHY: Sometimes whether data was missing is predictive.
df_with_indicator = df.copy()
df_with_indicator['Age_WasMissing'] = df_with_indicator['Age'].isnull().astype(int)
df_with_indicator['Age'].fillna(df_with_indicator['Age'].mean(), inplace=True)
print("\nWith 'was missing' indicator feature:")
print(df_with_indicator[['Age', 'Age_WasMissing']])

# KEY TAKEAWAY:
# Always inspect: isnull().sum() and isnull().sum()/len(df) for %
# Drop if < 5% missing or column > 50% missing.
# Fill numeric: with mean/median (median more robust).
# Fill categorical: with mode or 'Unknown'.
# Interpolate: for time series with ordered missing patterns.
# Group-aware filling: use group statistics when appropriate.
# Consider adding 'was_missing' indicators as features for models.
