# Revision Notes:
# Topic: Pandas Series and DataFrame Creation
# Why it matters for AI/ML: DataFrames are the standard for loading and manipulating data in Python.
# Unlike NumPy arrays, DataFrames have labels (column names) and handle mixed data types naturally.
# This is essential for exploratory data analysis and data preprocessing in ML workflows.

import pandas as pd
import numpy as np

# ---
# SERIES: 1D LABELED DATA
# ---

# WHY: Series are like labeled lists, mapping indices to values.
# Used for individual features, time series, or target variables.

# Create Series from list
# WHY: Convert simple data into labeled structure for pandas operations.
s1 = pd.Series([10, 20, 30, 40])
print("Series from list:")
print(s1)
print("\nDefault index:", s1.index.tolist())

# Create Series with custom index
# WHY: Meaningful labels (e.g., dates, names) make data more interpretable.
s2 = pd.Series([85, 92, 88, 95], index=['Alice', 'Bob', 'Charlie', 'Diana'])
print("\nSeries with custom index:")
print(s2)

# Access element by label
# WHY: Use labels instead of positions for more readable, self-documenting code.
print("\nAlice's score:", s2['Alice'])
print("Bob's score:", s2.loc['Bob'])

# Create Series from dictionary
# WHY: Dictionary keys become index labels automatically.
s3 = pd.Series({'apple': 5, 'banana': 7, 'orange': 3})
print("\nSeries from dict:")
print(s3)

# Series properties
# WHY: Understand data structure and metadata.
print("\nSeries dtype:", s2.dtype)
print("Series length:", len(s2))
print("Series values:", s2.values)  # NumPy array underlying the Series

# ---
# DATAFRAME: 2D LABELED DATA
# ---

# WHY: The most common data structure in ML. Each column is a feature,
# each row is a sample. Labels make operations intuitive.

# Create DataFrame from dictionary of lists
# WHY: Standard format for loading datasets.
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 28, 22],
    'Salary': [50000, 60000, 55000, 48000],
    'Department': ['Sales', 'IT', 'HR', 'Sales']
}
df = pd.DataFrame(data_dict)
print("\nDataFrame from dict:")
print(df)

# DataFrame properties
# WHY: Essential for understanding dataset structure before analysis.
print("\nDataFrame shape:", df.shape)  # (4, 4) → 4 rows, 4 columns
print("DataFreame columns:", df.columns.tolist())
print("DataFrame dtypes:\n", df.dtypes)
print("\nDataFrame info:")
df.info()  # Concise summary of data types and missing values

# Access column as Series
# WHY: Extract a single feature for analysis or model input.
names = df['Name']
ages = df['Age']
print("\nAccess 'Age' column:", ages.values)

# Multiple columns
# WHY: Select feature subset for modeling.
subset = df[['Name', 'Salary']]
print("\nSubset [Name, Salary]:")
print(subset)

# ---
# CREATE DATAFRAME FROM LISTS OF DICTS
# ---

# WHY: Alternative format that's sometimes more intuitive for row-based data.

records = [
    {'Name': 'Alice', 'Age': 25, 'Salary': 50000},
    {'Name': 'Bob', 'Age': 30, 'Salary': 60000},
    {'Name': 'Charlie', 'Age': 28, 'Salary': 55000}
]
df2 = pd.DataFrame(records)
print("\nDataFrame from list of dicts:")
print(df2)

# ---
# CREATE DATAFRAME FROM NUMPY ARRAYS
# ---

# WHY: Convert NumPy arrays (from ML models or calculations) to DataFrames.

arr = np.random.randn(3, 3)
df3 = pd.DataFrame(arr, columns=['Feature1', 'Feature2', 'Feature3'])
print("\nDataFrame from NumPy array:")
print(df3)

# ---
# DATAFRAME INDEXING AND LABELS
# ---

# WHY: Use meaningful indices for easier data interpretation and joining.

df_indexed = pd.DataFrame(
    {'Value1': [10, 20, 30], 'Value2': [100, 200, 300]},
    index=['ID1', 'ID2', 'ID3']
)
print("\nDataFrame with custom index:")
print(df_indexed)

# Set column as index
# WHY: Move a column to the index for hierarchical or multi-level operations.
df_with_index = df.set_index('Name')
print("\nDataFrame with 'Name' as index:")
print(df_with_index)

# Reset index
# WHY: Convert index back to a column.
df_reset = df_with_index.reset_index()
print("\nAfter reset_index():")
print(df_reset)

# ---
# BASIC STATISTICS
# ---

# WHY: Understand data distribution before modeling.

print("\nDescriptive statistics:")
print(df.describe())  # Auto-calculates mean, std, min, max, quartiles for numeric columns

# Statistics for specific columns
# WHY: Quick numerical summaries.
print("\nMean age:", df['Age'].mean())
print("Max salary:", df['Salary'].max())
print("Standard deviation of age:", df['Age'].std())

# ---
# DATAFRAME OPERATIONS
# ---

# WHY: Transform and manipulate data in vectorized fashion.

# Add new column
# WHY: Create derived features or target transformations.
df['Bonus'] = df['Salary'] * 0.1
print("\nAfter adding Bonus column:")
print(df)

# Apply function to column
# WHY: Custom transformation on each element without explicit loops.
df['AgeGroup'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Senior')
print("\nAfter applying AgeGroup:")
print(df)

# Drop column
# WHY: Remove unnecessary or redundant features.
df_dropped = df.drop('AgeGroup', axis=1)
print("\nAfter dropping AgeGroup:")
print(df_dropped.head())

# KEY TAKEAWAY:
# Series are 1D labeled arrays; DataFrames are 2D labeled tables.
# Use meaningful labels for indexing and column names.
# describe() provides quick statistical summaries.
# apply() and vectorized operations transform data without loops.
# DataFrames integrate seamlessly with ML pipelines via column/index manipulation.
