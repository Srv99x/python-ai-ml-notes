# Revision Notes:
# Topic: Pandas Indexing - loc vs iloc
# Why it matters for AI/ML: Efficient data selection is critical for feature engineering,
# creating subsets for train/test splits, and sample inspection during model debugging.
# loc and iloc are the two primary indexing methods; understanding both is essential.

import pandas as pd
import numpy as np

# Create sample DataFrame
df = pd.DataFrame({
    'StudentID': ['S001', 'S002', 'S003', 'S004', 'S005'],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Math': [85, 92, 88, 95, 78],
    'English': [90, 88, 85, 92, 95],
    'Science': [88, 85, 90, 88, 85]
})

print("Sample DataFrame:")
print(df)
print("\n" + "="*60)

# ---
# ILOC: INTEGER-LOCATION BASED INDEXING
# ---

# WHY: Use when you need to select by position (0, 1, 2, ...).
# Works like NumPy array indexing.

# Select single row by position
# WHY: Access nth row regardless of its index label.
row_2 = df.iloc[1]  # Get second row (index 1)
print("\nSingle row (iloc[1]):")
print(row_2)

# Select multiple rows by position
# WHY: Get contiguous or specific rows using slicing or lists.
rows_0_2 = df.iloc[0:3]  # Rows at positions 0, 1, 2 (stop is exclusive)
print("\nRows (iloc[0:3]):")
print(rows_0_2)

# Select specific rows and columns by position
# WHY: Extract rectangular subset (e.g., features for specific samples).
subset = df.iloc[1:4, 1:3]  # Rows 1-3, columns 1-2 (Name, Math)
print("\nSubset (iloc[1:4, 1:3]):")
print(subset)

# Select single element by position
# WHY: Get specific cell value.
value = df.iloc[2, 3]  # Row 2, Column 3 (Charlie's English score)
print("\nSingle element (iloc[2, 3]):", value)  # 85

# Select last rows/columns
# WHY: Access most recent data or rightmost features using negative indices.
last_row = df.iloc[-1]  # Last row
last_two_rows = df.iloc[-2:]  # Last 2 rows
print("\nLast row (iloc[-1]):")
print(last_row)

# Fancy indexing with lists
# WHY: Select non-contiguous rows or columns.
fancy = df.iloc[[0, 2, 4], [1, 3]]  # Rows 0, 2, 4 and columns 1, 3
print("\nFancy indexing (iloc[[0,2,4], [1,3]]):")
print(fancy)

# ---
# LOC: LABEL-BASED INDEXING
# ---

# WHY: Use when you know the label/name (column name, index label).
# More intuitive and self-documenting than positional indexing.

# Select single column by name
# WHY: Access feature by meaningful name instead of position.
names = df.loc[:, 'Name']  # All rows, column 'Name'
print("\n'Name' column (loc[:, 'Name']):")
print(names)

# Select specific row by label
# WHY: If index has meaningful labels (IDs, dates), use them directly.
df_labeled = df.set_index('StudentID')
student_s003 = df_labeled.loc['S003']  # Row with index 'S003'
print("\nStudent S003 data (loc['S003']):")
print(student_s003)

# Select multiple columns by name
# WHY: Create feature subset for modeling.
scores = df.loc[:, ['Math', 'English', 'Science']]
print("\nScores columns:")
print(scores)

# Select rows and columns by labels
# WHY: Extract specific cells or subcells using meaningful names.
subset_loc = df_labeled.loc[['S002', 'S004'], ['Math', 'Science']]
print("\nSubset (loc[['S002','S004'], ['Math','Science']]):")
print(subset_loc)

# Slicing with labels (inclusive on both ends)
# WHY: Unlike iloc, loc slicing includes the stop label.
df_indexed = df.set_index('Name')
slice_loc = df_indexed.loc['Alice':'Charlie']  # Includes 'Charlie'
print("\nSlice 'Alice' to 'Charlie' (inclusive):")
print(slice_loc)

# ---
# BOOLEAN INDEXING WITH LOC
# ---

# WHY: Filter rows based on conditions (essential for data cleaning and feature selection).

# Single condition
# WHY: Extract samples meeting a criterion (e.g., high performers).
high_math = df.loc[df['Math'] > 85]  # Rows where Math score > 85
print("\nRows where Math > 85:")
print(high_math)

# Multiple conditions
# WHY: Combine conditions for complex filtering.
good_students = df.loc[(df['Math'] > 85) & (df['English'] > 85)]  # AND
print("\nRows where Math > 85 AND English > 85:")
print(good_students)

# OR condition
# WHY: Select if any condition is true.
top_performers = df.loc[(df['Math'] > 90) | (df['English'] > 90)]
print("\nRows where Math > 90 OR English > 90:")
print(top_performers)

# Invert condition with NOT
# WHY: Exclude rows meeting criteria.
below_90_math = df.loc[~(df['Math'] > 90)]  # NOT Math > 90
print("\nRows where NOT (Math > 90):")
print(below_90_math)

# Using isin for membership
# WHY: Filter by list of allowed values.
names_to_select = ['Alice', 'Bob', 'Eve']
selected = df.loc[df['Name'].isin(names_to_select)]
print("\nRows where Name in ['Alice', 'Bob', 'Eve']:")
print(selected)

# ---
# LOC WITH AT AND IAT (SINGLE ELEMENT)
# ---

# WHY: More efficient for single-element access than iloc/loc with indexing.

# at: single element by label
# WHY: Faster than loc for accessing single cell in large DataFrames.
charlie_math = df_labeled.at['S003', 'Math']
print(f"\nCharlie's Math score (at): {charlie_math}")

# iat: single element by position
# WHY: Faster than iloc for single-element access.
elem = df.iat[2, 3]  # Row 2, Column 3
print(f"Element at position [2, 3] (iat): {elem}")

# ---
# PRACTICAL MACHINE LEARNING SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Create train-test split
# WHY: Standard ML practice requires splitting data.
train_idx = [0, 1, 2]
test_idx = [3, 4]
X_train = df.iloc[train_idx, 1:5]  # Columns 1-4 (features)
X_test = df.iloc[test_idx, 1:5]
print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Scenario 2: Feature selection based on condition
# WHY: Remove low-variance features or select by domain knowledge.
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\nNumeric columns:", numeric_cols.tolist())

# Scenario 3: Create target variable
# WHY: Often target is derived from or separated from features.
y = (df['Math'] > 85).astype(int)  # Binary: 1 if Math > 85, else 0
print("\nTarget variable (Math > 85):")
print(y.to_dict())

# KEY TAKEAWAY:
# iloc: Use for position-based indexing (like NumPy).
# loc: Use for label-based indexing (more intuitive and readable).
# loc supports boolean indexing for filtering; iloc does not.
# Use at/iat for single-element access for efficiency.
# Mastering both enables effective data manipulation for ML workflows.
