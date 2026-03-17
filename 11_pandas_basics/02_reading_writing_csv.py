# Revision Notes:
# Topic: Pandas Reading and Writing CSV Files
# Why it matters for AI/ML: CSV is the standard format for datasets in data science.
# Loading and saving data efficiently is the first step of any ML pipeline.
# Understanding read_csv parameters and write modes is essential for data workflows.

import pandas as pd
import os

# ---
# READING CSV FILES
# ---

# WHY: CSV (Comma-Separated Values) is the most common data exchange format.

# Basic read_csv
# WHY: Load tabular data into a DataFrame for analysis and modeling.
# Example: df = pd.read_csv('data.csv')
# This would load a CSV file with default settings.

# Important parameters for read_csv:
# WHY: Customize loading behavior for different file formats and requirements.

# 1. sep/delimiter: Change delimiter if not comma
# WHY: Some files use tabs, semicolons, or pipes as separators.
# df = pd.read_csv('data.tsv', sep='\t')  # Tab-separated values
# df = pd.read_csv('data.csv', sep=';')   # Semicolon-separated

# 2. header: Specify which row contains column names
# WHY: Skip metadata rows or handle files without headers.
# df = pd.read_csv('data.csv', header=0)  # Default: first row is header
# df = pd.read_csv('data.csv', header=None)  # No header; auto-generate indices

# 3. names: Rename columns while reading
# WHY: Assign meaningful names if file lacks headers or has poor naming.
# df = pd.read_csv('data.csv', names=['ID', 'Name', 'Age', 'Salary'])

# 4. index_col: Set a column as index
# WHY: Use meaningful identifiers instead of default numeric indices.
# df = pd.read_csv('data.csv', index_col='ID')

# 5. usecols: Load only specific columns
# WHY: Reduce memory usage and loading time for large files with many columns.
# df = pd.read_csv('data.csv', usecols=['Name', 'Age', 'Salary'])

# 6. dtype: Specify data types for columns
# WHY: Control parsing and save memory (e.g., use 'int8' instead of 'int64').
# df = pd.read_csv('data.csv', dtype={'Age': 'int32', 'Salary': 'float32'})

# 7. nrows: Load only first N rows
# WHY: Test code on sample before loading entire large dataset.
# df = pd.read_csv('data.csv', nrows=1000)

# 8. skiprows: Skip initial rows
# WHY: Skip metadata, headers, or comments before actual data.
# df = pd.read_csv('data.csv', skiprows=2)

# 9. na_values: Specify additional strings representing missing values
# WHY: Standardize missing value representation (e.g., 'NA', 'N/A', '-').
# df = pd.read_csv('data.csv', na_values=['NA', 'N/A', '-'])

# 10. parse_dates: Treat columns as dates
# WHY: Automatically convert string dates to datetime for time series analysis.
# df = pd.read_csv('data.csv', parse_dates=['DateColumn'])

# Demonstrate read_csv with practical example
# Create sample CSV in memory and read it
csv_data = """Name,Age,Salary,Department
Alice,25,50000,Sales
Bob,30,60000,IT
Charlie,28,55000,HR
Diana,22,48000,Sales"""

# Write sample to temp file and read it back
temp_file = 'sample_data.csv'
with open(temp_file, 'w') as f:
    f.write(csv_data)

# Basic read
df = pd.read_csv(temp_file)
print("Read CSV (basic):")
print(df)

# Read with specific dtypes
df_typed = pd.read_csv(temp_file, dtype={'Age': 'int32', 'Salary': 'float32'})
print("\nRead CSV (with dtypes):")
print(df_typed.dtypes)

# Read only some rows
df_sample = pd.read_csv(temp_file, nrows=2)
print("\nRead first 2 rows:")
print(df_sample)

# ---
# WRITING CSV FILES
# ---

# WHY: Save processed data for sharing, archiving, or downstream pipelines.

# Basic to_csv
# WHY: Export DataFrames for use in other tools or for backup.
df.to_csv('output.csv', index=False)  # index=False prevents row number column
print("\nDataFrame saved to 'output.csv'")

# Important parameters for to_csv:
# WHY: Control output format and content.

# 1. index: Include row indices in output
# WHY: Exclude if indices are auto-generated; include if meaningful (IDs, dates).
# df.to_csv('output.csv', index=True)  # Include index column
# df.to_csv('output.csv', index=False)  # Exclude index column

# 2. columns: Select which columns to save
# WHY: Export subset of features or exclude temporary columns.
# df.to_csv('output.csv', columns=['Name', 'Salary'])

# 3. sep: Use different delimiter
# WHY: Create tab-separated or semicolon-separated files for compatibility.
# df.to_csv('output.tsv', sep='\t')

# 4. na_rep: Represent missing values as specific string
# WHY: Handle missing values consistently across systems.
# df.to_csv('output.csv', na_rep='NA')

# 5. header: Include column names in output
# WHY: Exclude headers if appending to existing file or creating raw data.
# df.to_csv('output.csv', header=True)
# df.to_csv('output.csv', header=False)

# 6. mode: Write ('w') or append ('a') to file
# WHY: Append new data to existing file instead of overwriting.
# df.to_csv('output.csv', mode='w')  # Overwrite file
# df.to_csv('output.csv', mode='a', header=False)  # Append without header

# Practical example: Save with options
df.to_csv('output_no_index.csv', index=False, columns=['Name', 'Salary'])
print("Saved subset to 'output_no_index.csv'")

# Append mode example
df.head(2).to_csv('output_append.csv', index=False)  # Create file
df.tail(2).to_csv('output_append.csv', index=False, mode='a', header=False)  # Append
print("Appended data to 'output_append.csv'")

# ---
# HANDLING LARGE FILES
# ---

# WHY: Entire dataset may not fit in memory; process in chunks.

# Read in chunks
# WHY: Process large files without loading everything into RAM.
chunk_size = 2
for chunk in pd.read_csv(temp_file, chunksize=chunk_size):
    print("\nChunk:")
    print(chunk)
    # Process each chunk independently (train incrementally, etc.)

# ---
# VERIFY AND INSPECT LOADED DATA
# ---

# WHY: Immediately check for issues after loading.

df_loaded = pd.read_csv(temp_file)
print("\n--- Data Inspection ---")
print("Shape:", df_loaded.shape)
print("Columns:", df_loaded.columns.tolist())
print("Data types:", df_loaded.dtypes.to_dict())
print("Missing values:\n", df_loaded.isnull().sum())
print("\nFirst few rows:")
print(df_loaded.head())

# Clean up temp files
os.remove(temp_file)
if os.path.exists('output.csv'):
    os.remove('output.csv')
if os.path.exists('output_no_index.csv'):
    os.remove('output_no_index.csv')
if os.path.exists('output_append.csv'):
    os.remove('output_append.csv')

# KEY TAKEAWAY:
# read_csv with key params: sep, header, usecols, dtype, nrows, parse_dates.
# to_csv with key params: index, columns, mode (write vs append).
# Always inspect loaded data for shape, dtypes, and missing values.
# Use chunksize for memory-efficient processing of large files.
# These skills are fundamental for any data science workflow.
