# Revision Notes:
# Topic: Pandas Filtering, GroupBy, and Aggregation
# Why it matters for AI/ML: Feature engineering and exploratory data analysis rely heavily
# on grouping data, computing statistics by group, and filtering subsets.
# These operations are foundational for understanding dataset patterns before modeling.

import pandas as pd
import numpy as np

# Create sample dataset
df = pd.DataFrame({
    'Department': ['Sales', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace'],
    'Salary': [50000, 70000, 55000, 48000, 75000, 52000, 49000],
    'Performance': [8.5, 9.0, 7.5, 8.0, 9.2, 8.3, 7.8],
    'YearsAtCompany': [2, 5, 3, 1, 6, 2, 1]
})

print("Sample Employee DataFrame:")
print(df)
print("\n" + "="*60)

# ---
# FILTERING (BOOLEAN INDEXING)
# ---

# WHY: Create subsets based on conditions. Essential for data exploration and train/test splits.

# Single condition
# WHY: Extract samples meeting one criterion.
salespeople = df[df['Department'] == 'Sales']
print("\nAll Salespeople:")
print(salespeople)

# Greater than / Less than
# WHY: Select high or low performers for analysis.
high_performers = df[df['Performance'] > 8.5]
print("\nHigh performers (Performance > 8.5):")
print(high_performers)

# Multiple conditions (AND)
# WHY: Combine conditions for precise filtering.
senior_sales = df[(df['Department'] == 'Sales') & (df['YearsAtCompany'] >= 2)]
print("\nSenior sales employees (2+ years):")
print(senior_sales)

# Multiple conditions (OR)
# WHY: Select if any condition matches.
high_earners = df[(df['Salary'] > 70000) | (df['Performance'] > 9.0)]
print("\nHigh earners OR top performers:")
print(high_earners)

# Negation (NOT)
# WHY: Exclude samples matching condition.
non_hr = df[df['Department'] != 'HR']
low_performance = df[~(df['Performance'] > 8.0)]  # NOT Performance > 8.0
print("\nNon-HR employees:")
print(non_hr)

# Using isin for multiple values
# WHY: Select from a list of allowed values (e.g., specific departments).
tech_staff = df[df['Department'].isin(['IT', 'HR'])]
print("\nIT and HR employees:")
print(tech_staff)

# Using str methods for string filtering
# WHY: Filter by text patterns (e.g., names starting with 'A').
names_with_a = df[df['Employee'].str.startswith('A')]
print("\nEmployees with names starting with 'A':")
print(names_with_a)

# ---
# GROUPBY: SPLIT-APPLY-COMBINE
# ---

# WHY: Group rows by category and compute statistics per group.
# Common in feature engineering and exploratory analysis.

# Group by single column and compute aggregate
# WHY: Summarize data per group (e.g., average salary per department).
groupby_dept = df.groupby('Department')
print("\n" + "="*60)
print("\nGroupBy Department - Average Salary:")
avg_salary = df.groupby('Department')['Salary'].mean()
print(avg_salary)

# Multiple aggregations
# WHY: Compute multiple statistics in one groupby operation.
dept_stats = df.groupby('Department')[['Salary', 'Performance']].agg(['mean', 'std', 'min', 'max'])
print("\nDepartment Statistics (Mean, Std, Min, Max):")
print(dept_stats)

# Custom aggregation names
# WHY: Rename aggregated columns for clarity.
dept_summary = df.groupby('Department')['Salary'].agg(['mean', 'count']).rename(
    columns={'mean': 'AvgSalary', 'count': 'Count'}
)
print("\nDepartment Summary (custom names):")
print(dept_summary)

# Group by multiple columns
# WHY: Create hierarchical groups (e.g., by department AND performance level).
df['PerfLevel'] = pd.cut(df['Performance'], bins=[0, 8.0, 10.0], labels=['Standard', 'Excellent'])
multi_group = df.groupby(['Department', 'PerfLevel'])[['Salary']].mean()
print("\nSalary by Department and Performance Level:")
print(multi_group)

# Count occurrences per group
# WHY: Understand distribution or class imbalance.
dept_counts = df['Department'].value_counts()
print("\nEmployee counts by department:")
print(dept_counts)

# ---
# AGG: APPLY MULTIPLE FUNCTIONS
# ---

# WHY: Compute multiple aggregations efficiently in a single call.

# Different functions for different columns
# WHY: Not all aggregations apply equally; salary needs mean, performance needs std.
agg_dict = {
    'Salary': ['mean', 'std'],
    'Performance': ['mean', 'max'],
    'YearsAtCompany': 'median'
}
result = df.groupby('Department').agg(agg_dict)
print("\n" + "="*60)
print("\nCustom aggregation by department:")
print(result)

# Using lambda for custom aggregations
# WHY: Apply non-standard operations (e.g., salary range).
custom_agg = df.groupby('Department').agg({
    'Salary': ['min', 'max', lambda x: x.max() - x.min()],
    'Performance': lambda x: x.sum() / len(x)
})
print("\nCustom aggregation (with lambda):")
print(custom_agg)

# ---
# TRANSFORM: GROUP-WISE TRANSFORMATIONS
# ---

# WHY: Apply function per group while preserving original shape (broadcast back).

# Standardize salary within each department
# WHY: Remove department-level bias before modeling.
df['Salary_Standardized'] = df.groupby('Department')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print("\n" + "="*60)
print("\nSalary standardized within departments:")
print(df[['Department', 'Salary', 'Salary_Standardized']])

# Rank within groups
# WHY: Get relative performance within each department.
df['Dept_SalaryRank'] = df.groupby('Department')['Salary'].transform('rank')
print("\nSalary rank within departments:")
print(df[['Department', 'Salary', 'Dept_SalaryRank']])

# ---
# FILTERING GROUPS
# ---

# WHY: Select entire groups based on aggregate criterion (e.g., departments with high average salary).

# Filter groups by aggregate condition
# WHY: Keep only departments with average salary > 55000.
high_avg_depts = df.groupby('Department').filter(lambda x: x['Salary'].mean() > 55000)
print("\n" + "="*60)
print("\nEmployees in high-paying departments (avg > 55000):")
print(high_avg_depts)

# Keep groups with sufficient size
# WHY: Exclude small groups (e.g., departments with < 2 employees).
large_depts = df.groupby('Department').filter(lambda x: len(x) >= 2)
print("\nDepartments with 2+ employees:")
print(large_depts)

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Feature engineering - department averages as features
# WHY: Add group statistics as features (domain knowledge injection).
dept_avg = df.groupby('Department')['Salary'].transform('mean').rename('Dept_AvgSalary')
X_with_feature = pd.concat([df[['Salary', 'Performance']], dept_avg], axis=1)
print("With department average salary feature:")
print(X_with_feature.head())

# Scenario 2: Identify outliers within groups
# WHY: Outliers may differ by group; detect them within-group.
def identify_outliers(group):
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    return (group < Q1 - 1.5*IQR) | (group > Q3 + 1.5*IQR)

df['SalaryOutlier'] = df.groupby('Department')['Salary'].transform(identify_outliers)
print("\nOutliers by department (using IQR):")
print(df[['Department', 'Salary', 'SalaryOutlier']])

# Scenario 3: Stratified sampling for train/test split
# WHY: Ensure each group is represented in train and test sets.
train_mask = df.groupby('Department').apply(lambda x: np.random.choice([True, False], size=len(x), p=[0.7, 0.3])).reset_index(level=0, drop=True)
train_set = df[train_mask]
test_set = df[~train_mask]
print("\nStratified train/test split:")
print(f"Train: {len(train_set)}, Test: {len(test_set)}")
print(f"Dept distribution in train: {train_set['Department'].value_counts().to_dict()}")

# KEY TAKEAWAY:
# Filter using boolean masks; combine conditions with & (AND) and | (OR).
# groupby() splits data and applies functions per group.
# agg() computes multiple statistics; transform() preserves original shape.
# filter() selects entire groups based on aggregate criteria.
# These operations are essential for exploratory analysis and feature engineering in ML.
