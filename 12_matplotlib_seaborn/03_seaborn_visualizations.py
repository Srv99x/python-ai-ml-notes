# Revision Notes:
# Topic: Seaborn Visualizations
# Why it matters for AI/ML: Seaborn builds on matplotlib with statistical plotting.
# It simplifies creating complex visualizations and handles DataFrames natively.
# Essential for exploratory data analysis and relationship visualization.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
sns.set_theme(style="whitegrid")

# Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'Age': np.random.randint(20, 60, 200),
    'Salary': np.random.randint(30000, 120000, 200),
    'Department': np.random.choice(['Sales', 'IT', 'HR', 'Marketing'], 200),
    'YearsExp': np.random.randint(0, 30, 200),
    'Performance': np.random.uniform(5, 10, 200)
})

print("Sample dataset loaded")
print(f"Shape: {data.shape}")

# ---
# RELPLOT: RELATIONSHIP PLOTS
# ---

# WHY: Visualize relationships between variables; can show subsets by category.

# Scatter plot with hue
# WHY: Color by category to see if relationship differs by group.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='Salary', hue='Department', size='Performance', alpha=0.6)
plt.title('Salary vs Age by Department')
plt.savefig('seaborn_scatter.png')
plt.close()
print("Scatter plot saved")

# Regression plot
# WHY: Show linear relationship with confidence interval.
plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='Age', y='Salary', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title('Salary vs Age (with linear fit)')
plt.savefig('seaborn_regplot.png')
plt.close()
print("Regression plot saved")

# ---
# HEATMAP: CORRELATION MATRIX
# ---

# WHY: Visualize correlations between all numeric variables at once.

plt.figure(figsize=(8, 6))
numeric_data = data[['Age', 'Salary', 'YearsExp', 'Performance']].corr()
sns.heatmap(numeric_data, annot=True, cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Correlation'}, square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('seaborn_heatmap.png')
plt.close()
print("Heatmap saved")

# ---
# BOXPLOT: COMPARE DISTRIBUTIONS BY GROUP
# ---

# WHY: Show distribution across groups; identify outliers and quartiles.

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Department', y='Salary', hue='Department', palette='Set2')
plt.title('Salary Distribution by Department')
plt.savefig('seaborn_boxplot.png')
plt.close()
print("Boxplot saved")

# Violin plot (smoother representation of distribution)
# WHY: Shows full distribution shape; better than boxplot for complex distributions.
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='Department', y='Salary', palette='muted')
plt.title('Salary Distribution (Violin Plot)')
plt.savefig('seaborn_violin.png')
plt.close()
print("Violin plot saved")

# ---
# HISTPLOT: DISTRIBUTION WITH BY-GROUP COMPARISON
# ---

# WHY: Compare distributions with precise bin control and categorical subsets.

plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='Salary', hue='Department', kde=True, stat='density', bins=30)
plt.title('Salary Distribution by Department')
plt.savefig('seaborn_histplot.png')
plt.close()
print("Histplot saved")

# ---
# BARPLOT: AGGREGATE VALUES BY GROUP
# ---

# WHY: Show mean/aggregate with confidence intervals across categories.

plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='Department', y='Salary', estimator=np.mean, errorbar='sd')
plt.title('Mean Salary by Department (with error bars)')
plt.savefig('seaborn_barplot.png')
plt.close()
print("Barplot saved")

# ---
# COUNTPLOT: COUNT OBSERVATIONS BY CATEGORY
# ---

# WHY: Show frequency distribution across categorical variable.

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Department', palette='husl')
plt.title('Employee Count by Department')
plt.savefig('seaborn_countplot.png')
plt.close()
print("Countplot saved")

# ---
# PAIRPLOT: MATRIX OF SCATTER PLOTS
# ---

# WHY: Quickly explore all pairwise relationships in dataset.

# Create smaller dataset for faster processing
data_small = data.sample(50)
sns.pairplot(data_small[['Age', 'Salary', 'YearsExp', 'Performance']], 
             diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot: All Relationships', y=1.00)
plt.savefig('seaborn_pairplot.png')
plt.close()
print("Pairplot saved")

# Pairplot with hue
# WHY: Color points by category to see if patterns differ by group.
sns.pairplot(data_small[['Age', 'Salary', 'Department', 'Performance']], 
             hue='Department', diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot by Department', y=0.995)
plt.savefig('seaborn_pairplot_hue.png')
plt.close()
print("Pairplot with hue saved")

# ---
# STRIPPLOT & SWARMPLOT: CATEGORICAL SCATTER
# ---

# WHY: Show individual points for categorical x-axis (avoids overlapping).

plt.figure(figsize=(10, 6))
sns.stripplot(data=data, x='Department', y='Salary', hue='Department', jitter=True, size=8, alpha=0.6)
plt.title('Salary by Department (Strip Plot)')
plt.savefig('seaborn_stripplot.png')
plt.close()
print("Stripplot saved")

# Swarmplot (prevents overlap via positioning)
# WHY: Shows all points without overplotting when data is not too large.
plt.figure(figsize=(10, 6))
sns.swarmplot(data=data_small, x='Department', y='Salary', hue='Department', size=8)
plt.title('Salary by Department (Swarm Plot)')
plt.savefig('seaborn_swarmplot.png')
plt.close()
print("Swarmplot saved")

# ---
# LINEPLOT: TIME SERIES OR SEQUENCE
# ---

# WHY: Show trends with confidence intervals across groups.

# Create time series data
ts_data = pd.DataFrame({
    'Month': np.tile(range(12), 3),
    'Sales': np.concatenate([
        100 + np.cumsum(np.random.randn(12) * 5),
        110 + np.cumsum(np.random.randn(12) * 5),
        105 + np.cumsum(np.random.randn(12) * 5)
    ]),
    'Region': np.repeat(['North', 'South', 'East'], 12)
})

plt.figure(figsize=(12, 6))
sns.lineplot(data=ts_data, x='Month', y='Sales', hue='Region', marker='o', linewidth=2)
plt.title('Sales Trend by Region')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.savefig('seaborn_lineplot.png')
plt.close()
print("Lineplot saved")

# ---
# FACETGRID: MULTIPLE SUBPLOTS BY CATEGORY
# ---

# WHY: Create grid of plots, one per category, for side-by-side comparison.

g = sns.FacetGrid(data, col='Department', col_wrap=2, height=4)
g.map(sns.scatterplot, 'Age', 'Salary', alpha=0.6)
g.set_axis_labels('Age', 'Salary')
plt.savefig('seaborn_facetgrid.png')
plt.close()
print("FacetGrid saved")

# ---
# CLUSTERMAP: HIERARCHICAL CLUSTERING HEATMAP
# ---

# WHY: Reveal patterns via hierarchical clustering on heatmap.

# Prepare data for clustering
cluster_data = data[['Age', 'Salary', 'YearsExp', 'Performance']].sample(30).reset_index(drop=True)
cluster_data_normalized = (cluster_data - cluster_data.mean()) / cluster_data.std()

g = sns.clustermap(cluster_data_normalized, cmap='vlag', center=0, method='average')
g.ax_heatmap.set_title('Hierarchical Clustering Heatmap')
plt.savefig('seaborn_clustermap.png')
plt.close()
print("Clustermap saved")

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Feature correlation analysis before modeling
# WHY: Identify highly correlated features to remove (multicollinearity).
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = data[['Age', 'Salary', 'YearsExp', 'Performance']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax)
plt.title('Feature Correlations (check for multicollinearity)')
plt.savefig('scenario_correlation.png')
plt.close()

# Scenario 2: Distribution before and after preprocessing
# WHY: Validate transformations visually.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(data=data, x='Salary', kde=True, ax=axes[0])
axes[0].set_title('Original Salary Distribution')

# Log transform
data['SalaryLog'] = np.log(data['Salary'])
sns.histplot(data=data, x='SalaryLog', kde=True, ax=axes[1])
axes[1].set_title('Log-Transformed Salary')
plt.tight_layout()
plt.savefig('scenario_preprocessing.png')
plt.close()

print("ML scenario plots saved")

# Clean up
import os
for fname in ['seaborn_scatter.png', 'seaborn_regplot.png', 'seaborn_heatmap.png',
              'seaborn_boxplot.png', 'seaborn_violin.png', 'seaborn_histplot.png',
              'seaborn_barplot.png', 'seaborn_countplot.png', 'seaborn_pairplot.png',
              'seaborn_pairplot_hue.png', 'seaborn_stripplot.png', 'seaborn_swarmplot.png',
              'seaborn_lineplot.png', 'seaborn_facetgrid.png', 'seaborn_clustermap.png',
              'scenario_correlation.png', 'scenario_preprocessing.png']:
    try:
        os.remove(fname)
    except:
        pass

# KEY TAKEAWAY:
# Seaborn integrates with pandas DataFrames; use 'data', 'x', 'y', 'hue' parameters.
# Use hue to add categorical dimension; use col/row to create subplots.
# Heatmap: correlation analysis; pairplot: explore all relationships.
# Boxplot/violin: compare distributions across groups; stripplot/swarmplot: show individual points.
# FacetGrid: create multiple plots by category for systematic comparison.
# Always explore data visually before modeling to catch patterns and anomalies.
