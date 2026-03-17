# Revision Notes:
# Topic: Matplotlib Plot Types
# Why it matters for AI/ML: Exploratory Data Analysis (EDA) is essential before modeling.
# Visualizations reveal patterns, outliers, and relationships that guide feature engineering.
# Different plot types suit different data and questions.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2 * x + np.random.randn(50) * 5
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

# ---
# LINE PLOT
# ---

# WHY: Show trends over time or relationships between continuous variables.
plt.figure(figsize=(10, 5))
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot: Trend Over Time')
plt.grid(True, alpha=0.3)
plt.savefig('plot_01_line.png')
plt.close()
print("Line plot saved")

# Multiple lines
# WHY: Compare trends across multiple series.
y2 = 1.5 * x + np.random.randn(50) * 4
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Series 1', marker='o')
plt.plot(x, y2, label='Series 2', marker='s')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Multiple Line Plots')
plt.savefig('plot_02_multi_line.png')
plt.close()
print("Multiple line plot saved")

# ---
# SCATTER PLOT
# ---

# WHY: Visualize relationship between two variables; show clusters or correlation.
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.6, s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot: Relationship')
plt.grid(True, alpha=0.3)
plt.savefig('plot_03_scatter.png')
plt.close()
print("Scatter plot saved")

# Colored by category
# WHY: Show third dimension via color.
categories_bool = np.random.choice(['Class A', 'Class B'], size=len(x))
colors = ['red' if c == 'Class A' else 'blue' for c in categories_bool]
plt.figure(figsize=(10, 5))
plt.scatter(x, y, c=colors, alpha=0.6, s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot: Colored by Class')
plt.legend(['Class A', 'Class B'])
plt.savefig('plot_04_scatter_colored.png')
plt.close()
print("Colored scatter plot saved")

# ---
# BAR PLOT
# ---

# WHY: Compare categorical values; show magnitudes across groups.
plt.figure(figsize=(10, 5))
plt.bar(categories, values, color=['red', 'blue', 'green', 'orange'])
plt.ylabel('Values')
plt.title('Bar Plot: Compare Categories')
for i, v in enumerate(values):
    plt.text(i, v + 1, str(v), ha='center')  # Add value labels
plt.savefig('plot_05_bar.png')
plt.close()
print("Bar plot saved")

# Horizontal bar (better for long category names)
# WHY: Readability for many categories.
plt.figure(figsize=(10, 5))
plt.barh(categories, values, color=['red', 'blue', 'green', 'orange'])
plt.xlabel('Values')
plt.title('Horizontal Bar Plot')
plt.savefig('plot_06_bar_h.png')
plt.close()
print("Horizontal bar plot saved")

# ---
# HISTOGRAM
# ---

# WHY: Show distribution of single variable; reveal skewness, bimodality, outliers.
data = np.random.randn(1000) * 10 + 50
plt.figure(figsize=(10, 5))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram: Distribution of Data')
plt.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.1f}')
plt.axvline(np.median(data), color='green', linestyle='--', label=f'Median: {np.median(data):.1f}')
plt.legend()
plt.savefig('plot_07_histogram.png')
plt.close()
print("Histogram saved")

# Multiple histograms overlay
# WHY: Compare distributions of different groups.
data1 = np.random.randn(500) * 5 + 50
data2 = np.random.randn(500) * 8 + 55
plt.figure(figsize=(10, 5))
plt.hist(data1, bins=25, alpha=0.5, label='Group 1')
plt.hist(data2, bins=25, alpha=0.5, label='Group 2')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Overlapping Histograms')
plt.legend()
plt.savefig('plot_08_hist_multi.png')
plt.close()
print("Multi-histogram saved")

# ---
# BOX PLOT
# ---

# WHY: Show distribution, quartiles, and outliers; compare across groups.
data_groups = [np.random.randn(100) + 50, np.random.randn(100) + 55, np.random.randn(100) + 48]
plt.figure(figsize=(10, 5))
plt.boxplot(data_groups, labels=['Group A', 'Group B', 'Group C'])
plt.ylabel('Values')
plt.title('Box Plot: Compare Distributions')
plt.grid(True, alpha=0.3)
plt.savefig('plot_09_boxplot.png')
plt.close()
print("Box plot saved")

# ---
# PIE CHART
# ---

# WHY: Show proportions and parts of a whole.
sizes = [30, 25, 20, 25]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
colors = ['red', 'blue', 'green', 'orange']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Pie Chart: Proportions')
plt.savefig('plot_10_pie.png')
plt.close()
print("Pie chart saved")

# ---
# COMBINATION AND TEXT ANNOTATION
# ---

# WHY: Mix plot types and annotate for clarity.
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.6, s=50, label='Data points')
z = np.polyfit(x, y, 1)  # Linear fit
p = np.poly1d(z)
plt.plot(x, p(x), 'r-', linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter with Trend Line')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_11_scatter_fit.png')
plt.close()
print("Scatter with fit line saved")

# Clean up
import os
for i in range(1, 12):
    try:
        os.remove(f'plot_{i:02d}_*.png')
    except:
        pass

# KEY TAKEAWAY:
# Line plots: trends over time or continuous relationships.
# Scatter plots: relationships between two variables, cluster detection.
# Bar plots: compare values across categorical groups.
# Histograms: distribution shapes, skewness, outliers.
# Box plots: quartiles and outlier detection across groups.
# Pie charts: part-to-whole proportions.
# Always label axes, add titles, and use legends for clarity.
# Color and markers aid interpretation and help distinguish groups.
