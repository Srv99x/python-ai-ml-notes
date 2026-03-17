# Revision Notes:
# Topic: Matplotlib Subplots and Figure Customization
# Why it matters for AI/ML: Professional visualizations aid presentations and reports.
# Subplots enable side-by-side comparisons; customization improves clarity and aesthetics.
# These skills are essential for communicating findings to stakeholders.

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# ---
# BASIC SUBPLOTS
# ---

# WHY: Display multiple plots together for comparison or comprehensive analysis.

# Create 2x2 grid of subplots
# WHY: Compare four different aspects of data in one figure.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line plot
axes[0, 0].plot([1, 2, 3, 4], [1, 4, 2, 3], marker='o')
axes[0, 0].set_title('Line Plot')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')

# Plot 2: Scatter plot
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50), alpha=0.6)
axes[0, 1].set_title('Scatter Plot')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')

# Plot 3: Histogram
axes[1, 0].hist(np.random.randn(1000), bins=30, edgecolor='black')
axes[1, 0].set_title('Histogram')
axes[1, 0].set_xlabel('Values')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Bar plot
axes[1, 1].bar(['A', 'B', 'C', 'D'], [10, 24, 36, 18], color='steelblue')
axes[1, 1].set_title('Bar Plot')
axes[1, 1].set_ylabel('Values')

plt.tight_layout()  # Prevent label overlap
plt.savefig('subplot_2x2.png')
plt.close()
print("2x2 subplot saved")

# ---
# DIFFERENT SUBPLOT SIZES
# ---

# WHY: Emphasize certain plots by making them larger.

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Fill subplots
for i in range(2):
    for j in range(3):
        axes[i, j].plot(np.random.cumsum(np.random.randn(50)))
        axes[i, j].set_title(f'Plot {i*3 + j + 1}')

plt.suptitle('Multiple Subplots (2x3)', fontsize=16)  # Overall title
plt.tight_layout()
plt.savefig('subplot_2x3.png')
plt.close()
print("2x3 subplot saved")

# ---
# GRIDSPEC: CUSTOM SUBPLOT LAYOUT
# ---

# WHY: Create complex layouts where subplots have different sizes.

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(3, 3, figure=fig)

# Large plot spanning 2x2
ax_large = fig.add_subplot(gs[0:2, 0:2])
ax_large.plot(np.cumsum(np.random.randn(100)))
ax_large.set_title('Large Plot (spans 2x2)')

# Small plots
ax_small1 = fig.add_subplot(gs[0, 2])
ax_small1.hist(np.random.randn(100), bins=15)
ax_small1.set_title('Small 1')

ax_small2 = fig.add_subplot(gs[1, 2])
ax_small2.scatter(np.random.randn(50), np.random.randn(50))
ax_small2.set_title('Small 2')

# Bottom plot spanning full width
ax_bottom = fig.add_subplot(gs[2, :])
ax_bottom.bar(['A', 'B', 'C', 'D', 'E'], np.random.rand(5))
ax_bottom.set_title('Bottom Plot (full width)')

plt.tight_layout()
plt.savefig('subplot_gridspec.png')
plt.close()
print("GridSpec subplot saved")

# ---
# FIGURE CUSTOMIZATION
# ---

# WHY: Improve appearance and readability for presentations and publications.

fig, ax = plt.subplots(figsize=(12, 6))

# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot with styling
# WHY: Different line styles, widths, and markers for clarity.
ax.plot(x, y1, label='sin(x)', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
ax.plot(x, y2, label='cos(x)', linewidth=2.5, linestyle='--', marker='s', markersize=4, alpha=0.8)
ax.plot(x, y1 * y2, label='sin(x)*cos(x)', linewidth=2, linestyle=':', color='green')

# Labels and title
# WHY: Clear self-explanatory figure for audience.
ax.set_xlabel('Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=14, fontweight='bold')
ax.set_title('Trigonometric Functions', fontsize=16, fontweight='bold', pad=20)

# Grid
# WHY: Easier value reading.
ax.grid(True, alpha=0.3, linestyle='--')

# Legend
# WHY: Identify each line.
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Axis limits
# WHY: Focus on relevant range.
ax.set_xlim(0, 10)
ax.set_ylim(-1.2, 1.2)

# Add annotations
# WHY: Highlight key points.
max_idx = np.argmax(y1)
ax.annotate(f'Max: {y1[max_idx]:.2f}', xy=(x[max_idx], y1[max_idx]),
            xytext=(x[max_idx], y1[max_idx] + 0.3),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

# Background color
# WHY: Enhance visual appeal.
ax.set_facecolor('#eeeeee')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('customized_plot.png')
plt.close()
print("Customized plot saved")

# ---
# SHARED AXES
# ---

# WHY: Compare plots with same scale or coordinate system.

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# Generate different data
data1 = np.random.randn(100) + 50
data2 = np.random.randn(100) + 55
data3 = np.random.randn(100) + 48

# All histograms use same y-axis
axes[0].hist(data1, bins=20, color='red', alpha=0.7)
axes[0].set_title('Group A')
axes[0].set_ylabel('Frequency')

axes[1].hist(data2, bins=20, color='blue', alpha=0.7)
axes[1].set_title('Group B')

axes[2].hist(data3, bins=20, color='green', alpha=0.7)
axes[2].set_title('Group C')

plt.suptitle('Shared Y-Axis', fontsize=14)
plt.tight_layout()
plt.savefig('subplot_shared_y.png')
plt.close()
print("Shared axis subplot saved")

# ---
# SECONDARY AXIS
# ---

# WHY: Plot two variables with different scales on same figure.

fig, ax1 = plt.subplots(figsize=(10, 6))

# First y-axis
x = np.linspace(0, 10, 100)
y1 = np.sin(x) * 100  # Large scale
ax1.plot(x, y1, 'b-', linewidth=2, label='sin(x) * 100')
ax1.set_xlabel('X')
ax1.set_ylabel('sin(x) * 100', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')

# Second y-axis
ax2 = ax1.twinx()
y2 = np.exp(-x) * 10  # Small scale
ax2.plot(x, y2, 'r-', linewidth=2, label='exp(-x) * 10')
ax2.set_ylabel('exp(-x) * 10', color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Dual Y-Axes')
plt.tight_layout()
plt.savefig('secondary_axis.png')
plt.close()
print("Secondary axis plot saved")

# ---
# STYLE AND COLORS
# ---

# WHY: Professional appearance and consistency.

# Available styles: 'seaborn', 'ggplot', 'bmh', 'fivethirtyeight', etc.
plt.style.use('seaborn-v0_8-darkgrid')

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 50)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for i, color in enumerate(colors):
    y = np.sin(x + i * 0.5)
    ax.plot(x, y, label=f'Series {i+1}', linewidth=2, color=color, marker='o', markersize=3)

ax.set_title('Styled Plot', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.tight_layout()
plt.savefig('styled_plot.png')
plt.close()
print("Styled plot saved")

# Reset style
plt.style.use('default')

# Clean up
import os
for fname in ['subplot_2x2.png', 'subplot_2x3.png', 'subplot_gridspec.png', 
              'customized_plot.png', 'subplot_shared_y.png', 'secondary_axis.png', 'styled_plot.png']:
    try:
        os.remove(fname)
    except:
        pass

# KEY TAKEAWAY:
# plt.subplots() creates grid layouts; use tight_layout() to prevent overlap.
# GridSpec enables complex custom layouts with different subplot sizes.
# Customization: font sizes, colors, markers, line styles, annotations improve clarity.
# Secondary y-axes show two variables with different scales.
# Shared axes enable side-by-side comparisons with consistent scales.
# Professional visualizations use consistent styling and clear labeling.
