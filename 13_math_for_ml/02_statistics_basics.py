# Revision Notes:
# Topic: Statistics Basics for ML
# Why it matters for AI/ML: Statistical understanding guides feature engineering,
# model evaluation, hypothesis testing, and interpretation of results.
# Distributions and relationships are foundational ML concepts.

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# ---
# DESCRIPTIVE STATISTICS
# ---

# WHY: Summarize data to understand distribution and identify anomalies.

data = np.array([35, 42, 48, 50, 52, 55, 58, 60, 65, 100])  # Include outlier

# Central tendency
# WHY: Understand typical values.
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=True).mode[0]

print("=== Descriptive Statistics ===")
print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode: {mode}")

# WHY use median over mean?
# - Median is robust to outliers (unaffected by 100 vs 50)
# - Mean is sensitive (heavily influenced by extreme values)

# Spread / Dispersion
# WHY: Measure variability to understand data consistency.
variance = np.var(data)
std_dev = np.std(data)
iqr = np.percentile(data, 75) - np.percentile(data, 25)

print(f"\nVariance: {variance:.2f}")
print(f"Std Dev: {std_dev:.2f}")
print(f"IQR (Interquartile Range): {iqr:.2f}")

# WHY: Low std dev = data clustered; high = spread out.
# IQR = Q3 - Q1 = middle 50% range; robust to outliers.

# Percentiles and quantiles
# WHY: Understand data at specific thresholds (e.g., top 10%, bottom quartile).
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
print(f"\nQ1 (25th percentile): {q1:.2f}")
print(f"Q3 (75th percentile): {q3:.2f}")

# Skewness and Kurtosis
# WHY: Describe distribution shape (skew = asymmetry, kurtosis = tailedness).
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)
print(f"\nSkewness: {skewness:.2f} (right-skewed; outlier on right)")
print(f"Kurtosis: {kurtosis:.2f} (heavier tails than normal)")

# ---
# DISTRIBUTIONS
# ---

# WHY: Real data often follows known distributions; recognizing them enables proper analysis.

# Normal (Gaussian) Distribution
# WHY: Most common in statistics; many ML algorithms assume normality.
from scipy.stats import norm

mu = 100  # Mean
sigma = 15  # Standard deviation
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
pdf = norm.pdf(x, mu, sigma)

print("\n" + "="*60)
print("\n=== Normal Distribution ===")
print(f"Mean: {mu}, Std Dev: {sigma}")

# Probability that random draw < 110
prob_less_110 = norm.cdf(110, mu, sigma)
print(f"P(X < 110): {prob_less_110:.4f} (~{prob_less_110*100:.1f}%)")

# Generate samples
samples = np.random.normal(mu, sigma, 1000)
print(f"Generated samples mean: {samples.mean():.2f}, std: {samples.std():.2f}")

# Uniform Distribution
# WHY: Used in initialization, sampling, and random baseline comparisons.
uniform_samples = np.random.uniform(0, 1, 1000)
print("\n=== Uniform Distribution ===")
print(f"Uniform[0,1] - mean: {uniform_samples.mean():.2f}, std: {uniform_samples.std():.2f}")

# Exponential Distribution
# WHY: Models waiting times and rare events (e.g., time until failure).
exp_samples = np.random.exponential(2, 1000)
print("\n=== Exponential Distribution ===")
print(f"Exponential(λ=0.5) - mean: {exp_samples.mean():.2f}, std: {exp_samples.std():.2f}")

# ---
# CORRELATION AND COVARIANCE
# ---

# WHY: Understand relationships between variables (features in ML).

# Create two variables with relationship
x = np.array([1, 2, 3, 4, 5])
y = 2 * x + np.random.randn(5) * 0.5  # y ≈ 2x with noise

# Covariance
# WHY: Measure of joint variability; sign indicates relationship direction.
cov_matrix = np.cov(x, y)
print("\n" + "="*60)
print("\n=== Covariance and Correlation ===")
print(f"Covariance matrix:\n{cov_matrix}")
print(f"Cov(X, Y): {cov_matrix[0, 1]:.4f}")

# Correlation (Pearson)
# WHY: Standardized covariance [-1, 1]; easier to interpret.
pearson_r = np.corrcoef(x, y)[0, 1]
print(f"\nPearson correlation: {pearson_r:.4f}")
print("Interpretation: r ≈ 1 = strong positive relationship")

# Spearman Correlation
# WHY: Measure ordinal (rank-based) relationship; robust to outliers.
spearman_r, p_value = stats.spearmanr(x, y)
print(f"Spearman correlation: {spearman_r:.4f}")
print(f"P-value: {p_value:.4f} (is relationship statistically significant?)")

# ---
# Z-SCORES AND STANDARDIZATION
# ---

# WHY: Scale variables to comparable units; remove units; identify outliers.

data = np.array([32, 45, 52, 50, 49, 55, 70, 48, 105, 90])
mean = np.mean(data)
std = np.std(data)

# Z-score: how many std deviations from mean
# WHY: Standardized score; values > |3| are extreme outliers.
z_scores = (data - mean) / std
print("\n" + "="*60)
print("\n=== Z-Scores ===")
print(f"Original data: {data}")
print(f"Z-scores: {z_scores.round(2)}")
print(f"Extreme outliers (|z| > 3): {data[np.abs(z_scores) > 3]}")

# Standardization for modeling
# WHY: Many ML algorithms perform better with standardized features.
X_standardized = (data - mean) / std
print(f"\nStandardized mean: {X_standardized.mean():.4f} (≈ 0)")
print(f"Standardized std: {X_standardized.std():.4f} (≈ 1)")

# ---
# HYPOTHESIS TESTING
# ---

# WHY: Determine if observed differences are statistically significant or random.

group1 = np.array([85, 88, 90, 92, 91, 87])
group2 = np.array([78, 80, 82, 84, 81, 79])

# Independent t-test
# WHY: Compare means of two independent groups.
t_stat1, p_value1 = stats.ttest_ind(group1, group2)
print("\n" + "="*60)
print("\n=== Hypothesis Testing ===")
print("Group 1 mean:", group1.mean())
print("Group 2 mean:", group2.mean())
print(f"T-statistic: {t_stat_1:.4f}")
print(f"P-value: {p_value1:.4f}")
if p_value1 < 0.05:
    print("Result: Significant difference (reject null hypothesis)")
else:
    print("Result: No significant difference (fail to reject null hypothesis)")

# Paired t-test
# WHY: Compare before/after or matched pairs.
before = np.array([100, 105, 98, 102, 99])
after = np.array([102, 108, 100, 105, 101])
t_stat2, p_value2 = stats.ttest_rel(before, after)
print(f"\nPaired t-test (before vs after):")
print(f"T-statistic: {t_stat2:.4f}, P-value: {p_value2:.4f}")

# Chi-square test for categorical data
# WHY: Test association between categorical variables.
# Contingency table: [observed counts]
observed = np.array([[10, 20], [30, 40]])
chi2, p_chi, dof, expected = stats.chi2_contingency(observed)
print(f"\nChi-square test for independence:")
print(f"Chi-square statistic: {chi2:.4f}, P-value: {p_chi:.4f}")

# ---
# PROBABILITY CALCULATIONS
# ---

# WHY: Compute likelihoods for decision-making and model interpretation.

# Birthday problem: probability that 2+ people share birthday
n_people = 23
prob_all_different = 1
for i in range(1, n_people):
    prob_all_different *= (365 - i) / 365
prob_shared = 1 - prob_all_different

print("\n" + "="*60)
print("\n=== Probability ===")
print(f"With {n_people} people, P(≥2 share birthday): {prob_shared:.4f} (~{prob_shared*100:.1f}%)")

# Bayes' Theorem
# WHY: Update probability given new evidence (foundation of Bayesian ML).
# P(A|B) = P(B|A) * P(A) / P(B)

# Example: Disease diagnosis
P_disease = 0.001  # 0.1% of population has disease
P_test_positive_given_disease = 0.99  # 99% accuracy if have disease
P_test_positive_no_disease = 0.05  # 5% false positive rate

P_test_positive = (P_test_positive_given_disease * P_disease + 
                   P_test_positive_no_disease * (1 - P_disease))
P_disease_given_positive = (P_test_positive_given_disease * P_disease) / P_test_positive

print(f"\nDiseaseTest:")
print(f"P(test+|disease) = {P_disease_given_positive:.4f}")
print(f"Even with positive test, only ~{P_disease_given_positive*100:.1f}% likely to have disease")
print("(Low prevalence makes false positives misleading!)")

# ---
# PRACTICAL ML SCENARIOS
# ---

print("\n" + "="*60)
print("ML Scenarios:\n")

# Scenario 1: Outlier detection
# WHY: Remove extreme values that might skew models.
data = np.random.normal(100, 15, 100)
data = np.append(data, [200, -100])  # Add outliers
z_scores = np.abs((data - np.mean(data)) / np.std(data))
outliers = data[z_scores > 3]
print(f"Detected {len(outliers)} outliers: {outliers}")

# Scenario 2: Feature scaling before modeling
# WHY: ML algorithms sensitive to feature magnitude.
X = np.array([[1000, 50], [2000, 100], [3000, 150]])
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
print(f"\nBefore scaling:\n{X}")
print(f"After scaling:\n{X_scaled.round(3)}")

# Scenario 3: Checking feature independence
# WHY: Correlated features may hurt model interpretability.
df = pd.DataFrame({'Age': np.random.randn(100), 'Income': np.random.randn(100)})
correlation = df.corr().iloc[0, 1]
print(f"\nCorrelation between Age and Income: {correlation:.4f} (is it problematic?)")

# KEY TAKEAWAY:
# Mean vs Median: use median for skewed data with outliers.
# Std Dev and IQR: measure spread.
# Normal distribution: most common; test for normality before analysis.
# Correlation: measure relationships; use Pearson or Spearman appropriately.
# Z-scores: standardize and detect outliers.
# Hypothesis tests: determine statistical significance.
# Bayes: update beliefs with evidence (foundation of probabilistic ML).
