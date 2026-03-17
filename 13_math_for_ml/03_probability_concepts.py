# Revision Notes:
# Topic: Probability Concepts in Code
# Why it matters for AI/ML: Probabilistic thinking underlies classification, generative models,
# uncertainty quantification, and Bayesian inference. Random sampling enables testing and validation.
# Understanding distributions and sampling is foundation for statistical ML.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# ---
# RANDOM VARIABLES AND DISTRIBUTIONS
# ---

# WHY: Models represent uncertainty and randomness inherent in real data.

# Discrete distribution example: coin flips
# WHY: Model binary outcomes (heads vs tails, positive vs negative).
n_flips = 1000
coin_flips = np.random.binomial(n=1, p=0.5, size=n_flips)
heads_pct = coin_flips.sum() / n_flips
print("=== Discrete Distribution: Coin Flips ===")
print(f"Heads percentage from {n_flips} flips: {heads_pct:.2%}")

# Continuous distribution: normal distribution
# WHY: Heights, measurement errors, test scores follow normal distribution.
heights = np.random.normal(loc=170, scale=10, size=1000)  # loc=mean, scale=std
print(f"\n=== Continuous Distribution: Heights ===")
print(f"Mean: {heights.mean():.2f} cm")
print(f"Std: {heights.std():.2f} cm")
print(f"Min: {heights.min():.2f} cm, Max: {heights.max():.2f} cm")

# ---
# PROBABILITY MASS FUNCTION (PMF) AND PROBABILITY DENSITY FUNCTION (PDF)
# ---

# WHY: PMF for discrete; PDF for continuous.

# PMF example: rolling a die
# WHY: Each outcome (1-6) has equal probability.
die_rolls = np.random.randint(1, 7, size=1000)
for face in range(1, 7):
    prob = (die_rolls == face).sum() / len(die_rolls)
    print(f"P(die = {face}): {prob:.3f}")

# PDF example: normal distribution
# WHY: Continuous probability at any point.
from scipy.stats import norm
x = np.linspace(-3, 3, 100)
pdf = norm.pdf(x)  # Standard normal: mean=0, std=1
cdf = norm.cdf(x)  # Cumulative: P(X <= x)

print(f"\n=== Normal Distribution PDF/CDF ===")
print(f"PDF at x=0: {norm.pdf(0):.4f} (peak of bell curve)")
print(f"CDF at x=0: {norm.cdf(0):.4f} (50% of probability below mean)")
print(f"CDF at x=1: {norm.cdf(1):.4f} (68% below 1 std from mean)")
print(f"CDF at x=2: {norm.cdf(2):.4f} (95% below 2 stds from mean)")

# Inverse CDF (Quantile function)
# WHY: Find value x such that P(X <= x) = p
x_at_95_pct = norm.ppf(0.95)
print(f"Value at 95th percentile: {x_at_95_pct:.4f}")

# ---
# EXPECTATIONS AND VARIANCE
# ---

# WHY: E[X] = average value (center); Var[X] = spread.

# For discrete: E[X] = Σ x * P(x)
# WHY: Weighted average by probabilities.
x_values = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6] * 6)  # Fair die
expected_value = (x_values * probabilities).sum()
print(f"\n=== Expectation and Variance ===")
print(f"E[fair die]: {expected_value:.2f} (should be 3.5)")

# For continuous: E[X] = ∫ x * f(x) dx
# WHY: Scipy integrates for us; expectation of normal is its mean.
x_continuous = np.linspace(-10, 10, 1000)
pdf_values = norm.pdf(x_continuous, loc=5, scale=2)
expected = (x_continuous * pdf_values).mean()
print(f"E[Normal(μ=5, σ=2)]: {expected:.2f} (near 5)")

# Variance: Var[X] = E[(X - E[X])^2]
# WHY: Measure of spread; high variance = uncertain predictions.
samples = np.random.normal(loc=100, scale=15, size=1000)
variance = np.var(samples)
std_dev = np.std(samples)
print(f"Var[X]: {variance:.2f}")
print(f"Std[X]: {std_dev:.2f}")

# ---
# JOINT AND CONDITIONAL PROBABILITY
# ---

# WHY: Model relationships between events and variables.

# Joint probability: P(A and B)
# WHY: Probability of two events happening together.
# Example: P(rain AND we go outside)
P_rain = 0.3
P_go_given_rain = 0.2  # Conditional probability
P_rain_and_go = P_rain * P_go_given_rain  # Chain rule
print(f"\n=== Joint and Conditional Probability ===")
print(f"P(rain) = {P_rain}")
print(f"P(go|rain) = {P_go_given_rain}")
print(f"P(rain AND go) = {P_rain_and_go}")

# Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
# WHY: Update beliefs given evidence (foundation of Bayesian inference).
# Example: Spam detection
P_spam = 0.1  # 10% of emails are spam
P_word_given_spam = 0.8  # 80% of spam contains word "Buy"
P_word_given_ham = 0.05  # 5% of legitimate emails contain "Buy"

P_word = (P_word_given_spam * P_spam + P_word_given_ham * (1 - P_spam))
P_spam_given_word = (P_word_given_spam * P_spam) / P_word

print(f"\nSpam Detection:")
print(f"P(spam) = {P_spam}")
print(f"P('Buy' in email | spam) = {P_word_given_spam}")
print(f"P('Buy' in email | ham) = {P_word_given_ham}")
print(f"P(spam | 'Buy' in email) = {P_spam_given_word:.4f}")
print("Even with 'Buy', only ~15% likely spam because spam is rare")

# ---
# SAMPLING AND MONTE CARLO
# ---

# WHY: Simulate complex systems; estimate integrals and expectations.

# Estimate π using Monte Carlo
# WHY: Generate random points in unit square; count if in circle.
n_samples = 100000
x = np.random.uniform(-1, 1, n_samples)
y = np.random.uniform(-1, 1, n_samples)
distances = np.sqrt(x**2 + y**2)
in_circle = distances <= 1
pi_estimate = 4 * in_circle.sum() / n_samples

print(f"\n=== Monte Carlo Estimation ===")
print(f"Estimated π: {pi_estimate:.4f} (actual: {np.pi:.4f})")
print(f"Error: {abs(pi_estimate - np.pi):.4f}")

# Importance sampling
# WHY: Sample from easy distribution; weight by importance.
# Example: Estimate probability of extreme event using biased sampling.
normal_samples = np.random.normal(loc=0, scale=1, size=10000)
extreme_events = (normal_samples > 5).sum()
print(f"\nP(X > 5) with normal sampling: {extreme_events / 10000}")
print("Using standard normal: very few samples in tail!")

# Importance sampling: sample from shifted distribution
shifted_samples = np.random.normal(loc=5, scale=1, size=10000)
weights = norm.pdf(shifted_samples) / norm.pdf(shifted_samples, loc=5)
p_extreme_importance = (weights[shifted_samples > 5].sum() / weights.sum())
print(f"P(X > 5) with importance sampling: {p_extreme_importance:.6f}")

# ---
# PROBABILITY DISTRIBUTIONS FOR ML
# ---

# WHY: Different distributions model different phenomena in ML.

# Bernoulli: binary outcome
# WHY: Classification with 2 classes.
p_heads = 0.6
bernoulli_trials = np.random.binomial(n=1, p=p_heads, size=1000)
print(f"\n=== Common Distributions for ML ===")
print(f"Bernoulli(p={p_heads}): {bernoulli_trials.mean():.3f} (≈ p)")

# Binomial: number of successes in n trials
# WHY: Classification with multiple trials; vote counting.
successes = np.random.binomial(n=10, p=0.5, size=1000)
print(f"Binomial(n=10, p=0.5) mean: {successes.mean():.2f} (≈ 5)")

# Poisson: count of rare events in time/space
# WHY: Model number of defects, arrivals, mutations.
events = np.random.poisson(lam=3, size=1000)
print(f"Poisson(λ=3) mean: {events.mean():.2f} (≈ 3)")

# Exponential: time until next event
# WHY: Model waiting times, lifetimes, delays.
waiting_times = np.random.exponential(scale=2, size=1000)
print(f"Exponential(λ=0.5) mean: {waiting_times.mean():.2f} (≈ 2)")

# Dirichlet: multinomial probabilities (generalization of Beta)
# WHY: Topic modeling, multiclass posteriors.
alpha = [1, 1, 1]  # Uniform prior over 3 categories
probabilities_dist = np.random.dirichlet(alpha, size=5)
print(f"Dirichlet (normalized): each row sums to 1")
print(f"Sample:\n{probabilities_dist.round(3)}")

# ---
# PRACTICAL ML SCENARIOS
# ---

print(f"\n{'='*60}")
print("ML Scenarios:\n")

# Scenario 1: Classification with predicted probabilities
# WHY: Convert model outputs to interpretable probabilities.
predicted_scores = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.4])
probabilities = 1 / (1 + np.exp(-predicted_scores))  # Sigmoid
predictions = (probabilities > 0.5).astype(int)
print("Classification predictions:")
for i, (score, prob, pred) in enumerate(zip(predicted_scores, probabilities, predictions)):
    print(f"  Sample {i}: score={score:.2f}, P(class=1)={prob:.3f}, prediction={pred}")

# Scenario 2: Confidence intervals for uncertainty quantification
# WHY: Report not just prediction but also confidence range.
sample_mean = 100
sample_std = 15
confidence = 0.95
z_critical = norm.ppf((1 + confidence) / 2)
margin_of_error = z_critical * (sample_std / np.sqrt(100))
ci_lower, ci_upper = sample_mean - margin_of_error, sample_mean + margin_of_error
print(f"\n95% Confidence interval for mean: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Scenario 3: Likelihood for model comparison
# WHY: Evaluate how well model fits data.
observations = np.array([10, 12, 11, 13, 12, 11, 14])
likelihood_poisson_3 = np.prod(stats.poisson.pmf(observations, mu=3))
likelihood_poisson_12 = np.prod(stats.poisson.pmf(observations, mu=12))
print(f"\nLikelihood (μ=3): {likelihood_poisson_3:.2e}")
print(f"Likelihood (μ=12): {likelihood_poisson_12:.2e}")
print("μ=12 fits the data (average ≈ 12) much better than μ=3")

# KEY TAKEAWAY:
# PMF for discrete; PDF for continuous distributions.
# Expectation: weighted average; Variance: measure of spread.
# Joint P(A,B) = P(A|B) * P(B); Bayes updates beliefs.
# Monte Carlo: simulate complex distributions by random sampling.
# Importance sampling: efficiently sample from hard distributions.
# Each distribution models different phenomena; choose appropriately.
# Probabilistic interpretation of models enables uncertainty quantification.
