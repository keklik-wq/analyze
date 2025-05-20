import numpy as np
import matplotlib.pyplot as plt
import math


# Define the PDF
def pdf(X, theta):
    """
    Calculate the probability density function (PDF) for the given X and theta.

    Parameters:
        X (float or numpy array): The value(s) at which to evaluate the PDF.
        theta (float): The parameter of the distribution (theta > 0).

    Returns:
        float or numpy array: The PDF value(s) at X.
    """
    # Check for valid inputs
    if theta <= 0:
        raise ValueError("theta must be greater than 0.")
    if np.any(X <= 0):
        raise ValueError("X must be greater than 0.")

    # Compute the PDF
    coefficient = X**4 / (math.factorial(4) * theta**5)
    exponential_term = np.exp(-X / theta)
    return coefficient * exponential_term


# Define the proposal distribution (uniform over a larger range)
theta = 2
x_max = 20  # Upper limit for the proposal distribution


def proposal_distribution():
    return np.random.uniform(0.1, x_max)  # Adjusted range to cover more of the PDF


M = np.max(
    pdf(np.linspace(1, x_max, 1000), theta)
)  # Compute M as the maximum of the PDF

# Rejection sampling
samples = []
num_samples = 50  # Increased number of samples for better results

while len(samples) < num_samples:
    x = proposal_distribution()
    u = np.random.uniform(0.1, 1)

    if u <= pdf(x, theta) / (M * 1):  # g(x) = 1 for uniform distribution
        samples.append(x)
print(len(samples))
# Plot the histogram of the samples
plt.hist(samples, bins=5, density=True, alpha=0.6, color="b", label="Sampled Data")

# Plot the true PDF for comparison
x = np.linspace(0.1, x_max, 50)
pdf_values = pdf(x, theta)
plt.plot(x, pdf_values, "r-", lw=2, label="True PDF")

plt.title("Rejection Sampling")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()
