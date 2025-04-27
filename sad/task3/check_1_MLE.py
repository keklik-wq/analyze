import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import chi2, gaussian_kde
import seaborn as sns

def compute_statistic(samples, theta_0):
    """
    Вычисляет статистику для проверки гипотезы H0: θ = θ0.
    
    Параметры:
        samples (numpy array): Массив наблюдений X_1, X_2, ..., X_n.
        theta_0 (float): Значение θ из нулевой гипотезы.
    
    Возвращает:
        float: Значение статистики.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("Массив samples не может быть пустым.")
    
    # Оценка максимального правдоподобия
    theta_mle = np.sum(samples) / (5 * n)
    
    # Информация Фишера при θ = θ0
    I_theta0 = 5 / (theta_0**2)
    
    # Вычисление статистики
    statistic = n * (theta_mle - theta_0)**2 * I_theta0
    return statistic

def compute_statistic_2(samples, theta_0):
    """
    Вычисляет статистику для проверки гипотезы H0: θ = θ0.
    
    Параметры:
        samples (numpy array): Массив наблюдений X_1, X_2, ..., X_n.
        theta_0 (float): Значение θ из нулевой гипотезы.
    
    Возвращает:
        float: Значение статистики.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("Массив samples не может быть пустым.")
    
    # Оценка максимального правдоподобия
    theta_mle = np.sum(samples) / (5 * n)
    
    # Информация Фишера при θ = θ_mle
    I_theta_mle = 5 / (theta_mle**2)
    
    # Вычисление статистики
    statistic = n * (theta_mle - theta_0)**2 * I_theta_mle
    return statistic

def log_likelihood(samples, theta):
    """
    Вычисляет логарифмическую функцию правдоподобия для заданного theta.
    
    Параметры:
        samples (numpy array): Массив наблюдений X_1, X_2, ..., X_n.
        theta (float): Значение параметра theta.
    
    Возвращает:
        float: Логарифмическая функция правдоподобия.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("Массив samples не может быть пустым.")
    
    # Вычисляем логарифмическую функцию правдоподобия
    log_lik = np.sum(4 * np.log(samples) - np.log(math.factorial(4)) - 5 * np.log(theta) - samples / theta)
    return log_lik

def likelihood_ratio_test(samples, theta_0):
    """
    Вычисляет критерий отношения правдоподобия для проверки гипотезы H0: θ = θ0.
    
    Параметры:
        samples (numpy array): Массив наблюдений X_1, X_2, ..., X_n.
        theta_0 (float): Значение θ из нулевой гипотезы.
    
    Возвращает:
        float: Значение статистики Λ.
        float: p-value.
    """
    samples = np.array(samples)
    n = len(samples)
    if n == 0:
        raise ValueError("Массив samples не может быть пустым.")
    
    # Оценка максимального правдоподобия
    theta_mle = np.sum(samples) / (5 * n)
    
    # Логарифмическая функция правдоподобия при θ = θ0
    log_lik_theta0 = log_likelihood(samples, theta_0)
    
    # Логарифмическая функция правдоподобия при θ = θ_mle
    log_lik_theta_mle = log_likelihood(samples, theta_mle)
    
    # Критерий отношения правдоподобия
    Lambda = -2 * (log_lik_theta0 - log_lik_theta_mle)

    
    return Lambda

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
theta_0 = 0.2
x_max = 10  # Upper limit for the proposal distribution

def proposal_distribution():
    return np.random.uniform(0.1, x_max)  # Adjusted range to cover more of the PDF

M = np.max(pdf(np.linspace(1, x_max, 1000), theta_0))  # Compute M as the maximum of the PDF

# Rejection sampling
samples = []
num_samples = 50  # Number of samples

while len(samples) < num_samples:
    x = proposal_distribution()
    u = np.random.uniform(0.1, 1)
    
    if u <= pdf(x, theta_0) / (M * 1):  # g(x) = 1 for uniform distribution
        samples.append(x)

print(f"Number of samples: {len(samples)}")

# Plot the histogram of the samples
plt.hist(samples, bins=5, density=True, alpha=0.6, color='b', label='Sampled Data')
    
# Plot the true PDF for comparison
x = np.linspace(0.1, x_max, num_samples)
pdf_values = pdf(x, theta_0)
plt.plot(x, pdf_values, 'r-', lw=2, label='True PDF')

# Plot the kernel density estimate (KDE)
sns.kdeplot(samples, color='g', lw=2, label='KDE')

plt.title('Rejection Sampling with KDE')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

# Likelihood Ratio Test
n = len(samples)
theta_m = np.mean(samples) / 5  # MLE for theta

# Test statistic
test_statistic_criteria = ((np.mean(samples) / (5 * theta_0))**(5*n)) * np.exp(5*n*(1-np.mean(samples)/theta_0))
log_test_statistic_criteria = -2*np.log(test_statistic_criteria)

test_statistic_11 = (5 * n / (theta_0**2)) * (theta_m - theta_0)**2

aaa = np.mean(np.array(samples)**2)
bbb = (25 * n**3) / (aaa)
test_statistic_2 = (bbb) * (theta_m - theta_0)**2
# Critical value (chi-squared distribution, 1 degree of freedom, alpha = 0.1)
alpha = 0.1
df = 1
critical_value = chi2.ppf(1 - alpha, df=df)

test_statistic_1 = compute_statistic(samples=samples, theta_0=theta_0)
test_statistic_2 = compute_statistic_2(samples=samples, theta_0=theta_0)
likelihood_ratio_value = likelihood_ratio_test(samples=samples, theta_0=theta_0)
# p-value
p_value1 = 1 - chi2.cdf(test_statistic_1, df=df)
p_value2 = 1 - chi2.cdf(test_statistic_2, df=df)
p_value3 = 1 - chi2.cdf(likelihood_ratio_value, df=df)

# Output results
print(f"Значение theta_0: {theta_0}")
print(f"Значение theta_mle: {theta_m}")
print(f"Критическое значение (χ²(1, 1-α)): {critical_value}. Степени свободы: {df}")
print(f"p-value: {p_value1}")
print(f"Вычисленная статистика (5n/(θ0^2) * (θm - θ0)^2): {round(test_statistic_1,3)}")
# Decision
if test_statistic_1 > critical_value:
    print("Отвергаем нулевую гипотезу H0: θ = θ0.")
else:
    print("Нет оснований отвергать нулевую гипотезу H0: θ = θ0.")
print(f"-----------------------------")
print(f"Вычисленная статистика ((25 * n**3) / (np.mean(samples**2))) * (theta_m - theta_0)**2:")
print(f"p-value: {p_value2}")
print(f" -> {round(test_statistic_2,3)}")
# Decision
if test_statistic_2 > critical_value:
    print("Отвергаем нулевую гипотезу H0: θ = θ0.")
else:
    print("Нет оснований отвергать нулевую гипотезу H0: θ = θ0.")
print(f"-----------------------------")
print(f"p-value: {p_value3}")
print(f"Вычисленная статистика -2*np.log(((np.mean(samples) / (5 * theta_0))**(5*n)) * np.exp(5*n*(1-np.mean(samples)/theta_0)))")
print(f" -> {round(likelihood_ratio_value,3)}")
if likelihood_ratio_value > critical_value:
    print("Отвергаем нулевую гипотезу H0: θ = θ0.")
else:
    print("Нет оснований отвергать нулевую гипотезу H0: θ = θ0.")