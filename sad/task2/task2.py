import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from statsmodels.stats.weightstats import DescrStatsW

# Заданная плотность распределения
def density(x, theta):
    return (np.sqrt(theta) / np.sqrt(np.pi)) * np.exp(-theta * x**2)

# Логарифмическая функция правдоподобия
def log_likelihood(theta, data):
    return -np.sum(np.log(density(data, theta)))

# Метод Монте-Карло для генерации выборки
def monte_carlo_sampling(theta, M, N):
    Y = np.random.normal(0, 1, M)  # Вспомогательное распределение N(0, 1)
    p_Y = density(Y, theta)
    q_Y = np.exp(-Y**2 / 2) / np.sqrt(2 * np.pi)
    w = p_Y / q_Y
    w_normalized = w / np.sum(w)
    X = np.random.choice(Y, size=N, p=w_normalized)
    return X

# Произвольное значение параметра theta
theta_true = 2.0  # Истинное значение параметра

# Генерация выборки объёма N = 50
N = 50
M = 10000  # Размер вспомогательной выборки
data = monte_carlo_sampling(theta_true, M, N)

# Нахождение оценки theta методом максимального правдоподобия
result = minimize_scalar(log_likelihood, args=(data,), bounds=(0.01, 10), method='bounded')
theta_hat = result.x

# Асимптотический доверительный интервал для theta
# Информация Фишера: I(theta) = 1 / (2 * theta^2)
fisher_information = 1 / (2 * theta_hat**2)
se_theta = 1 / np.sqrt(N * fisher_information)  # Стандартная ошибка
z = stats.norm.ppf(0.95)  # Квантиль для уровня надёжности 0.95
ci_lower = theta_hat - z * se_theta
ci_upper = theta_hat + z * se_theta

# Эмпирическая функция распределения
def empirical_cdf(data, x):
    return np.mean(data <= x)

x_values = np.linspace(-5, 5, 1000)
ecdf_values = [empirical_cdf(data, x) for x in x_values]

# Ядерная оценка плотности
from scipy.stats import gaussian_kde
kde = gaussian_kde(data)
kde_values = kde(x_values)

# Визуализация
plt.figure(figsize=(12, 6))

# Гистограмма и ядерная оценка плотности
plt.subplot(1, 2, 1)
plt.hist(data, bins=15, density=True, alpha=0.6, color='g', label='Гистограмма')
plt.plot(x_values, kde_values, 'r-', label='Ядерная оценка плотности')
plt.plot(x_values, density(x_values, theta_true), 'b--', label='Истинная плотность')
plt.xlabel('X')
plt.ylabel('Плотность')
plt.title('Гистограмма и ядерная оценка плотности')
plt.legend()

# Эмпирическая функция распределения
plt.subplot(1, 2, 2)
plt.step(x_values, ecdf_values, where='post', label='Эмпирическая функция распределения')
plt.plot(x_values, stats.norm.cdf(x_values, scale=1/np.sqrt(2 * theta_true)), 'b--', label='Теоретическая функция распределения')
plt.xlabel('X')
plt.ylabel('F(X)')
plt.title('Эмпирическая функция распределения')
plt.legend()

plt.tight_layout()
plt.show()

# Вывод результатов
print(f"Истинное значение theta: {theta_true}")
print(f"Оценка theta (метод максимального правдоподобия): {theta_hat}")
print(f"Доверительный интервал для theta (0.95): ({ci_lower}, {ci_upper})")