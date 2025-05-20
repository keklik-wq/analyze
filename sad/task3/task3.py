import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, chi2, gaussian_kde, norm


# 1. Определите функцию плотности
def density(X, theta):
    """Плотность p(X|θ) = ((X^4)/(4! * θ^5)) * exp(-X / θ)"""
    return (X**4 / (np.math.factorial(4) * theta**5)) * np.exp(-X / theta)


# 2. Генерация выборки:
# Произвольно выбрать значение параметра θ для нулевой гипотезы
theta0 = 1.0
N = 500000
alpha = 0.1

# Сгенерируйте выборку из гамма-распределения (поскольку X/θ имеет гамма-распределение)
# X ~ Gamma(shape=5, scale=theta)
sample = np.random.gamma(shape=5, scale=theta0, size=N)

# 3. Критерий отношения правдоподобия (LRT)
# Вычислите выборочное среднее
X_bar = np.mean(sample)

# Вычислите оценку MLE
theta_hat = X_bar / 5

# Вычислите статистику отношения правдоподобия Lambda
Lambda = (X_bar / (5 * theta0)) ** (5 * N) * np.exp(5 * N * (1 - (X_bar / theta0)))

# Вычислите статистику -2logLambda
minus_2_log_Lambda = -2 * np.log(Lambda)

# Вычислите критическое значение для хи-квадрат
chi2_critical = chi2.ppf(1 - alpha, df=1)

# Вычислите P-значение
p_value_LRT = chi2.sf(minus_2_log_Lambda, df=1)

# 4. Критерий на основе MLE
# Вычислите статистику T
T = np.abs(theta_hat - theta0)

# В идеале: найдите распределение T и вычислите критическое значение
# или p-значение на основе этого распределения.
# В качестве аппроксимации: используйте нормальное приближение для theta_hat
# (только для демонстрационных целей).
# ВНИМАНИЕ: Это приближение и может быть неточным для небольших N.

# Оцените стандартную ошибку theta_hat (используя дельта-метод или бутстрэп)
# Здесь используем дельта-метод (асимптотический)
# Var(X_bar) = Var(X) / N = (shape * scale^2) / N = (5 * theta0^2) / N
# SE(theta_hat) = SE(X_bar / 5) = SE(X_bar) / 5 = sqrt(Var(X_bar)) / 5
SE_theta_hat = np.sqrt((5 * theta0**2) / N) / 5

# Вычислите z-статистику (стандартизованную T)
z_statistic = (theta_hat - theta0) / SE_theta_hat

# Вычислите p-значение (двустороннее)
p_value_MLE = 2 * (1 - np.abs(norm.cdf(z_statistic)))

# Вычислите критическое значение для z
z_critical = norm.ppf(1 - alpha / 2)

# 5. Ядерная оценка плотности (KDE)
kde = gaussian_kde(sample)
x_grid = np.linspace(min(sample), max(sample), 200)

# 6. Графики
plt.figure(figsize=(10, 6))
plt.plot(x_grid, kde(x_grid), label="Ядерная оценка плотности")
plt.hist(sample, bins=20, density=True, alpha=0.5, label="Гистограмма выборки")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.title("Ядерная оценка плотности сгенерированной выборки")
plt.legend()
plt.show()

# 7. Вывод результатов
print("Результаты:")
print(f"Выбранное значение θ0: {theta0}")
print(f"Выборочное среднее X_bar: {X_bar:.3f}")
print(f"Оценка MLE θ_hat: {theta_hat:.3f}")

print("\nКритерий отношения правдоподобия:")
print(f"Статистика Lambda: {Lambda:.3f}")
print(f"Статистика -2logLambda: {minus_2_log_Lambda:.3f}")
print(f"Критическое значение хи-квадрат: {chi2_critical:.3f}")
print(f"P-значение: {p_value_LRT:.3f}")
if minus_2_log_Lambda > chi2_critical:
    print("Отклоняем H0")
else:
    print("Не отклоняем H0")

print("\nКритерий на основе MLE (с нормальным приближением):")
print(f"Статистика T: {T:.3f}")
print(f"Стандартная ошибка SE(θ_hat): {SE_theta_hat:.3f}")
print(f"Z-статистика: {z_statistic:.3f}")
print(f"Критическое значение Z: {z_critical:.3f}")
print(f"P-значение: {p_value_MLE:.3f}")
if np.abs(z_statistic) > z_critical:
    print("Отклоняем H0 (приблизительно)")
else:
    print("Не отклоняем H0 (приблизительно)")
