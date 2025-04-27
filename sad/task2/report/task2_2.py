import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde


# 1. Определите функцию плотности
def density(X, theta):
  """Плотность p(X|θ) = (√θ/√π) * exp(-X^2 * θ)"""
  return (np.sqrt(theta) / np.sqrt(np.pi)) * np.exp(-X**2 * theta)

# 2. Правила генерации выборки:
# Произвольно выбрать значение параметра θ
theta_true = 2.0  # Выберите произвольное значение для θ
N = 50  # Размер выборки

# Сгенерируйте выборку объёма N = 50
# Обратите внимание: не существует простого способа напрямую сгенерировать выборку из этой плотности.
# Здесь мы используем метод отклонения для генерации выборки.
def rejection_sampling(N, theta):
    """Генерирует выборку из плотности p(X|θ) с использованием отбраковки."""
    samples = []
    while len(samples) < N:
        # Предложите из распределения Коши (тяжелые хвосты)
        Y = np.random.standard_cauchy()
        # Вычислите константу отклонения M (требуется верхняя граница плотности)
        M = np.sqrt(theta) / (np.sqrt(np.pi) * density(0, theta))  # Наихудший случай в 0
        U = np.random.uniform(0, 1)
        if U <= density(Y, theta) / (M * (1 / (np.pi * (1 + Y**2)))):
            samples.append(Y)
    return np.array(samples)

sample = rejection_sampling(N, theta_true)

# 3. Эмпирическая функция распределения (ECDF)
ecdf = ECDF(sample)

# 4. Ядерная оценка плотности (KDE)

kde = gaussian_kde(sample)
x_grid = np.linspace(min(sample), max(sample), 200)  # Сетка для построения графиков KDE

# 5. Оценка параметра θ (bθ)
# Оценка максимального правдоподобия (MLE)
# Логарифмическая правдоподобность: l(θ) = Σ [0.5*log(θ) - X_i^2 * θ] - N*0.5*log(pi)
# Взяв производную по θ и приравняв к 0, получим: θ_hat = N / (2 * Σ X_i^2)

theta_hat = N / (2 * np.sum(sample**2))

# 6. Двусторонний асимптотический доверительный интервал надёжности 0.95 для параметра θ
alpha = 0.05  # Уровень значимости
z_alpha_2 = norm.ppf(1 - alpha/2)  # Квантиль Z для 0.975

# Асимптотический доверительный интервал
# Обратите внимание: это упрощенный асимптотический интервал, и его точность может быть ограничена,
# особенно для небольших размеров выборки или ненормальных распределений.
# Для более надежных интервалов можно использовать бутстрэп или профилирование правдоподобия.
std_err = theta_hat / np.sqrt(N)  # Оценка стандартной ошибки (упрощенная)
lower_bound = theta_hat - z_alpha_2 * std_err
upper_bound = theta_hat + z_alpha_2 * std_err

# 7. Графики
plt.figure(figsize=(12, 6))

# a. ECDF
plt.subplot(1, 2, 1)
plt.plot(ecdf.x, ecdf.y, label='ECDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Эмпирическая функция распределения')

# b. KDE
plt.subplot(1, 2, 2)
plt.plot(x_grid, kde(x_grid), label='Ядерная оценка плотности')
plt.hist(sample, bins=20, density=True, alpha=0.5, label='Гистограмма выборки')  # Наложите гистограмму
plt.xlabel('x')
plt.ylabel('Плотность')
plt.title('Ядерная оценка плотности')

plt.tight_layout()
plt.show()

# 8. Вывод
print(f"Оценка параметра θ (θ_hat): {theta_hat:.3f}")
print(f"Приблизительный 95% доверительный интервал для θ: ({lower_bound:.3f}, {upper_bound:.3f})")