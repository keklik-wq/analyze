import numpy as np
from scipy.stats import norm

def generate_and_estimate(theta, N=50, alpha=0.05):
    """
    Генерирует выборку из заданного распределения, находит оценку максимального правдоподобия (ОМП)
    и строит асимптотический доверительный интервал.

    Args:
        theta (float): Истинное значение параметра θ.
        N (int): Размер выборки.
        alpha (float): Уровень значимости (1 - надёжность).

    Returns:
        tuple: (theta_hat, ci_lower, ci_upper) - ОМП и границы доверительного интервала.
    """

    # 1. Генерация выборки
    # Поскольку у нас нет стандартной функции для генерации из этого распределения,
    # сгенерируем выборку, используя метод обратной функции (если это возможно)
    # или метод принятия-отклонения.  В данном случае, проще всего использовать метод принятия-отклонения.

    def pdf(x, theta):
        """Плотность вероятности."""
        return (np.sqrt(theta) / np.sqrt(np.pi)) * np.exp(-x**2 * theta)

    # Выбираем M > max(pdf(x)) для всех x.  Например, M = sqrt(theta) / sqrt(pi).
    M = np.sqrt(theta) / np.sqrt(np.pi)

    sample = []
    while len(sample) < N:
        x = np.random.uniform(-5, 5)  # Выбираем диапазон для x, чтобы покрыть большую часть плотности
        u = np.random.uniform(0, M)

        if u <= pdf(x, theta):
            sample.append(x)

    sample = np.array(sample)  # Преобразуем в numpy array

    # 2. Вычисление ОМП (theta_hat)
    theta_hat = N / (2 * np.sum(sample**2))

    # 3. Вычисление доверительного интервала
    z_alpha_2 = norm.ppf(1 - alpha / 2)  # Квантиль стандартного нормального распределения
    ci_lower = theta_hat * (1 - z_alpha_2 * np.sqrt(2 / N))
    ci_upper = theta_hat * (1 + z_alpha_2 * np.sqrt(2 / N))

    return theta_hat, ci_lower, ci_upper


# Пример использования:
theta_true = 2.0  # Выбираем произвольное значение theta
theta_hat, ci_lower, ci_upper = generate_and_estimate(theta_true)

print(f"Истинное значение theta: {theta_true}")
print(f"Оценка theta (theta_hat): {theta_hat}")
print(f"Доверительный интервал (95%): [{ci_lower}, {ci_upper}]")

# Проверка попадания истинного значения в доверительный интервал
if ci_lower <= theta_true <= ci_upper:
    print("Истинное значение theta попало в доверительный интервал.")
else:
    print("Истинное значение theta НЕ попало в доверительный интервал.")