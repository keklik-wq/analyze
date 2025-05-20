import numpy as np


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
    statistic = n * (theta_mle - theta_0) ** 2 * I_theta0
    return statistic


# Пример использования
samples = np.array([1.2, 2.3, 3.4, 4.5, 5.6])  # Замените на ваши данные
theta_0 = 2  # Значение из нулевой гипотезы

statistic_value = compute_statistic(samples, theta_0)
print(f"Значение статистики: {statistic_value}")
