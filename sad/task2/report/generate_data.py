import numpy as np
import matplotlib.pyplot as plt

# Оценка параметра theta, полученная методом максимального правдоподобия
theta_hat = 2.0  # Пример значения, полученного методом МП

# Размер вспомогательной выборки
M = 1000
# Размер конечной выборки
N = 50


# Метод Монте-Карло (отбор по значимости)
def monte_carlo_sampling(theta, M, N):
    # Вспомогательное распределение (нормальное)
    Y = np.random.normal(0, 1, M)  # Генерация из N(0, 1)

    # Плотность p(X|theta)
    p_Y = (np.sqrt(theta) / np.sqrt(np.pi)) * np.exp(-theta * Y**2)

    # Плотность вспомогательного распределения q(X) = N(0, 1)
    q_Y = np.exp(-(Y**2) / 2) / np.sqrt(2 * np.pi)

    # Веса
    w = p_Y / q_Y
    w_normalized = w / np.sum(w)  # Нормализация весов

    # Ресемплизация (взвешенная выборка)
    X = np.random.choice(Y, size=N, p=w_normalized)
    return X


# Генерация выборки
X = monte_carlo_sampling(theta_hat, M, N)

# Визуализация выборки
plt.hist(X, bins=30, density=True, alpha=0.6, color="g", label="Гистограмма выборки")

# Теоретическая плотность
x_values = np.linspace(-5, 5, 1000)
theoretical_density = (np.sqrt(theta_hat) / np.sqrt(np.pi)) * np.exp(
    -theta_hat * x_values**2
)
plt.plot(x_values, theoretical_density, "r-", label="Теоретическая плотность")

# Настройка графика
plt.xlabel("X")
plt.ylabel("Плотность")
plt.title("Генерация выборки методом Монте-Карло")
plt.legend()
plt.grid(True)
plt.show()
