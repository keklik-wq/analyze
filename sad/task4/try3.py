import numpy as np
from scipy.stats import kstest, chi2
from scipy.special import gamma, digamma, polygamma
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.stats import gaussian_kde


def estimate_theta(theta_0, samples):
    """
    Оценивает параметр θ гамма-распределения методом максимального правдоподобия.

    :param theta_0: начальное приближение для решения уравнения
    :param samples: выборка данных из гамма-распределения
    :return: оценка максимального правдоподобия θ^
    """
    mean_log_sample = np.mean(np.log(samples))

    def equation(theta):
        return digamma(theta) - mean_log_sample

    result = root(equation, x0=theta_0)

    if result.success:
        theta_hat = result.x[0]
        return theta_hat
    else:
        raise ValueError("Численное решение не удалось найти.")


def pdf(X, theta):
    """
    Вычисляет плотность вероятности гамма-распределения.

    Параметры:
    X : float или numpy.ndarray
        Значение (или массив значений), для которого вычисляется плотность.
    theta : float
        Параметр формы (shape parameter) гамма-распределения.

    Возвращает:
    float или numpy.ndarray
        Значение плотности вероятности для заданных X и theta.
    """
    X = np.array(X)
    if (X <= 0).any() or theta <= 0:
        raise ValueError("X и theta должны быть больше 0")

    return (X ** (theta - 1) / gamma(theta)) * np.exp(-X)


def proposal_distribution(x_min, x_max):
    return np.random.uniform(x_min, x_max)


def generate_sample(theta_0, x_min, x_max, num_samples):
    M = np.max(pdf(np.linspace(x_min, x_max, 1000), theta_0))
    samples = np.array([])

    while len(samples) < num_samples:
        x = proposal_distribution(x_min, x_max)
        u = np.random.uniform(0, 1)

        if u <= pdf(x, theta_0) / M:
            samples = np.append(samples, x)
    return samples


def likelihood_ratio_test(sample_X, sample_Y, theta_0):
    theta_x_hat = estimate_theta(theta_0, sample_X)
    theta_y_hat = estimate_theta(theta_0, sample_Y)

    log_likelihood_0 = np.sum(
        np.log(sample_X) * (theta_0 - 1) - sample_X - np.log(gamma(theta_0))
    ) + np.sum(np.log(sample_Y) * (theta_0 - 1) - sample_Y - np.log(gamma(theta_0)))

    log_likelihood_1 = np.sum(
        np.log(sample_X) * (theta_x_hat - 1) - sample_X - np.log(gamma(theta_x_hat))
    ) + np.sum(
        np.log(sample_Y) * (theta_y_hat - 1) - sample_Y - np.log(gamma(theta_y_hat))
    )

    test_statistic = 2 * (log_likelihood_1 - log_likelihood_0)

    p_value = 1 - chi2.cdf(
        test_statistic, df=2
    )  # Хи-квадрат распределение с 2 степенями свободы

    return test_statistic, p_value


def fisher_information(theta):
    return polygamma(1, theta)


def additional_statistics(sample_X, sample_Y, theta_0_x, theta_0_y):
    n = len(sample_X)
    m = len(sample_Y)

    theta_x_hat = estimate_theta(theta_0_x, sample_X)
    theta_y_hat = estimate_theta(theta_0_y, sample_Y)

    theta_mle = (theta_x_hat + theta_y_hat) / 2

    I_theta_mle = fisher_information(theta_mle)
    I_theta_0 = fisher_information(theta_0_x)

    stat1 = n * (theta_mle - theta_0_x) * I_theta_mle * (theta_mle - theta_0_x) + m * (
        theta_mle - theta_0_y
    ) * I_theta_mle * (theta_mle - theta_0_y)

    stat2 = n * (theta_mle - theta_0_x) * I_theta_0 * (theta_mle - theta_0_x) + m * (
        theta_mle - theta_0_y
    ) * I_theta_0 * (theta_mle - theta_0_y)

    return stat1, stat2


theta_0 = 1.3
theta_0_x = theta_0
theta_0_y = theta_0
N = 50
M = 50
alpha = 0.1
x_min = 0.1
x_max = 10

np.random.seed(0)
sample_X = generate_sample(theta_0, x_min, x_max, N)
sample_Y = generate_sample(theta_0, x_min, x_max, M)

llr_statistic, llr_p_value = likelihood_ratio_test(sample_X, sample_Y, theta_0)
print(
    f"Статистика отношения правдоподобия: {llr_statistic:.4f}, p-значение: {llr_p_value:.4f}"
)

ks_statistic, ks_p_value = kstest(sample_X, sample_Y)
print(
    f"Статистика Колмогорова-Смирнова: {ks_statistic:.4f}, p-значение: {ks_p_value:.4f}"
)

stat_mle, stat_0 = additional_statistics(sample_X, sample_Y, theta_0_x, theta_0_y)
p_value_stat_mle = 1 - chi2.cdf(stat_mle, df=2)
p_value_stat_0 = 1 - chi2.cdf(stat_0, df=2)
print(f"Статистика максимального правдоподобия theta_mle: {stat_mle:.4f}")
print(f"Статистика максимального правдоподобия theta_0: {stat_0:.4f}")

if p_value_stat_mle < alpha:
    print(
        "Отвергаем нулевую гипотезу на уровне значимости 0.1 (критерий максимального правдоподобия theta_mle)"
    )
else:
    print(
        "Не отвергаем нулевую гипотезу на уровне значимости 0.1 (критерий максимального правдоподобия theta_mle)"
    )

if p_value_stat_0 < alpha:
    print(
        "Отвергаем нулевую гипотезу на уровне значимости 0.1 (критерий максимального правдоподобия theta_0)"
    )
else:
    print(
        "Не отвергаем нулевую гипотезу на уровне значимости 0.1 (критерий максимального правдоподобия theta_0)"
    )

if llr_p_value < alpha:
    print(
        "Отвергаем нулевую гипотезу на уровне значимости 0.1 (критерий отношения правдоподобия)"
    )
else:
    print(
        "Не отвергаем нулевую гипотезу на уровне значимости 0.1 (критерий отношения правдоподобия)"
    )

if ks_p_value < alpha:
    print(
        "Отвергаем нулевую гипотезу на уровне значимости 0.1 (критерий Колмогорова-Смирнова)"
    )
else:
    print(
        "Не отвергаем нулевую гипотезу на уровне значимости 0.1 (критерий Колмогорова-Смирнова)"
    )

plt.figure(figsize=(10, 6))
plt.hist(
    sample_X,
    bins=20,
    density=True,
    alpha=0.5,
    color="blue",
    label="Гистограмма выборки X",
)
plt.hist(
    sample_Y,
    bins=20,
    density=True,
    alpha=0.5,
    color="red",
    label="Гистограмма выборки Y",
)

kde_X = gaussian_kde(sample_X)
kde_Y = gaussian_kde(sample_Y)
x_grid = np.linspace(x_min, x_max, 1000)
plt.plot(x_grid, kde_X(x_grid), color="blue", label="Ядерная оценка плотности X")
plt.plot(x_grid, kde_Y(x_grid), color="red", label="Ядерная оценка плотности Y")

plt.xlabel("x")
plt.ylabel("Плотность")
plt.title("Ядерная оценка плотности сгенерированной выборки")
plt.legend()
plt.show()
