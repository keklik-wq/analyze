import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Загрузка данных
data = pd.read_csv("heart.csv")
DEF = []
class aaa:
    

# 1. Описание данных
def describe_data(data):
    print("Описание данных:\n")
    print(f"Количество наблюдений: {data.shape[0]}")
    print(f"Количество переменных: {data.shape[1]}")
    print("\nТипы переменных:")
    print(data.dtypes)
    print("\nПервые 5 строк данных:")
    print(data.head())

describe_data(data)

# 2. Статистические характеристики
def calculate_statistics(data):
    print("\nСтатистические характеристики:\n")
    print(data.describe())

calculate_statistics(data)

# 3. Графики численных переменных
def plot_numerical_variables(data, variables):
    for var in variables:
        plt.figure(figsize=(15, 10))

        # Эмпирическая функция распределения (ECDF)
        plt.subplot(2, 3, 1)
        sns.ecdfplot(data[var])
        plt.title(f'ECDF для {var}')

        # Гистограмма
        plt.subplot(2, 3, 2)
        plt.hist(data[var], bins=20)
        plt.title(f'Гистограмма для {var}')

        # Ядерная оценка плотности (KDE)
        plt.subplot(2, 3, 3)
        sns.kdeplot(data[var])
        plt.title(f'KDE для {var}')

        # Ящик с усами
        plt.subplot(2, 3, 4)
        sns.boxplot(x=data[var])
        plt.title(f'Ящик с усами для {var}')

        # Выборочная квантиль-нормальная квантиль
        plt.subplot(2, 3, 5)
        stats.probplot(data[var], dist="norm", plot=plt)
        plt.title(f'Q-Q plot для {var}')

        plt.tight_layout()
        plt.show()

numerical_vars = ['age', 'resting blood pressure', 'serum cholestoral', 'maximum heart rate', 'oldpeak']
plot_numerical_variables(data, numerical_vars)

# 4. Графики категориальных переменных
def plot_categorical_variables(data, variables):
    for var in variables:
        plt.figure(figsize=(8, 6))
        data[var].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title(f'Распределение {var}')
        plt.ylabel('')
        plt.show()

categorical_vars = ['sex', 'chest pain type', 'fasting blood sugar', 'resting electrocardiographic results',
                    'exercise induced angina', 'number of major vessels', 'thal', 'desease']
plot_categorical_variables(data, categorical_vars)

# 5. Корреляционная матрица
def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Корреляционная матрица')
    plt.show()

plot_correlation_matrix(data)


# 6. Парные графики рассеяния (pairplot)
def plot_pairplot(data, variables):
    sns.pairplot(data[variables + ['desease']], hue='desease', diag_kind='kde', markers='+')
    plt.suptitle('Парные графики рассеяния с разделением по наличию заболевания', y=1.02)
    plt.show()

plot_pairplot(data, numerical_vars)

# 7. Связь категориальной и непрерывной переменной
def plot_categorical_vs_continuous(data, categorical_vars, continuous_vars):
    for cat_var in categorical_vars:
        for cont_var in continuous_vars:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=cat_var, y=cont_var, data=data)
            plt.title(f'Ящик с усами: {cont_var} по {cat_var}')
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.violinplot(x=cat_var, y=cont_var, data=data)
            plt.title(f'Violin plot: {cont_var} по {cat_var}')
            plt.show()

categorical_vars = ['sex', 'chest pain type', 'fasting blood sugar', 'resting electrocardiographic results',
                    'exercise induced angina', 'number of major vessels', 'thal']
plot_categorical_vs_continuous(data, categorical_vars, numerical_vars)

# 8. Условные графики (coplot) - эмуляция с facet grid
def plot_coplot_emulation(data, cont_var1, cont_var2, cat_var1, cat_var2):
    g = sns.FacetGrid(data, col=cat_var1, row=cat_var2, margin_titles=True)
    g.map(plt.scatter, cont_var1, cont_var2, edgecolor="w")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Зависимость {cont_var2} от {cont_var1} по {cat_var1} и {cat_var2}')
    plt.show()

plot_coplot_emulation(data, 'age', 'maximum heart rate', 'sex', 'chest pain type')
plot_coplot_emulation(data, 'age', 'serum cholestoral', 'fasting blood sugar', 'exercise induced angina')
plot_coplot_emulation(data, 'resting blood pressure', 'oldpeak', 'number of major vessels', 'thal')