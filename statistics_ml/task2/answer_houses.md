1. Бинарная логистическая регрессия для каждого уровня ['ExterQual']:

Уровень ['ExterQual'] = Ex:
- Точность (Accuracy): 0.9349
- Классы для уровня Ex: [1126   42]
- Отчёт:               precision    recall  f1-score   support

           0       0.99      0.94      0.97       282
           1       0.30      0.70      0.42        10

    accuracy                           0.93       292
   macro avg       0.65      0.82      0.69       292
weighted avg       0.97      0.93      0.95       292

- Коэффициенты модели (первые 5): [-1.15432613  0.18174662 -0.87407733  2.1294569   0.16194832]...
- Свободный член (Intercept): -8.437273509861242

Уровень ['ExterQual'] = Fa:
- Точность (Accuracy): 0.9658
- Классы для уровня Fa: [1157   11]
- Отчёт:               precision    recall  f1-score   support

           0       0.99      0.98      0.98       289
           1       0.00      0.00      0.00         3

    accuracy                           0.97       292
   macro avg       0.49      0.49      0.49       292
weighted avg       0.98      0.97      0.97       292

- Коэффициенты модели (первые 5): [ 0.73110525 -0.35219886  0.44595807 -1.81116644 -0.52601904]...
- Свободный член (Intercept): -13.169419893134688

Уровень ['ExterQual'] = Gd:
- Точность (Accuracy): 0.8801
- Классы для уровня Gd: [778 390]
- Отчёт:               precision    recall  f1-score   support

           0       0.93      0.88      0.91       194
           1       0.79      0.88      0.83        98

    accuracy                           0.88       292
   macro avg       0.86      0.88      0.87       292
weighted avg       0.89      0.88      0.88       292

- Коэффициенты модели (первые 5): [ 0.18867533 -0.0445105   0.62242885  1.06859685 -0.54974902]...
- Свободный член (Intercept): -0.8419704206950845

Уровень ['ExterQual'] = TA:
- Точность (Accuracy): 0.9144
- Классы для уровня TA: [443 725]
- Отчёт:               precision    recall  f1-score   support

           0       0.86      0.92      0.89       111
           1       0.95      0.91      0.93       181

    accuracy                           0.91       292
   macro avg       0.91      0.92      0.91       292
weighted avg       0.92      0.91      0.91       292

- Коэффициенты модели (первые 5): [-0.23240236  0.12506564 -0.78729214 -0.76611824  0.58884475]...
- Свободный член (Intercept): 0.34234704789883313




2. Пробит-регрессия для каждого уровня ['ExterQual']:

Уровень ['ExterQual'] = Ex:
- Классы для уровня Ex: [1126   42]
- Точность (Accuracy): 0.9555
- Отчёт:               precision    recall  f1-score   support

           0       0.98      0.98      0.98       282
           1       0.36      0.40      0.38        10

    accuracy                           0.96       292
   macro avg       0.67      0.69      0.68       292
weighted avg       0.96      0.96      0.96       292

- Коэффициенты модели (первые 5): [0.         0.         0.         0.         0.02558171]...
- Свободный член (Intercept): -2.343083158862432

Уровень ['ExterQual'] = Fa:
- Классы для уровня Fa: [1157   11]
- Точность (Accuracy): 0.9863
- Отчёт:               precision    recall  f1-score   support

           0       0.99      1.00      0.99       289
           1       0.00      0.00      0.00         3

    accuracy                           0.99       292
   macro avg       0.49      0.50      0.50       292
weighted avg       0.98      0.99      0.98       292

- Коэффициенты модели (первые 5): [ 0.          0.          0.         -0.07012772  0.        ]...
- Свободный член (Intercept): -2.5377106112389947

Уровень ['ExterQual'] = Gd:
- Классы для уровня Gd: [778 390]
- Точность (Accuracy): 0.9041
- Отчёт:               precision    recall  f1-score   support

           0       0.92      0.94      0.93       194
           1       0.88      0.83      0.85        98

    accuracy                           0.90       292
   macro avg       0.90      0.88      0.89       292
weighted avg       0.90      0.90      0.90       292

- Коэффициенты модели (первые 5): [ 0.02578772 -0.00246718  0.34800725  0.44285094  0.        ]...
- Свободный член (Intercept): -0.6728873829325959

Уровень ['ExterQual'] = TA:
- Классы для уровня TA: [443 725]
- Точность (Accuracy): 0.9075
- Отчёт:               precision    recall  f1-score   support

           0       0.90      0.85      0.87       111
           1       0.91      0.94      0.93       181

    accuracy                           0.91       292
   macro avg       0.91      0.90      0.90       292
weighted avg       0.91      0.91      0.91       292

- Коэффициенты модели (первые 5): [-0.04834575  0.05342337 -0.37911481 -0.34872797  0.        ]...
- Свободный член (Intercept): 0.401980293929742
Формирование предсказаний по бинарным моделям...

Результаты оценки:
Логистическая регрессия — Accuracy: 0.8630, Log-Likelihood: -0.9091, F1: 0.5367
Пробит-регрессия — Accuracy: 0.8733, Log-Likelihood: -1.1137, F1: 0.5373
Лучшая модель по правдоподобию: Логистическая регрессия

Multinomial Logistic Regression
- Точность (Accuracy): 0.8493
- Отчёт:               precision    recall  f1-score   support

           0       0.42      0.80      0.55        10
           1       0.20      0.67      0.31         3
           2       0.84      0.83      0.83       100
           3       0.95      0.87      0.90       179

    accuracy                           0.85       292
   macro avg       0.60      0.79      0.65       292
weighted avg       0.88      0.85      0.86       292

- Коэффициенты модели (первые 5): [-0.90329614  0.09991599 -0.09896944  2.58367503 -0.17505025]...
- Свободный член (Intercept): -3.1603422911511445



Gamma and Gaussian Regressions
- Отчёт по гамме: mse - 0.5644404856376956 mae - 0.418170737032156
- Отчёт по гауссиану: mse - 0.23100542614962957 mae - 0.34028270993539333